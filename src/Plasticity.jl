# * Event-driven STDP --- pair-based spike-timing-dependent plasticity with analytic
# between-spike trace decay. A plastic projection wraps a base synapse state (CUBA/COBA/delta ---
# the transmission is orthogonal to the learning) and adds its OWN mutable per-edge weight array
# (CSR-parallel), leaving the shared `SparseCSR` immutable (so the same connectome stays sharable,
# and the batched path's one-CSR-across-B broadcast is never violated). Two per-neuron eligibility
# traces (`x_pre`, `x_post`) decay exponentially and are bumped on spikes.
#
# The rule rides the existing edge scatter: a thread owns edge `e`, so the weight write needs NO
# atomic (only the ring deposit keeps its atomic). On a PRE spike it deposits the current weight
# AND depresses by `Aminus·x_post[post]`; on a POST spike it potentiates by `Aplus·x_pre[pre]`.
# Plasticity uses ACTUAL spike times (`integ.spiked` this step); the conduction delay is purely a
# transmission detail. Trace decay folds into :integrate (broadcast `_decay!` / fused `_syn_one`),
# the bump into :propagate, so the decay→update→bump order falls out of the phase order with NO new
# schedule phase --- and the fused fast path stays engaged for non-plastic networks.

using KernelAbstractions: @kernel, @index, @Const, get_backend
import KernelAbstractions as _KA
import Atomix

"""
    AbstractPlasticityRule

A synaptic learning rule. Attached to a [`Projection`](@ref) via `plasticity =`; turns its
weights mutable and updates them on spikes. The canonical rule is [`STDP`](@ref).
"""
abstract type AbstractPlasticityRule end
Base.Broadcast.broadcastable(r::AbstractPlasticityRule) = Ref(r)

"""
    STDP(; Aplus, Aminus, τplus, τminus, wmin=-Inf, wmax=Inf)

Pair-based additive STDP. A post-after-pre pairing potentiates by `Aplus·exp(-Δt/τplus)`, a
pre-after-post pairing depresses by `Aminus·exp(-Δt/τminus)`; weights are clamped to `[wmin, wmax]`.
"""
struct STDP{T} <: AbstractPlasticityRule
    Aplus::T
    Aminus::T
    τplus::T
    τminus::T
    wmin::T
    wmax::T
end
function STDP(; Aplus, Aminus, τplus, τminus, wmin = -Inf, wmax = Inf)
    return STDP(promote(Aplus, Aminus, to_time(τplus), to_time(τminus), float(wmin), float(wmax))...)
end
export STDP

@inline trace_decay_pre(r::STDP, dt) = exp(-dt / r.τplus)
@inline trace_decay_post(r::STDP, dt) = exp(-dt / r.τminus)

# A plastic projection's runtime state: the base synapse state (transmission), the MUTABLE per-edge
# weights, the pre/post eligibility traces, and the precomputed per-step trace decays.
struct PlasticState{B <: AbstractSynapseState, W, XP, XO, R, T} <: AbstractSynapseState
    base::B
    weight::W      # (nedges,) mutable per-edge weights (initialised from conn.weight)
    x_pre::XP      # (npre,)  presynaptic eligibility trace (decays at τplus)
    x_post::XO     # (npost,) postsynaptic eligibility trace (decays at τminus)
    rule::R
    dpre::T        # exp(-dt/τplus)
    dpost::T       # exp(-dt/τminus)
end
Adapt.@adapt_structure PlasticState

# --- build the runtime state (called from `init`, dispatched on the projection's plasticity) ---
_make_synstate(arch, syn, conn, ::Nothing, ::Type{T}, N, dt) where {T} = _make_synstate(arch, syn, conn, T, N, dt)
function _make_synstate(arch, syn, conn, rule::AbstractPlasticityRule, ::Type{T}, N, dt) where {T}
    npre(conn) == npost(conn) ||
        throw(ArgumentError("STDP needs a recurrent projection (npre == npost); got $(npre(conn)) vs $(npost(conn))"))
    base = _make_synstate(arch, syn, conn, T, N, dt)
    w = copy(conn.weight)                                  # mutable working copy on the architecture
    xpre = fill!(allocate(arch, T, npre(conn)), zero(T))
    xpost = fill!(allocate(arch, T, npost(conn)), zero(T))
    return PlasticState(base, w, xpre, xpost, rule, T(trace_decay_pre(rule, dt)), T(trace_decay_post(rule, dt)))
end

# --- phase dispatch: transmission delegates to the base; learning is layered on ---
@inline _deliver!(syn::PlasticState, integ) = _deliver!(syn.base, integ)
@inline _accumulate!(syn::PlasticState, gtot, itot, V) = _accumulate!(syn.base, gtot, itot, V)
# broadcast-path decay (:integrate): decay the base synaptic state AND the eligibility traces
@inline function _decay!(syn::PlasticState)
    _decay!(syn.base)
    @. syn.x_pre *= syn.dpre
    @. syn.x_post *= syn.dpost
    return nothing
end
# fused-path per-neuron contribution: decay this neuron's traces in-kernel, then the base's
# deliver+accumulate+decay (so the trace decay matches the broadcast path; once per neuron per step)
@inline function _syn_one(s::PlasticState, i, n, v, gtot, itot)
    @inbounds s.x_pre[i] *= s.dpre
    @inbounds s.x_post[i] *= s.dpost
    return _syn_one(s.base, i, n, v, gtot, itot)
end

# --- the plastic scatter: deposit (current weight) + STDP weight update, one thread per edge ---
@kernel function _plastic_scatter_kernel!(
        slots, weight, @Const(spiked), @Const(src), @Const(post), @Const(delay),
        @Const(x_pre), @Const(x_post), Aplus, Aminus, wmin, wmax, now, L,
    )
    e = @index(Global)
    @inbounds begin
        pre = src[e]
        po = post[e]
        w = weight[e]
        if spiked[pre]
            slot = mod(now + delay[e], L) + 1
            Atomix.@atomic slots[po, slot] += w           # transmission: deposit the CURRENT weight
            w -= Aminus * x_post[po]                       # depression on the pre-spike
        end
        if spiked[po]
            w += Aplus * x_pre[pre]                        # potentiation on the post-spike
        end
        weight[e] = clamp(w, wmin, wmax)                   # per-edge write: single owner, no atomic
    end
end

# Dispatched on the KernelAbstractions backend (the same split Fused.jl uses): native CPU takes the
# serial fast path; every device backend (CUDA, the JLArrays reference GPU) takes the edge kernel.
# Backend dispatch avoids the parametric-method ambiguity of constraining the weight storage type.
@inline plastic_scatter!(syn::PlasticState, spiked::AbstractArray, now::Integer) =
    _plastic_scatter!(get_backend(syn.weight), syn, spiked, Int(now))

# device path: the edge-parallel kernel (one thread per synapse)
function _plastic_scatter!(backend, syn::PlasticState, spiked, now::Int)
    base = syn.base
    buf, conn, rule = base.buf, base.conn, syn.rule
    ne = length(syn.weight)
    if ne > 0
        _plastic_scatter_kernel!(backend)(
            buf.slots, syn.weight, spiked, conn.src, conn.post, conn.delay,
            syn.x_pre, syn.x_post, rule.Aplus, rule.Aminus, rule.wmin, rule.wmax, now, buf.L;
            ndrange = ne,
        )
    end
    return nothing
end

# CPU fast path: a serial per-edge walk (no KA launch round-trip → allocation-free, deterministic).
# Each edge is visited once, so the weight write order is fixed and the ring deposit accumulates in a
# fixed edge order (bit-reproducible, no atomics) --- matching the device kernel for exact weights.
function _plastic_scatter!(::_KA.CPU, syn::PlasticState, spiked, now::Int)
    base = syn.base
    buf, conn, rule = base.buf, base.conn, syn.rule
    slots, L, w = buf.slots, buf.L, syn.weight
    src, post, delay = conn.src, conn.post, conn.delay
    xpre, xpost = syn.x_pre, syn.x_post
    Ap, Am, lo, hi = rule.Aplus, rule.Aminus, rule.wmin, rule.wmax
    @inbounds for e in eachindex(w)
        pre, po = src[e], post[e]
        we = w[e]
        if spiked[pre]
            slots[po, mod(now + delay[e], L) + 1] += we    # deposit the current weight
            we -= Am * xpost[po]                            # depression on the pre-spike
        end
        spiked[po] && (we += Ap * xpre[pre])               # potentiation on the post-spike
        w[e] = clamp(we, lo, hi)
    end
    return nothing
end

# bump the eligibility traces for THIS step's spikes (after the scatter has read the decayed traces)
@inline function _bump_traces!(syn::PlasticState, spiked)
    @. syn.x_pre += spiked
    @. syn.x_post += spiked
    return nothing
end

# the :propagate hook for a plastic projection (overrides the base `_propagate_nosync!`): scatter +
# weight update, then the trace bump. Used by BOTH the broadcast and fused paths.
@inline function _propagate_nosync!(syn::PlasticState, integ)
    plastic_scatter!(syn, integ.spiked, integ.n)
    _bump_traces!(syn, integ.spiked)
    return nothing
end
