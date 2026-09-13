# * Event-driven STDP: pair-based spike-timing-dependent plasticity with analytic
# between-spike trace decay. A plastic projection wraps a base synapse state (CUBA/COBA/delta:
# the transmission is orthogonal to the learning) and adds its own mutable per-edge weight array
# (CSR-parallel), leaving the shared `SparseCSR` immutable (so the same connectome stays sharable,
# and the batched path's one-CSR-across-B broadcast is never violated). Two per-neuron eligibility
# traces (`x_pre`, `x_post`) decay exponentially and are bumped on spikes.
#
# The rule rides the existing edge scatter: a thread owns edge `e`, so the weight write needs no
# atomic (only the ring deposit keeps its atomic). On a pre spike it deposits the current weight
# and depresses by `Aminus·x_post[post]`; on a post spike it potentiates by `Aplus·x_pre[pre]`.
# Plasticity uses actual spike times (`integ.spiked` this step); the conduction delay is purely a
# transmission detail. Trace decay folds into :integrate (broadcast `_decay!` / fused `_syn_one`),
# the bump into :propagate, so the decay→update→bump order falls out of the phase order with no new
# schedule phase; the fused fast path stays engaged for non-plastic networks.

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

# The traces are indexed by the flat neuron index (the fused step reads `x_pre[i]` for every
# `i in 1:N`), so they must span the population, not just the projection.
function _check_recurrent_full(rule, conn, N)
    (npre(conn) == npost(conn) == N) || throw(
        ArgumentError(
            "$rule needs a recurrent projection over the whole population (npre == npost == N); " *
                "got npre = $(npre(conn)), npost = $(npost(conn)), N = $N"
        )
    )
    return nothing
end

"""
    STDP(; Aplus, Aminus, τplus, τminus, wmin=-Inf, wmax=Inf)

Pair-based additive STDP. A post-after-pre pairing potentiates by `Aplus·exp(-Δt/τplus)`, a
pre-after-post pairing depresses by `Aminus·exp(-Δt/τminus)`; weights are clamped to `[wmin, wmax]`.

Pairings are timed at the two somata: `Δt = t_post - t_pre`, with the transmission `delay` playing no
part. A synapse physically sits between the somata, so the interval it sees is
`(t_post + d_dendritic) - (t_pre + d_axonal)`, and a simulator has to split the delay between the
axon and the back-propagating dendrite. Simulators disagree about that split, in both directions:
NEST's `stdp_synapse` assigns the whole delay to the dendrite (`Δt + d`), while Brian2 and BrainPy run
the depression from the arriving spike, assigning it all to the axon (`Δt - d`). The convention here
is the midpoint of those two, and matches NEST GPU. Expect weights to differ from any of them on a
projection whose delay is an appreciable fraction of `τplus`/`τminus`.

Splitting the delay explicitly (as GeNN does) would need the plasticity event to travel the delay ring
alongside the spike, or a per-edge trace.
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

# A plastic projection's runtime state: the base synapse state (transmission), the mutable per-edge
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

# A source model builds a state whose event generator a plastic wrapper would hide from
# `_synprestep!`, silently killing the drive, so every plasticity builder checks before wrapping.
# Derived from `is_source_synapse` rather than restated per type (see Synapses.jl).
_plastic_wrappable(s::AbstractSynapseModel) = !is_source_synapse(s)
_check_plastic_wrappable(syn) = _plastic_wrappable(syn) ||
    throw(ArgumentError("plasticity is not supported on an external drive model ($(nameof(typeof(syn)))); attach it to a network projection"))

# build the runtime state (called from `init`, dispatched on the projection's plasticity)
_make_synstate(arch, syn, conn, ::Nothing, ::Type{T}, N, dt) where {T} = _make_synstate(arch, syn, conn, T, N, dt)
function _make_synstate(arch, syn, conn, rule::AbstractPlasticityRule, ::Type{T}, N, dt) where {T}
    _check_plastic_wrappable(syn)
    _check_recurrent_full("STDP", conn, N)
    base = _make_synstate(arch, syn, conn, T, N, dt)
    w = clamp.(conn.weight, T(rule.wmin), T(rule.wmax))    # mutable working copy on the architecture, in bounds
    xpre = fill!(allocate(arch, T, npre(conn)), zero(T))
    xpost = fill!(allocate(arch, T, npost(conn)), zero(T))
    return PlasticState(base, w, xpre, xpost, rule, T(trace_decay_pre(rule, dt)), T(trace_decay_post(rule, dt)))
end

# phase dispatch: transmission delegates to the base; learning is layered on
@inline _deliver!(syn::PlasticState, integ) = _deliver!(syn.base, integ)
@inline _deliver_jump!(syn::PlasticState, integ) = _deliver_jump!(syn.base, integ)
@inline _accumulate!(syn::PlasticState, gtot, itot, V) = _accumulate!(syn.base, gtot, itot, V)
# broadcast-path decay (:integrate): decay the base synaptic state and the eligibility traces
@inline function _decay!(syn::PlasticState)
    _decay!(syn.base)
    @. syn.x_pre *= syn.dpre
    @. syn.x_post *= syn.dpost
    return nothing
end
# fused-path per-neuron contribution: decay this neuron's traces in-kernel, then the base's
# deliver+accumulate+decay (so the trace decay matches the broadcast path; once per neuron per step)
@inline function _syn_one(s::PlasticState, i, n, v0, vj, gtot, itot)
    @inbounds s.x_pre[i] *= s.dpre
    @inbounds s.x_post[i] *= s.dpost
    return _syn_one(s.base, i, n, v0, vj, gtot, itot)
end

# the plastic scatter: deposit (current weight) + STDP weight update, one thread per edge
@kernel function _plastic_scatter_kernel!(
        slots, weight, @Const(spiked), @Const(src), @Const(post), @Const(delay),
        @Const(x_pre), @Const(x_post), Aplus, Aminus, wmin, wmax, now, L, scale,
    )
    e = @index(Global)
    @inbounds begin
        pre = src[e]
        po = post[e]
        w = weight[e]
        if spiked[pre]
            slot = mod(now + delay[e], L) + 1
            Atomix.@atomic slots[po, slot] += _fp_quantise(eltype(slots), w, scale)   # transmission: deposit the current weight
            w -= Aminus * x_post[po]                       # depression on the pre-spike
        end
        if spiked[po]
            w += Aplus * x_pre[pre]                        # potentiation on the post-spike
        end
        # Only a spiking edge can have changed, and a dense write of every edge every step is the
        # bandwidth cost of the kernel at biological rates.
        if spiked[pre] | spiked[po]
            weight[e] = clamp(w, wmin, wmax)               # per-edge write: single owner, no atomic
        end
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
            syn.x_pre, syn.x_post, rule.Aplus, rule.Aminus, rule.wmin, rule.wmax, now, buf.L, buf.scale;
            ndrange = ne,
        )
    end
    return nothing
end

# CPU fast path: a serial per-edge walk (no KA launch round-trip → allocation-free, deterministic).
# Each edge is visited once, so the weight write order is fixed and the ring deposit accumulates in a
# fixed edge order (bit-reproducible, no atomics): matching the device kernel for exact weights.
function _plastic_scatter!(::_KA.CPU, syn::PlasticState, spiked, now::Int)
    base = syn.base
    buf, conn, rule = base.buf, base.conn, syn.rule
    slots, L, scale, w = buf.slots, buf.L, buf.scale, syn.weight
    src, post, delay = conn.src, conn.post, conn.delay
    xpre, xpost = syn.x_pre, syn.x_post
    Ap, Am, lo, hi = rule.Aplus, rule.Aminus, rule.wmin, rule.wmax
    @inbounds for e in eachindex(w)
        pre, po = src[e], post[e]
        we = w[e]
        if spiked[pre]
            slots[po, mod(now + delay[e], L) + 1] += _fp_quantise(eltype(slots), we, scale)   # deposit the current weight
            we -= Am * xpost[po]                            # depression on the pre-spike
        end
        spiked[po] && (we += Ap * xpre[pre])               # potentiation on the post-spike
        (spiked[pre] | spiked[po]) && (w[e] = clamp(we, lo, hi))   # quiet edges cannot have changed
    end
    return nothing
end

# bump the eligibility traces for this step's spikes (after the scatter has read the decayed traces)
@inline function _bump_traces!(syn::PlasticState, spiked)
    @. syn.x_pre += spiked
    @. syn.x_post += spiked
    return nothing
end

# the :propagate hook for a plastic projection (overrides the base `_propagate_nosync!`): scatter +
# weight update, then the trace bump. Used by both the broadcast and fused paths.
@inline function _propagate_nosync!(syn::PlasticState, integ)
    plastic_scatter!(syn, integ.spiked, integ.n)
    _bump_traces!(syn, integ.spiked)
    return nothing
end

# * Short-term depression (Tsodyks-Markram), riding the same machinery.
# `PlasticState` already carries a per-presynaptic-neuron trace updated on spikes through the edge
# scatter; STD reuses it with the trace reinterpreted as the resource x ∈ (0,1]: recovery to 1
# replaces decay to 0, the deposit is w·x rather than w, and the pre-spike update is depletion of x
# rather than a weight write. Weights stay static, so the shared-CSR batched path can carry STD
# too: it only needs a per-member (N,B) resource, never per-column weights.

"""
    STD(; U = 0.2, τD = 400.0)

Tsodyks-Markram short-term synaptic depression: a presynaptic spike transmits `w·x_pre` and then
depletes the presynaptic resource, `x ← (1-U)·x`, which recovers towards 1 as `ẋ = (1-x)/τD` (ms).
Attach to a projection via `plasticity = STD(...)`.

Depression makes the recurrent gain rate-dependent, `λ_eff = λ/(1 + U·ν·τD)`: an intrinsic
saturation that can sustain balanced activity without external drive (Vinci, Angulo-Garcia &
Torcini 2025); silence stays absorbing throughout.

Weights are never mutated, so unlike [`STDP`](@ref) the rule is supported on the batched
(ensemble) path with `scatter = :edge` (the default).
"""
struct STD{T} <: AbstractPlasticityRule
    U::T
    τD::T
end
STD(; U = 0.2, τD = 400.0) = STD(promote(float(U), to_time(τD))...)
export STD

const STDState = PlasticState{<:AbstractSynapseState, <:Any, <:Any, <:Any, <:STD}

# batched-support trait: STDP needs per-column (nedges,B) weights, which the shared-CSR batch
# forbids; STD only needs the (N,B) resource, so it opts in.
_batch_plastic_supported(::AbstractPlasticityRule) = false
_batch_plastic_supported(::STD) = true

# The resource starts full, so STDP's `_make_synstate` (traces at zero) cannot be inherited.
function _make_synstate(arch, syn, conn, rule::STD, ::Type{T}, N, dt) where {T}
    _check_plastic_wrappable(syn)
    _check_recurrent_full("STD", conn, N)
    base = _make_synstate(arch, syn, conn, T, N, dt)
    x = fill!(allocate(arch, T, npre(conn)), one(T))
    xpost = fill!(allocate(arch, T, npost(conn)), zero(T))   # unused
    return PlasticState(base, copy(conn.weight), x, xpost, rule, T(exp(-dt / rule.τD)), one(T))
end

# Recovery 1 − (1−x)·e^(−dt/τD) replaces the trace decay, on both engine paths.
@inline function _decay!(s::STDState)
    _decay!(s.base)
    @. s.x_pre = 1 - (1 - s.x_pre) * s.dpre
    return nothing
end
@inline function _syn_one(s::STDState, i, n, v0, vj, gtot, itot)
    @inbounds s.x_pre[i] = 1 - (1 - s.x_pre[i]) * s.dpre
    return _syn_one(s.base, i, n, v0, vj, gtot, itot)
end

# Deposit w·x, with x read before this step's depletion: the scatter runs before the bump.
function _plastic_scatter!(::_KA.CPU, s::STDState, spiked, now::Int)
    buf, conn = s.base.buf, s.base.conn
    slots, L, scale, w = buf.slots, buf.L, buf.scale, s.weight
    rowptr, post, delay = conn.rowptr, conn.post, conn.delay
    x = s.x_pre
    # Walk only spiking rows: STDP's all-edge loop is needed for post-spike updates, but STD
    # touches nothing on quiet edges, and the all-edge walk is ~50× slower at ε = 0.1.
    @inbounds for pre in eachindex(x)
        spiked[pre] || continue
        xp = x[pre]
        for e in rowptr[pre]:(rowptr[pre + 1] - 1)
            slots[post[e], mod(now + delay[e], L) + 1] +=
                _fp_quantise(eltype(slots), w[e] * xp, scale)
        end
    end
    return nothing
end

# device path: the edge-parallel scatter (the STDP kernel minus every weight/trace write)
@kernel function _std_scatter_kernel!(
        slots, @Const(weight), @Const(spiked), @Const(src), @Const(post), @Const(delay),
        @Const(x_pre), now, L, scale,
    )
    e = @index(Global)
    @inbounds begin
        pre = src[e]
        if spiked[pre]
            slot = mod(now + delay[e], L) + 1
            Atomix.@atomic slots[post[e], slot] += _fp_quantise(eltype(slots), weight[e] * x_pre[pre], scale)
        end
    end
end
function _plastic_scatter!(backend, s::STDState, spiked, now::Int)
    buf, conn = s.base.buf, s.base.conn
    ne = length(s.weight)
    if ne > 0
        _std_scatter_kernel!(backend)(
            buf.slots, s.weight, spiked, conn.src, conn.post, conn.delay,
            s.x_pre, now, buf.L, buf.scale; ndrange = ne,
        )
    end
    return nothing
end

# Depletion once per spiking neuron, not per edge: a neuron's out-edges share its resource.
@inline function _bump_traces!(s::STDState, spiked)
    @. s.x_pre *= 1 - s.rule.U * spiked
    return nothing
end

# * Batched STD: the shared-CSR ensemble batch (src/Batch.jl) plus a per-member (N,B) resource.
# Wraps the generic `BatchedSyn` (transmission untouched); recovery folds into `_bsyn_one` (called
# exactly once per (i,b) per step by the megakernel, matching the scalar integrate-before-propagate
# order), the deposit is an STD variant of the edge scatter reading pre-depletion x, and depletion
# is a GPU-safe (N,B) broadcast after the scatter.
struct BatchedSTDState{BS <: AbstractSynapseState, X, R, T} <: AbstractSynapseState
    base::BS     # BatchedSyn: (N,B) accumulators + (N,B,L) ring + shared CSR
    x::X         # (N,B) presynaptic resource, initialised to one
    rule::R
    dpre::T      # exp(-dt/τD)
end
Adapt.@adapt_structure BatchedSTDState

function _make_batched_synstate(arch, m, conn, rule::STD, ::Type{T}, N, B, dt, over) where {T}
    _check_plastic_wrappable(m)
    _check_recurrent_full("STD", conn, N)
    base = _make_batched_synstate(arch, m, conn, T, N, B, dt, over)
    x = fill!(allocate(arch, T, Int(N), Int(B)), one(T))
    return BatchedSTDState(base, x, rule, T(exp(-dt / rule.τD)))
end

# in-kernel recovery, then delegate transmission to the base
@inline function _bsyn_one(s::BatchedSTDState, i, b, n, v0, vj, gtot, itot)
    x = getfield(s, :x)
    @inbounds x[i, b] = 1 - (1 - x[i, b]) * getfield(s, :dpre)
    return _bsyn_one(getfield(s, :base), i, b, n, v0, vj, gtot, itot)
end

# STD edge scatter: `_batched_scatter_edge_kernel!` depositing weight[e]·x[pre,b]
@kernel function _batched_std_scatter_kernel!(
        slots, @Const(spiked), @Const(src), @Const(post), @Const(weight), @Const(delay),
        @Const(x), now, L, scale,
    )
    I = @index(Global, Cartesian)
    e = I[1]
    b = I[2]
    @inbounds begin
        pre = src[e]
        if spiked[pre, b]
            slot = mod(now + delay[e], L) + 1
            Atomix.@atomic slots[post[e], b, slot] += _fp_quantise(eltype(slots), weight[e] * x[pre, b], scale)
        end
    end
end
function _batched_std_scatter!(buf::BatchedRing, conn::SparseCSR, x, spiked, now::Integer)
    backend = get_backend(buf.slots)
    ne = nedges(conn)
    if ne > 0
        _batched_std_scatter_kernel!(backend)(
            buf.slots, spiked, conn.src, conn.post, conn.weight, conn.delay, x,
            Int(now), buf.L, buf.scale; ndrange = (ne, size(spiked, 2)),
        )
    end
    return nothing
end
# CPU fast path: threaded per-column row-walk (disjoint slabs, no atomics), mirroring `batched_scatter!`.
function _batched_std_scatter!(
        buf::BatchedRing{<:Array}, conn::SparseCSR{<:Array, <:Array, <:Array, <:Array},
        x::Array, spiked::AbstractArray, now::Integer,
    )
    slots, L, scale = buf.slots, buf.L, buf.scale
    rowptr, post, weight, delay = conn.rowptr, conn.post, conn.weight, conn.delay
    n = Int(now)
    N, B = size(spiked)
    if Threads.nthreads() == 1 || _scatter_work(conn, spiked) < Threads.nthreads() * _SCATTER_MIN_PER_THREAD
        @inbounds for b in 1:B, pre in 1:N
            spiked[pre, b] || continue
            xp = x[pre, b]
            for e in rowptr[pre]:(rowptr[pre + 1] - 1)
                slots[post[e], b, mod(n + delay[e], L) + 1] +=
                    _fp_quantise(eltype(slots), weight[e] * xp, scale)
            end
        end
    else
        Threads.@threads for b in 1:B
            @inbounds for pre in 1:N
                spiked[pre, b] || continue
                xp = x[pre, b]
                for e in rowptr[pre]:(rowptr[pre + 1] - 1)
                    slots[post[e], b, mod(n + delay[e], L) + 1] +=
                        _fp_quantise(eltype(slots), weight[e] * xp, scale)
                end
            end
        end
    end
    return nothing
end

# :propagate seam: scatter (reads pre-depletion x), then deplete once per spiking (neuron, member).
function _bpropagate!(syn::BatchedSTDState, integ)
    base = getfield(syn, :base)
    x = getfield(syn, :x)
    _batched_std_scatter!(base.buf, base.conn, x, integ.spiked, integ.n)
    U = getfield(syn, :rule).U
    @. x *= 1 - U * integ.spiked
    return nothing
end

_bcompacted_propagate!(syn::BatchedSTDState, active, na, max_na, now) =
    throw(ArgumentError("STD does not support scatter = :compacted; use scatter = :edge (the default)"))

# monitors: `Trace(:Isyn; projection = j)` reads the projection's `.acc`; the plastic wrappers
# (scalar and batched) keep transmission one level down, so unwrap for the monitor seam.
@inline _syn_acc(s::PlasticState) = _syn_acc(s.base)
@inline _syn_acc(s::BatchedSTDState) = _syn_acc(getfield(s, :base))
