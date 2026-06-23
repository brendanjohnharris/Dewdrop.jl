# * WRCircuit reproduction --- the spatial FNS E/I "working-regime" circuit (WRCircuit.jl / BrainPy
# `Spatial`) re-expressed in native Dewdrop and reproduced bit-for-bit (up to numerical error).
#
# Two synapse variants here reproduce BrainPy's *integration scheme* exactly (the structural pieces ---
# connectome, per-edge weights, initial V, the external Poisson drive --- are JAX-PRNG outputs and must
# be ingested, not regenerated; see test/simulator_comparisons/wrcircuit):
#
#  1. `FrozenDualExpSynapse` --- a dual-exponential COBA synapse whose current `g·(Erev − V)` is FROZEN
#     at the pre-step `V` (BrainPy/Brian `sum_current_inputs` semantics), contributing to the input
#     current `itot` only and NOT to the membrane leak `gtot`. Dewdrop's default `DualExpSynapse` instead
#     folds the synaptic conductance into the leak (the exact COBA propagator) --- more accurate, but a
#     different O(dt) scheme. The frozen variant matches BrainPy. Because it leaves `gtot` carrying only
#     the neuron's own adaptation conductance, the unmodified `FNSNeuron._step_V` (`denom = 1 + R·gK`)
#     reproduces BrainPy's `A = −(gL + gK)/C` exactly: the "factor moved to the right level" --- the
#     explicit/implicit choice lives in the synapse, not the neuron.
#
#  2. `PrescribedCOBA` --- a per-target conductance trajectory prescribed step-by-step (the external
#     Poisson population's dual-exp drive, precomputed from an exported spike raster). Same frozen-current
#     COBA contribution, with `g(n)` read from a stored `(N, nsteps)` matrix rather than evolved from
#     delivered spikes. Lets the external drive be replayed without a spike-source population.

# --- 1. Frozen-current (explicit-COBA) dual-exponential synapse ---------------------------------
"""
    FrozenDualExpSynapse(; τr, τd, Erev)

Dual-exponential COBA synapse whose current `g·(Erev − V)` is evaluated at the PRE-step `V` and held
constant over the step (BrainPy/Brian `sum_current_inputs` semantics), rather than folding the synaptic
conductance into the membrane leak (Dewdrop's default [`DualExpSynapse`](@ref)). Same conductance
kinetics `g(t) = a·(g_decay − g_rise)` as `DualExpSynapse`; the difference is purely how the resulting
current enters the membrane step (input current `itot`, not effective leak `gtot`). Use to reproduce a
BrainPy/Brian COBA network exactly; prefer `DualExpSynapse` (exact propagator) for new models.
"""
struct FrozenDualExpSynapse{T} <: AbstractSynapseModel
    τr::T
    τd::T
    Erev::T
end
function FrozenDualExpSynapse(; τr, τd, Erev)
    r, d, e = promote(to_time(τr), to_time(τd), to_voltage(Erev))
    r == d && throw(ArgumentError("FrozenDualExpSynapse requires τr ≠ τd (got τr = τd = $r)"))
    return FrozenDualExpSynapse(r, d, e)
end
export FrozenDualExpSynapse

# Same fields as DualExpCOBAState (a DISTINCT type so the frozen-current `_accumulate!`/`_syn_one`
# dispatch); see Engine.jl for the implicit DualExpCOBAState.
struct FrozenDualExpCOBAState{G, B, C, T} <: AbstractSynapseState
    g_rise::G
    g_decay::G
    buf::B
    conn::C
    decay_r::T
    decay_d::T
    a::T
    Erev::T
end
Adapt.@adapt_structure FrozenDualExpCOBAState

function _make_synstate(arch, syn::FrozenDualExpSynapse, conn, ::Type{T}, N, dt) where {T}
    g_rise = fill!(allocate(arch, T, N), zero(T))
    g_decay = fill!(allocate(arch, T, N), zero(T))
    buf = DelayBuffer(arch, T, N, maximum(conn.delay; init = 0))
    return FrozenDualExpCOBAState(g_rise, g_decay, buf, conn,
        T(exp(-dt / syn.τr)), T(exp(-dt / syn.τd)), T(_dualexp_a(syn.τr, syn.τd)), T(syn.Erev))
end

# Broadcast path (Serial backend): deliver → accumulate (frozen current at pre-step V) → decay.
@inline _deliver!(syn::FrozenDualExpCOBAState, integ) =
    (deliver_due_dual!(syn.g_rise, syn.g_decay, syn.buf, integ.n); nothing)
@inline function _accumulate!(syn::FrozenDualExpCOBAState, gtot, itot, V)
    @. itot += syn.a * (syn.g_decay - syn.g_rise) * (syn.Erev - V)   # frozen current; gtot untouched
    return nothing
end
@inline _decay!(syn::FrozenDualExpCOBAState) =
    (@. syn.g_rise *= syn.decay_r; @. syn.g_decay *= syn.decay_d; nothing)

# Fused / tight path (used by Heterogeneous E/I): per-neuron deliver + frozen-current accumulate + decay.
@inline function _syn_one(s::FrozenDualExpCOBAState, i, n, v, gtot, itot)
    L = s.buf.L
    slot = mod(n, L) + 1
    @inbounds due = s.buf.slots[i, slot]
    @inbounds s.buf.slots[i, slot] = zero(due)
    @inbounds gr = s.g_rise[i] + due
    @inbounds gd = s.g_decay[i] + due
    g = s.a * (gd - gr)
    itot += g * (s.Erev - v)                                          # frozen current at pre-step v
    @inbounds s.g_rise[i] = gr * s.decay_r
    @inbounds s.g_decay[i] = gd * s.decay_d
    return (v, gtot, itot)
end

# --- 2. Prescribed (replayed) external conductance ---------------------------------------------
"""
    PrescribedCOBA(g, Erev)

A per-target conductance trajectory prescribed step by step: at step `n` (0-based) the synapse adds the
frozen current `g[:, n+1]·(Erev − V)` to every neuron's input current. `g` is an `(N, nsteps)` matrix.
Used to replay an external drive (e.g. the WRCircuit external Poisson population's dual-exponential
conductance, precomputed from an exported spike raster) without instantiating a spike-source population.
"""
struct PrescribedCOBA{M, T} <: AbstractSynapseModel
    g::M
    Erev::T
end
function PrescribedCOBA(g::AbstractMatrix, Erev::Real)
    e = to_voltage(Erev)
    return PrescribedCOBA{typeof(g), typeof(e)}(g, e)   # inner ctor (avoid recursion onto this method)
end
export PrescribedCOBA

struct PrescribedCOBAState{M, G, B, C, T} <: AbstractSynapseState
    g_all::M       # (N, nsteps) prescribed conductance per target per step
    g::G           # (N,) scratch column (broadcast path)
    buf::B         # empty ring (no delayed delivery: the matrix is already pre-delayed)
    conn::C        # empty CSR (0 edges → the scatter is a no-op)
    Erev::T
end
Adapt.@adapt_structure PrescribedCOBAState

function _make_synstate(arch, syn::PrescribedCOBA, conn, ::Type{T}, N, dt) where {T}
    g_all = on_architecture(arch, T.(syn.g))
    g = fill!(allocate(arch, T, N), zero(T))
    buf = DelayBuffer(arch, T, N, 0)
    return PrescribedCOBAState(g_all, g, buf, conn, T(syn.Erev))
end

# clamp the column index so a possible final fp-overshoot step cannot read out of bounds.
@inline _prescribed_col(g_all, n) = min(n + 1, size(g_all, 2))

# Broadcast path: load the column in deliver, add the frozen current in accumulate.
@inline function _deliver!(syn::PrescribedCOBAState, integ)
    @inbounds @views syn.g .= syn.g_all[:, _prescribed_col(syn.g_all, integ.n)]
    return nothing
end
@inline function _accumulate!(syn::PrescribedCOBAState, gtot, itot, V)
    @. itot += syn.g * (syn.Erev - V)
    return nothing
end
@inline _decay!(syn::PrescribedCOBAState) = nothing

# Fused / tight path: read the prescribed column entry directly.
@inline function _syn_one(s::PrescribedCOBAState, i, n, v, gtot, itot)
    @inbounds g = s.g_all[i, _prescribed_col(s.g_all, n)]
    itot += g * (s.Erev - v)
    return (v, gtot, itot)
end

# --- Native WRCircuit builder ------------------------------------------------------------------
# An empty (0-edge) CSR over the whole network, for projections whose contribution is not spike-driven
# (the prescribed external drive); makes the per-step scatter a no-op while satisfying the interface.
_empty_csr(arch, N) = SparseCSR(arch, Tuple{Int, Int, Float64, Int}[]; npre = N, npost = N)

"""
    wrcircuit(; NE, NI, E, I, projections, gext=nothing, positions=nothing, tspan, arch=CPU())

Assemble the spatial FNS E/I "working-regime" network in native Dewdrop. `E`/`I` are [`FNSNeuron`](@ref)
models (merged into a per-neuron [`Heterogeneous`](@ref) model when they differ, e.g. E adapts and I does
not); `projections` is a vector of recurrent [`Projection`](@ref)s over the flat `1:NE+NI` index space
(typically built from an ingested connectome with [`FrozenDualExpSynapse`](@ref)); `gext` is an optional
`(NE+NI, nsteps)` external conductance matrix replayed via [`PrescribedCOBA`](@ref). The `:E`/`:I`
subpopulations are registered for `sol[:E]`, `firing_rate(sol, :I)`, etc. Returns a [`DewdropNetwork`](@ref).
"""
function wrcircuit(; NE::Integer, NI::Integer, E::AbstractNeuronModel, I::AbstractNeuronModel,
        projections::AbstractVector, gext = nothing, positions = nothing, tspan,
        arch::AbstractArchitecture = CPU())
    N = Int(NE) + Int(NI)
    model = _combine_models(Any[E, I], Int[NE, NI])          # Heterogeneous over the differing fields
    subpops = (E = 1:Int(NE), I = (Int(NE) + 1):N)
    projs = if gext === nothing
        Tuple(projections)
    else
        size(gext, 1) == N || throw(ArgumentError("gext has $(size(gext,1)) rows but N = $N"))
        (Tuple(projections)..., Projection(PrescribedCOBA(gext, 0.0), _empty_csr(arch, N)))
    end
    return DewdropNetwork(model, N; input = 0.0, tspan = tspan, arch = arch,
        projections = projs, subpops = subpops, positions = positions)
end
export wrcircuit
