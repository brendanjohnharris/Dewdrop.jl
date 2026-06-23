# * Fixed-step engine (M1) behind the CommonSolve verb layer.
#
# SciML-faithful WITHOUT its container convention: we implement CommonSolve's
# init/step!/solve!/solve over our OWN concrete types and SoA state, never a flat
# `u`-vector + `f(du,u,p,t)`. The within-step schedule is compiled to compile-time
# `Val(::Symbol)` dispatch (no runtime Symbol comparison in the hot loop). The dense
# per-step phases are written as fused, allocation-free broadcasts over the SoA so the
# same code is GPU-ready (the sparse synaptic scatter, added later, is a separate kernel).

import CommonSolve: init, step!, solve!, solve
export init, step!, solve!, solve

"""
    Projection(synapse, conn)

A synaptic projection: a synapse model applied over connectivity `conn`, which carries the
per-synapse weights and (heterogeneous) delays. For M1 this is a recurrent projection within
one population.
"""
struct Projection{SM <: AbstractSynapseModel, C <: AbstractConnectivity, PL}
    synapse::SM
    conn::C
    plasticity::PL       # an AbstractPlasticityRule (STDP) or `nothing` (static synapse); see Plasticity.jl
end
Projection(synapse::AbstractSynapseModel, conn::AbstractConnectivity; plasticity = nothing) =
    Projection(synapse, conn, plasticity)
export Projection

# Runtime synaptic state assembled at `init`, dispatched on the synapse model:
abstract type AbstractSynapseState end

# Current-based (CUBA): a per-neuron synaptic current accumulator, its delay ring buffer,
# the connectivity, and the precomputed synaptic decay coefficient.
struct SynapseState{IS, B, C, T} <: AbstractSynapseState
    Isyn::IS
    buf::B
    conn::C
    decay::T
end
Adapt.@adapt_structure SynapseState

# Conductance-based (COBA): a per-neuron conductance accumulator + reversal potential; the
# synaptic current is g·(Erev − V), contributing to BOTH the effective leak and the drive.
struct COBAState{G, B, C, T} <: AbstractSynapseState
    g::G
    buf::B
    conn::C
    decay::T
    Erev::T
end
Adapt.@adapt_structure COBAState

# Delta (instantaneous voltage-jump): just the delay buffer + connectivity (no current, no decay).
struct DeltaSynapseState{B, C} <: AbstractSynapseState
    buf::B
    conn::C
end
Adapt.@adapt_structure DeltaSynapseState

# Dual-exponential COBA: two per-target conductance accumulators (`g_rise`, `g_decay`), each kicked
# by the delivered weight and decaying with its own coefficient; the conductance is
# `a·(g_decay − g_rise)`, contributing to the effective leak (`gtot`) and reversal drive (`itot`).
struct DualExpCOBAState{G, B, C, T} <: AbstractSynapseState
    g_rise::G
    g_decay::G
    buf::B
    conn::C
    decay_r::T
    decay_d::T
    a::T
    Erev::T
end
Adapt.@adapt_structure DualExpCOBAState

"""
    PoissonDrive(; rate, weight, seed=0)

External Poisson drive: each step every neuron receives `n ~ Poisson(rate·dt)` independent
external spikes, each an instantaneous voltage kick of `weight`. `rate` is the total external
rate per neuron (per unit time). Drawn reproducibly from the counter-based RNG keyed by
`(seed, step, neuron)`, so the drive is identical across runs, threads and devices.
"""
struct PoissonDrive{T}
    rate::T
    weight::T
    seed::UInt64
end
function PoissonDrive(; rate, weight, seed = 0)
    r, w = promote(to_rate(rate), to_voltage(weight))   # weight is a voltage kick
    return PoissonDrive(r, w, UInt64(seed))
end
export PoissonDrive

"""
    DewdropNetwork(model, N; input, tspan, arch=CPU(), schedule=default_schedule(), projection=nothing)

A simulation problem: `N` units of neuron `model` driven by external `input` (a scalar
constant current, or a per-unit array) over `tspan`. An optional recurrent [`Projection`](@ref)
adds synaptic coupling.
"""
struct DewdropNetwork{M <: AbstractNeuronModel, In, A <: AbstractArchitecture, S <: Schedule, T, P <: Tuple, DR, NO, SP <: NamedTuple, PO}
    model::M
    n::Int
    input::In
    tspan::Tuple{T, T}
    arch::A
    schedule::S
    projections::P              # a (possibly empty) tuple of Projections
    drive::DR
    noise::NO                   # WhiteNoise, or `nothing` (compiles away); see Noise.jl
    subpops::SP                 # named-subpopulation registry: name → contiguous range into 1:N; see Builder.jl
    positions::PO               # per-neuron positions (host metadata for spatial measures), or `nothing`
end
# normalise the `projection` (singular) / `projections` (plural) keywords to a tuple
_normalize_projections(p::Projection, ::Nothing) = (p,)
_normalize_projections(::Nothing, ::Nothing) = ()
_normalize_projections(::Nothing, ps) = Tuple(ps)
# normalise the subpop registry: always carry an implicit `:all` spanning the whole population, with
# any user-named subpops layered on top (a user-supplied `:all` wins). Pure host-side metadata.
_normalize_subpops(::Nothing, N::Int) = (all = 1:N,)
_normalize_subpops(nt::NamedTuple, N::Int) = merge((all = 1:N,), nt)
function DewdropNetwork(model::AbstractNeuronModel, N::Integer; input, tspan,
        arch::AbstractArchitecture = CPU(), schedule::Schedule = default_schedule(),
        projection = nothing, projections = nothing, drive = nothing, noise = nothing,
        subpops = nothing, positions = nothing)
    T = float_type(model)
    in_ = on_architecture(arch, to_current(input))     # per-unit input arrays move to the architecture
    projs = _normalize_projections(projection, projections)
    return DewdropNetwork(model, Int(N), in_, (T(to_time(tspan[1])), T(to_time(tspan[2]))), arch, schedule, projs, drive, noise, _normalize_subpops(subpops, Int(N)), positions)
end
export DewdropNetwork

"""
    FixedStep(dt)

The fixed-step, clock-driven algorithm with time step `dt`.
"""
struct FixedStep{T}
    dt::T
    FixedStep{T}(dt) where {T} = new{T}(dt)   # explicit inner ctor suppresses the auto outer one
end
FixedStep(dt) = (d = to_time(dt); FixedStep{typeof(d)}(d))   # strips units; inner ctor → no recursion
FixedStep(; dt) = FixedStep(dt)
export FixedStep

"""
    DewdropIntegrator

Mutable, concretely-typed integrator cache (the CommonSolve `integrator`). Holds the SoA
state and preallocated buffers; only the step counter `n` and time `t` mutate.
"""
mutable struct DewdropIntegrator{M, ST, In, A, S, T, B, C, SY, GT, MO, DR, CO, NO, BK, SP, PO}
    const model::M
    const state::ST
    const input::In
    const dt::T
    n::Int
    t::T
    const tend::T
    const arch::A
    const schedule::S
    const spiked::B
    const spike_count::C
    const syns::SY               # a tuple of synapse states (one per projection; possibly empty)
    const gtot::GT               # per-neuron conductance accumulator (scratch, COBA)
    const itot::GT               # per-neuron input-current accumulator (scratch)
    const monitors::MO           # NamedTuple of monitors (possibly empty); see Monitors.jl
    const drive::DR              # PoissonDrive, or `nothing`
    const sync_every::Int        # periodic device-sync cadence for long runs (0 = never); see Fused.jl
    const compaction::CO         # CompactionScratch (scatter = :compacted) or `nothing`; see Compaction.jl
    const noise::NO              # WhiteNoise (SDE diffusion) or `nothing` (compiles away); see Noise.jl
    const backend::BK            # resolved execution backend (Serial/Fused/Turbo); routes the step, see Fused.jl
    const subpops::SP            # named-subpopulation registry (host-side metadata; travels onto the solution)
    const positions::PO          # per-neuron positions (host-side metadata; travels onto the solution)
end
# Device-movable (GPU-readiness contract): adapt the SoA state + buffers, leave the host-side
# `subpops`/`positions` metadata (a custom rule keeps `positions` host-resident even on a GPU run).
Adapt.adapt_structure(to, integ::DewdropIntegrator) = DewdropIntegrator(
    adapt(to, integ.model), adapt(to, integ.state), adapt(to, integ.input), integ.dt,
    integ.n, integ.t, integ.tend, integ.arch, integ.schedule, adapt(to, integ.spiked),
    adapt(to, integ.spike_count), adapt(to, integ.syns), adapt(to, integ.gtot), adapt(to, integ.itot),
    adapt(to, integ.monitors), adapt(to, integ.drive), integ.sync_every, adapt(to, integ.compaction),
    adapt(to, integ.noise), integ.backend, integ.subpops, integ.positions,
)

function CommonSolve.init(prob::DewdropNetwork, alg::FixedStep;
        record = nothing, v0 = nothing, v0_seed::Unsigned = 0x5eed00d % UInt64,
        batch = nothing, input = nothing, streams = nothing, sync_every::Integer = _DEFAULT_WINDOW,
        scatter::Symbol = :auto, backend::SimBackend = Auto(), step::Union{Nothing, Symbol} = nothing)
    # `scatter = :auto` (default): on the GPU pick edge-parallel vs compacted from the connectome size
    # (the L2-spill crossover, see `_resolve_scatter`); CPU / plastic projections → always `:edge`.
    # Resolved BEFORE the batched dispatch so both the scalar and batched paths receive a concrete mode.
    scatter = _resolve_scatter(scatter, prob.arch, prob.projections)
    # `batch = B` routes to the ensemble (tensor) batched path (src/Batch.jl); the scalar B=1
    # path below is unchanged (batch defaults to `nothing`), so existing runs are bit-identical.
    # (the batched path is always the fused megakernel, so the backend does not apply there.)
    batch === nothing || return _batched_init(prob, alg, Int(batch);
        record, v0, v0_seed, input, streams, sync_every, scatter)
    # the execution backend (Auto/Serial/Fused/Turbo; see Backends.jl). `Auto` resolves to the best
    # available for this problem; the deprecated `step::Symbol` (:auto/:fused/…) maps onto a backend.
    bk = _resolve_backend(step === nothing ? backend : _step_to_backend(step), prob)
    _check_backend(bk, prob)
    _check_accum_record(bk, record)        # :gtot/:itot recording needs a backend that materialises them
    arch = prob.arch
    T = float_type(prob.model)
    N = prob.n
    dt = T(alg.dt)
    _check_drive(prob.drive, dt)
    # a Heterogeneous / MultiModel model (per-neuron / per-group resolution) runs the fused per-neuron
    # path (`_resolve_backend` returned `Fused`) and needs the canonical schedule. Validate the
    # override array lengths / group ranges against N.
    hetero = _is_hetero(prob.model)
    hetero && _check_hetero(prob.model, N)
    hetero && prob.schedule != default_schedule() &&
        throw(ArgumentError("Heterogeneous / MultiModel models require the canonical schedule (they run via the fused per-neuron step)"))
    state = Population(arch, prob.model, N)                 # type-stable (names from model type)
    _init_voltage_model!(state.state.V, prob.model, v0, T, v0_seed)   # refrac stays 0; per-group EL via _resting
    # `spiked`/`spike_count` eltype is Bool/Int for every backend (bit-identical), except the
    # `Differentiable` backend, which accumulates a REAL-valued surrogate spike (eltype = state float).
    ST = _spiked_eltype(bk, T)
    CT = _count_eltype(bk, T)
    spiked = fill!(allocate(arch, ST, N), zero(ST))
    spike_count = fill!(allocate(arch, CT, N), zero(CT))
    # plastic projections (STDP) build a PlasticState (mutable weights + traces); static ones the
    # base synapse state. The compacted scatter walks active SOURCES only, so it cannot drive STDP's
    # postsynaptic-potentiation branch --- plastic projections require the edge scatter.
    scatter === :compacted && any(p -> p.plasticity !== nothing, prob.projections) &&
        throw(ArgumentError("STDP (plastic projections) require scatter = :edge; the compacted scatter cannot drive the postsynaptic potentiation branch"))
    syns = map(p -> _make_synstate(arch, p.synapse, p.conn, p.plasticity, T, N, dt), prob.projections)
    gtot = fill!(allocate(arch, T, N), zero(T))
    itot = fill!(allocate(arch, T, N), zero(T))
    nsteps = round(Int, (prob.tspan[2] - prob.tspan[1]) / dt)
    monitors = _make_monitors(record, arch, T, N, nsteps, prob.subpops)
    compaction = _make_compaction(scatter, arch, N)
    return DewdropIntegrator(
        prob.model, state, prob.input, dt,
        0, prob.tspan[1], prob.tspan[2], arch, prob.schedule, spiked, spike_count, syns,
        gtot, itot, monitors, prob.drive, Int(sync_every), compaction, prob.noise, bk,
        prob.subpops, prob.positions,
    )
end

# `:edge` → the edge-parallel scatter (no per-step sync); `:compacted` → the compacted device fast path
# (src/Compaction.jl), allocating the per-step active-neuron scratch. `:auto` is resolved to one of these
# in `init` (see `_resolve_scatter`) before this is reached.
_make_compaction(scatter::Symbol, arch, N) =
    scatter === :compacted ? CompactionScratch(arch, N) :
    scatter === :edge ? nothing :
    throw(ArgumentError("scatter must be :edge, :compacted or :auto (got :$scatter)"))

# Default GPU L2 cache size (bytes) when the device can't be queried (CUDA unloaded / older driver); the
# CUDA extension overrides `_l2_cache_bytes(::GPU)` with the real device attribute.
const _DEFAULT_L2_BYTES = 40 * 1024 * 1024
_l2_cache_bytes(::AbstractArchitecture) = _DEFAULT_L2_BYTES

# Resolve `scatter = :auto`. The fused GPU edge-parallel scatter reads one source index PER EDGE every
# step (Θ(nedges) memory traffic, independent of how few neurons fire), so once the connectome's index
# array spills the L2 cache the step becomes bandwidth-bound and scales with edge count --- a large GPU
# network degrades superlinearly. The compacted scatter (Compaction.jl) touches only ACTIVE sources
# (work ∝ spikes, for the usual sparse firing), winning past that crossover despite its per-step host
# sync. So: CPU → always `:edge` (the CPU scatter already walks only spiking rows --- compaction has no
# upside and adds a compactify pass); plastic projections → `:edge` (the compacted scatter cannot drive
# STDP potentiation); GPU → `:compacted` once the index footprint exceeds ~half the L2 (calibrated to the
# measured ~1.3e7-edge / 96 MB-L2 crossover), else `:edge`. The firing rate is unknown at init, so this
# assumes the sparse-firing regime typical of SNNs --- force a mode (`scatter = :edge` / `:compacted`)
# for a dense large GPU network.
function _resolve_scatter(scatter::Symbol, arch::AbstractArchitecture, projections)
    scatter === :auto || return scatter
    (arch isa GPU && !isempty(projections) && all(p -> p.plasticity === nothing, projections)) || return :edge
    src_bytes = sum(p -> nedges(p.conn) * sizeof(eltype(p.conn.src)), projections; init = 0)
    return src_bytes > _l2_cache_bytes(arch) ÷ 2 ? :compacted : :edge
end

# Guard the Poisson drive against the sampler's underflow cliff: `poisson_count` inverts the
# CDF from `exp(-λ)`, which underflows to 0 for λ ≳ 745 and then silently returns the iteration
# cap every step. λ = rate·dt is the mean events per step, so `rate` must be in events per unit
# time matching `dt` --- passing a per-second rate with a per-millisecond `dt` overshoots 1000×.
_check_drive(::Nothing, dt) = nothing
function _check_drive(d::PoissonDrive, dt)
    λ = d.rate * dt
    λ < 700 || throw(ArgumentError(
        "PoissonDrive mean events/step λ = rate*dt = $λ is too large (≥ 700): the Poisson " *
        "sampler underflows and saturates. `rate` is in events per unit time matching `dt` " *
        "(here dt = $dt); did you pass a per-second rate with a per-millisecond dt (1000× too big)?"))
    return nothing
end

# Initial membrane potential. `nothing` => the leak reversal `EL` (synchronous default); a
# scalar => a uniform clamp; a `(lo, hi)` tuple => uniform-random per neuron via the
# counter-based RNG (deterministic, GPU-safe, breaks initial synchrony --- essential for the
# balanced asynchronous-irregular state); an explicit vector => copied verbatim.
_init_voltage!(V, EL, ::Nothing, ::Type{T}, seed) where {T} = (fill!(V, EL); V)
_init_voltage!(V, EL, v0::Real, ::Type{T}, seed) where {T} = (fill!(V, T(v0)); V)
_init_voltage!(V, EL, v0::AbstractVector, ::Type{T}, seed) where {T} = (copyto!(V, v0); V)
function _init_voltage!(V, EL, v0::Tuple{<:Real, <:Real}, ::Type{T}, seed) where {T}
    lo, hi = T(v0[1]), T(v0[2])
    idx = eachindex(V)
    @. V = lo + (hi - lo) * draw_uniform(T, seed, 0, idx)
    return V
end

function _make_synstate(arch, syn::CurrentSynapse, conn, ::Type{T}, N, dt) where {T}
    Isyn = fill!(allocate(arch, T, N), zero(T))
    buf = DelayBuffer(arch, T, N, maximum(conn.delay; init = 0))   # empty projection → L=1 no-op
    return SynapseState(Isyn, buf, conn, synapse_decay(syn, dt))
end
function _make_synstate(arch, syn::ConductanceSynapse, conn, ::Type{T}, N, dt) where {T}
    g = fill!(allocate(arch, T, N), zero(T))
    buf = DelayBuffer(arch, T, N, maximum(conn.delay; init = 0))   # empty projection → L=1 no-op
    return COBAState(g, buf, conn, synapse_decay(syn, dt), T(syn.Erev))
end
function _make_synstate(arch, ::DeltaSynapse, conn, ::Type{T}, N, dt) where {T}
    buf = DelayBuffer(arch, T, N, maximum(conn.delay; init = 0))   # empty projection → L=1 no-op
    return DeltaSynapseState(buf, conn)
end
function _make_synstate(arch, syn::DualExpSynapse, conn, ::Type{T}, N, dt) where {T}
    g_rise = fill!(allocate(arch, T, N), zero(T))
    g_decay = fill!(allocate(arch, T, N), zero(T))
    buf = DelayBuffer(arch, T, N, maximum(conn.delay; init = 0))   # empty projection → L=1 no-op
    return DualExpCOBAState(g_rise, g_decay, buf, conn,
        T(exp(-dt / syn.τr)), T(exp(-dt / syn.τd)), T(_dualexp_a(syn.τr, syn.τd)), T(syn.Erev))
end

# --- Within-step phases. Synapse work iterates the projection tuple, dispatching each
# operation on the per-projection synaptic-state type (compile-time unrolled), so a
# population can carry any mix of CUBA / COBA / delta projections, and an unconnected
# population (`syns === ()`) compiles the synapse work away entirely. ---

# Per-projection: apply this step's delivered increments (from the ring buffer).
@inline _deliver!(syn::SynapseState, integ) = (deliver_due!(syn.Isyn, syn.buf, integ.n); nothing)
@inline _deliver!(syn::COBAState, integ) = (deliver_due!(syn.g, syn.buf, integ.n); nothing)
@inline _deliver!(syn::DeltaSynapseState, integ) = (deliver_due!(integ.state.state.V, syn.buf, integ.n); nothing)
@inline _deliver!(syn::DualExpCOBAState, integ) = (deliver_due_dual!(syn.g_rise, syn.g_decay, syn.buf, integ.n); nothing)

# Per-projection: accumulate the input current (itot) and conductance (gtot) for this step.
@inline _accumulate!(syn::SynapseState, gtot, itot, V) = (@. itot += syn.Isyn; nothing)
@inline _accumulate!(syn::COBAState, gtot, itot, V) = (@. gtot += syn.g; @. itot += syn.g * syn.Erev; nothing)
@inline _accumulate!(syn::DeltaSynapseState, gtot, itot, V) = nothing  # applied directly to V at deliver
@inline function _accumulate!(syn::DualExpCOBAState, gtot, itot, V)    # conductance g = a·(g_decay − g_rise)
    @. gtot += syn.a * (syn.g_decay - syn.g_rise)
    @. itot += syn.a * (syn.g_decay - syn.g_rise) * syn.Erev
    return nothing
end

# Per-projection: decay the synaptic state by one step.
@inline _decay!(syn::SynapseState) = (@. syn.Isyn *= syn.decay; nothing)
@inline _decay!(syn::COBAState) = (@. syn.g *= syn.decay; nothing)
@inline _decay!(syn::DeltaSynapseState) = nothing
@inline _decay!(syn::DualExpCOBAState) = (@. syn.g_rise *= syn.decay_r; @. syn.g_decay *= syn.decay_d; nothing)

# Compile-time tuple unrolls (dispatch on each element's concrete type → no runtime dispatch).
@inline _deliver_all!(::Tuple{}, integ) = nothing
@inline _deliver_all!(s::Tuple, integ) = (_deliver!(first(s), integ); _deliver_all!(Base.tail(s), integ))
@inline _accum_all!(::Tuple{}, gtot, itot, V) = nothing
@inline _accum_all!(s::Tuple, gtot, itot, V) = (_accumulate!(first(s), gtot, itot, V); _accum_all!(Base.tail(s), gtot, itot, V))
@inline _decay_all!(::Tuple{}) = nothing
@inline _decay_all!(s::Tuple) = (_decay!(first(s)); _decay_all!(Base.tail(s)))

# Once-per-step GLOBAL synapse hook, run at the very start of the step in EVERY backend (before any
# per-neuron synaptic work). The default is a no-op (compiles away for ordinary synapses, so the step
# stays bit-identical and dispatch-free); a streaming drive overrides it to GENERATE + SCATTER its own
# events each step (the external Poisson population, with no precomputed (N×nsteps) matrix). It is kept
# distinct from `_deliver!` because the Fused/megakernel path fuses delivery into the per-neuron
# `_syn_one` and never calls `_deliver!`, yet a global once-per-step generator still needs a home there.
@inline _synprestep!(::AbstractSynapseState, integ) = nothing
@inline _synprestep_all!(::Tuple{}, integ) = nothing
@inline _synprestep_all!(s::Tuple, integ) = (_synprestep!(first(s), integ); _synprestep_all!(Base.tail(s), integ))

@inline function run_phase!(::Val{:deliver}, integ::DewdropIntegrator)
    _synprestep_all!(integ.syns, integ)                    # streaming drives generate + scatter their own events
    _deliver_all!(integ.syns, integ)                       # ring-buffer due → per-projection state (or V, for delta)
    _apply_drive!(integ.drive, integ)                      # external Poisson kicks → V
    return nothing
end

@inline _apply_drive!(::Nothing, integ) = nothing
@inline function _apply_drive!(drive::PoissonDrive, integ)
    V = integ.state.state.V
    λ = drive.rate * integ.dt
    w = drive.weight
    seed = drive.seed
    n = integ.n
    idx = eachindex(V)
    @. V += w * draw_poisson(λ, seed, n, idx)              # per-neuron Poisson voltage kicks
    return nothing
end

# Propagate through the compaction seam so `scatter = :compacted` is honoured on EVERY backend
# (the CPU broadcast path included), matching the fused GPU path and the batched path --- not a
# silent no-op. `nothing` → edge-parallel scatter; a CompactionScratch → the compacted scatter.
@inline run_phase!(::Val{:propagate}, integ::DewdropIntegrator) = _propagate_step!(integ.compaction, integ)

# Per-neuron subthreshold membrane update, DISPATCHED ON THE MODEL: advance V over `dt` given
# the accumulated conductance `gtot` and current `itot`. The engine calls this rather than
# inlining any particular model's fields, so a model defined by hand or by `@neuron` plugs in
# by providing its own method. LIF uses the COBA-capable exact propagator above.
@inline membrane_step(m::LIF, V, gtot, itot, dt) = _coba_step(V, m.EL, m.R, m.τ, gtot, itot, dt)

@inline function run_phase!(::Val{:integrate}, integ::DewdropIntegrator)
    st = integ.state.state
    V, refrac = st.V, st.refrac
    m = integ.model
    dt = integ.dt
    gtot, itot = integ.gtot, integ.itot
    # per-neuron conductance + input current from external input and every projection
    itot .= integ.input                                    # base: external input (scalar or per-unit)
    fill!(gtot, zero(eltype(gtot)))
    _accum_all!(integ.syns, gtot, itot, V)
    Vr = reset_value(m)
    z = zero(eltype(refrac))
    # subthreshold membrane step, dispatched on the model's state shape: V-only models (LIF, every
    # @neuron) take the exact prior broadcast unchanged; adaptation models co-advance (V, w).
    _integrate_membrane!(m, st, gtot, itot, dt, refrac, Vr, z)
    _apply_noise!(integ.noise, integ, refrac, z)           # SDE diffusion increment (no-op if no noise)
    @. refrac = max(refrac - dt, z)
    _decay_all!(integ.syns)                                # advance each projection's synaptic state
    return nothing
end

# SDE noise injection (M5a): add the exact-OU Gaussian increment to non-refractory neurons. The
# `nothing` method compiles the diffusion away entirely (bit-identical to a deterministic run); the
# `WhiteNoise` method draws one counter-based normal per neuron keyed by (seed, step, neuron).
@inline _apply_noise!(::Nothing, integ, refrac, z) = nothing
@inline function _apply_noise!(noise::WhiteNoise, integ::DewdropIntegrator, refrac, z)
    V = integ.state.state.V
    s = _noise_scale(noise, integ.model, integ.dt)
    seed = noise.seed
    n = integ.n
    idx = eachindex(V)
    @. V = ifelse(refrac > z, V, V + s * draw_normal(eltype(V), seed, n, idx))
    return nothing
end

@inline function run_phase!(::Val{:threshold}, integ::DewdropIntegrator)
    st = integ.state.state
    m = integ.model
    z = zero(eltype(st.refrac))
    @. integ.spiked = (st.refrac ≤ z) & threshold(m, st.V)   # model's threshold predicate
    return nothing
end

@inline function run_phase!(::Val{:reset}, integ::DewdropIntegrator)
    st = integ.state.state
    spiked = integ.spiked
    Vr = reset_value(integ.model)
    tref = refractory(integ.model)
    @. st.V = ifelse(spiked, Vr, st.V)
    @. st.refrac = ifelse(spiked, tref, st.refrac)
    _reset_aux!(integ.model, st, spiked)        # spike-triggered adaptation increment (no-op if no w)
    return nothing
end

@inline function run_phase!(::Val{:record}, integ::DewdropIntegrator)
    @. integ.spike_count += integ.spiked       # always-on per-neuron count (drives firing_rate)
    _record_all!(integ.monitors, integ)        # the requested monitors (Monitors.jl)
    return nothing
end

# Compile-time unroll of the schedule (carried in its type parameter) into a straight-line
# sequence of phase calls --- no runtime Symbol comparison, no dynamic dispatch.
@generated function run_phases!(::Schedule{P}, integ::DewdropIntegrator) where {P}
    calls = [:(run_phase!(Val($(QuoteNode(ph))), integ)) for ph in P]
    return Expr(:block, calls..., :(return nothing))
end

# A step runs the schedule's phases. `_run_step!` is the seam the fused device path
# (src/Fused.jl) specialises for the canonical schedule on a GPU backend; the generic method
# is the broadcast-per-phase execution used on the CPU and for any non-canonical schedule.
_run_step!(sched::Schedule, integ::DewdropIntegrator) = run_phases!(sched, integ)

function CommonSolve.step!(integ::DewdropIntegrator)
    _run_step!(integ.schedule, integ)
    integ.n += 1
    integ.t += integ.dt
    _maybe_sync!(integ)            # bound the device queue depth on long runs (no-op on CPU / B=0)
    return integ
end

function CommonSolve.solve!(integ::DewdropIntegrator)
    while integ.t < integ.tend - integ.dt / 2
        step!(integ)
    end
    _finalize_all!(integ.monitors)             # flush each monitor's partial last window to host
    return DewdropSolution(integ)
end

"""
    DewdropSolution

The result of a run: final SoA `state`, per-unit `spike_count`, number of steps, `dt`, `tspan`,
and `record` --- a NamedTuple of the requested monitors' results (see [`firing_rate`](@ref),
[`raster`](@ref), and the `record` kwarg to `solve`).
"""
struct DewdropSolution{ST, C, T, R, SP, PO}
    state::ST
    spike_count::C
    nsteps::Int
    dt::T
    tspan::Tuple{T, T}
    record::R         # NamedTuple{names}(::RecordResult...) keyed by monitor name
    subpops::SP       # named-subpopulation registry (name → range into 1:N); see `sol[:E]`
    positions::PO     # per-neuron positions (host metadata for spatial measures), or `nothing`
end
function DewdropSolution(integ::DewdropIntegrator)
    return DewdropSolution(
        integ.state, integ.spike_count, integ.n, integ.dt,
        (integ.t - integ.n * integ.dt, integ.t), map(_result, integ.monitors), integ.subpops, integ.positions,
    )
end
export DewdropSolution

"""
    duration(sol)

Simulated duration (`nsteps · dt`).
"""
duration(sol::DewdropSolution) = sol.nsteps * sol.dt

"""
    firing_rate(sol)

Per-unit firing rate (`spike_count / duration`), in inverse units of `dt`.
"""
firing_rate(sol::DewdropSolution) = sol.spike_count ./ duration(sol)
export firing_rate

# --- Named-subpopulation reference API (Phase A) ---
# A subpop is a contiguous range into the flat SoA, looked up in the solution's registry. `sol[:E]`
# returns a lightweight view (no copy); `firing_rate(sol, :E)` / `raster(sol; of = :E)` restrict to it.

# resolve a subpop name to its range, with a clear error listing the available names.
function _subrange(subpops::NamedTuple, name::Symbol)
    haskey(subpops, name) || throw(ArgumentError(
        "unknown subpopulation :$name --- available: $(join(keys(subpops), ", "))"))
    return subpops[name]
end

"""
    SubSolution

A view of a [`DewdropSolution`](@ref) restricted to one named subpopulation (see `sol[:E]`). Carries
the subpopulation's `state` (SoA view) and `spike_count` (view) over its range; works with
[`firing_rate`](@ref) and [`duration`](@ref).
"""
struct SubSolution{ST, C, S, R, PO}
    state::ST          # sub-StructArray view over the subpop range
    spike_count::C     # view over the subpop range
    parent::S          # the parent DewdropSolution
    name::Symbol
    range::R
    positions::PO      # per-neuron positions over the range (host metadata), or `nothing`
end
export SubSolution

# positions restricted to a subpop range (or `nothing` if the network carried none)
@inline _subpositions(::Nothing, r) = nothing
@inline _subpositions(positions, r) = @view positions[r]

function Base.getindex(sol::DewdropSolution, name::Symbol)
    r = _subrange(sol.subpops, name)
    # a StructArray view is a StructArray of column views (no copy); re-wrap as a Population so the
    # sub-solution presents the same `state.state.<col>` shape as the parent.
    return SubSolution(Population(view(sol.state.state, r)), view(sol.spike_count, r), sol, name, r,
        _subpositions(sol.positions, r))
end

duration(ss::SubSolution) = duration(ss.parent)
firing_rate(ss::SubSolution) = ss.spike_count ./ duration(ss.parent)

"""
    firing_rate(sol, name::Symbol)

Per-unit firing rate of named subpopulation `name` (e.g. `firing_rate(sol, :E)`).
"""
firing_rate(sol::DewdropSolution, name::Symbol) = view(sol.spike_count, _subrange(sol.subpops, name)) ./ duration(sol)

"""
    raster(sol; name=nothing, of=nothing) -> (times, ids)

Extract spike events from a solution recorded with a `Spikes()` monitor: `times[k]` and `ids[k]`
are the time and neuron index of the k-th spike (host-side analysis helper). `name` picks a
particular spike monitor; by default the first one is used. `of` (a subpopulation symbol, e.g.
`:E`) restricts the events to that subpopulation and rebases `ids` into `1:|of|`.
"""
function raster(sol::DewdropSolution; name = nothing, of = nothing)
    res = _find_spikes(sol.record, name)
    res === nothing && error("no spikes recorded --- pass `record = (spikes = Spikes(),)` to `solve`")
    idx = findall(res.data)                          # CartesianIndex(recorded-row, column)
    times = [I[2] * res.every * sol.dt for I in idx]
    ids = [_neuronid(res.idx, I[1]) for I in idx]
    of === nothing && return times, ids
    r = _subrange(sol.subpops, of)                   # restrict to subpop `of`, rebasing ids into 1:|of|
    keep = findall(in(r), ids)
    return times[keep], [ids[k] - first(r) + 1 for k in keep]
end
export raster

_find_spikes(record, name::Symbol) = record[name]
function _find_spikes(record, ::Nothing)
    for r in values(record)
        r.kind === :spikes && return r
    end
    return nothing
end
@inline _neuronid(::Colon, row) = row
@inline _neuronid(idx, row) = @inbounds idx[row]
