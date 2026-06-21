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
struct Projection{SM <: AbstractSynapseModel, C <: AbstractConnectivity}
    synapse::SM
    conn::C
end
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
struct DewdropNetwork{M <: AbstractNeuronModel, In, A <: AbstractArchitecture, S <: Schedule, T, P <: Tuple, DR}
    model::M
    n::Int
    input::In
    tspan::Tuple{T, T}
    arch::A
    schedule::S
    projections::P              # a (possibly empty) tuple of Projections
    drive::DR
end
# normalise the `projection` (singular) / `projections` (plural) keywords to a tuple
_normalize_projections(p::Projection, ::Nothing) = (p,)
_normalize_projections(::Nothing, ::Nothing) = ()
_normalize_projections(::Nothing, ps) = Tuple(ps)
function DewdropNetwork(model::AbstractNeuronModel, N::Integer; input, tspan,
        arch::AbstractArchitecture = CPU(), schedule::Schedule = default_schedule(),
        projection = nothing, projections = nothing, drive = nothing)
    T = float_type(model)
    in_ = on_architecture(arch, to_current(input))     # per-unit input arrays move to the architecture
    projs = _normalize_projections(projection, projections)
    return DewdropNetwork(model, Int(N), in_, (T(to_time(tspan[1])), T(to_time(tspan[2]))), arch, schedule, projs, drive)
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
mutable struct DewdropIntegrator{M, ST, In, A, S, T, B, C, SY, GT, MO, DR, CO}
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
end
# Device-movable (GPU-readiness contract): adapt the SoA state + buffers, leave scalars.
Adapt.@adapt_structure DewdropIntegrator

function CommonSolve.init(prob::DewdropNetwork, alg::FixedStep;
        record = nothing, v0 = nothing, v0_seed::Unsigned = 0x5eed00d % UInt64,
        batch = nothing, input = nothing, streams = nothing, sync_every::Integer = _DEFAULT_WINDOW,
        scatter::Symbol = :edge)
    # `batch = B` routes to the ensemble (tensor) batched path (src/Batch.jl); the scalar B=1
    # path below is unchanged (batch defaults to `nothing`), so existing runs are bit-identical.
    batch === nothing || return _batched_init(prob, alg, Int(batch);
        record, v0, v0_seed, input, streams, sync_every, scatter)
    arch = prob.arch
    T = float_type(prob.model)
    N = prob.n
    dt = T(alg.dt)
    _check_drive(prob.drive, dt)
    state = Population(arch, prob.model, N)                 # type-stable (names from model type)
    _init_voltage!(state.state.V, T(prob.model.EL), v0, T, v0_seed)   # refrac stays 0
    spiked = fill!(allocate(arch, Bool, N), false)
    spike_count = fill!(allocate(arch, Int, N), 0)
    syns = map(p -> _make_synstate(arch, p.synapse, p.conn, T, N, dt), prob.projections)
    gtot = fill!(allocate(arch, T, N), zero(T))
    itot = fill!(allocate(arch, T, N), zero(T))
    nsteps = round(Int, (prob.tspan[2] - prob.tspan[1]) / dt)
    monitors = _make_monitors(record, arch, T, N, nsteps)
    compaction = _make_compaction(scatter, arch, N)
    return DewdropIntegrator(
        prob.model, state, prob.input, dt,
        0, prob.tspan[1], prob.tspan[2], arch, prob.schedule, spiked, spike_count, syns,
        gtot, itot, monitors, prob.drive, Int(sync_every), compaction,
    )
end

# `scatter = :edge` (default) → the edge-parallel scatter (no per-step sync); `:compacted` → the
# compacted device fast path (src/Compaction.jl), allocating the per-step active-neuron scratch.
_make_compaction(scatter::Symbol, arch, N) =
    scatter === :compacted ? CompactionScratch(arch, N) :
    scatter === :edge ? nothing :
    throw(ArgumentError("scatter must be :edge or :compacted (got :$scatter)"))

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

# --- Within-step phases. Synapse work iterates the projection tuple, dispatching each
# operation on the per-projection synaptic-state type (compile-time unrolled), so a
# population can carry any mix of CUBA / COBA / delta projections, and an unconnected
# population (`syns === ()`) compiles the synapse work away entirely. ---

# Per-projection: apply this step's delivered increments (from the ring buffer).
@inline _deliver!(syn::SynapseState, integ) = (deliver_due!(syn.Isyn, syn.buf, integ.n); nothing)
@inline _deliver!(syn::COBAState, integ) = (deliver_due!(syn.g, syn.buf, integ.n); nothing)
@inline _deliver!(syn::DeltaSynapseState, integ) = (deliver_due!(integ.state.state.V, syn.buf, integ.n); nothing)

# Per-projection: accumulate the input current (itot) and conductance (gtot) for this step.
@inline _accumulate!(syn::SynapseState, gtot, itot, V) = (@. itot += syn.Isyn; nothing)
@inline _accumulate!(syn::COBAState, gtot, itot, V) = (@. gtot += syn.g; @. itot += syn.g * syn.Erev; nothing)
@inline _accumulate!(syn::DeltaSynapseState, gtot, itot, V) = nothing  # applied directly to V at deliver

# Per-projection: decay the synaptic state by one step.
@inline _decay!(syn::SynapseState) = (@. syn.Isyn *= syn.decay; nothing)
@inline _decay!(syn::COBAState) = (@. syn.g *= syn.decay; nothing)
@inline _decay!(syn::DeltaSynapseState) = nothing

# Compile-time tuple unrolls (dispatch on each element's concrete type → no runtime dispatch).
@inline _deliver_all!(::Tuple{}, integ) = nothing
@inline _deliver_all!(s::Tuple, integ) = (_deliver!(first(s), integ); _deliver_all!(Base.tail(s), integ))
@inline _accum_all!(::Tuple{}, gtot, itot, V) = nothing
@inline _accum_all!(s::Tuple, gtot, itot, V) = (_accumulate!(first(s), gtot, itot, V); _accum_all!(Base.tail(s), gtot, itot, V))
@inline _decay_all!(::Tuple{}) = nothing
@inline _decay_all!(s::Tuple) = (_decay!(first(s)); _decay_all!(Base.tail(s)))

@inline function run_phase!(::Val{:deliver}, integ::DewdropIntegrator)
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

# COBA-capable subthreshold step: conductances set an effective leak (`denom`) and reversal
# drive. With no conductance (gtot = 0) this reduces to the plain LIF exact propagator.
@inline function _coba_step(V, EL, R, τ, gtot, itot, dt)
    denom = 1 + R * gtot
    V∞ = (EL + R * itot) / denom
    return V∞ + (V - V∞) * exp(-dt * denom / τ)
end

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
    # refractory units clamp to Vr; others take the model's subthreshold membrane step.
    @. V = ifelse(refrac > z, Vr, membrane_step(m, V, gtot, itot, dt))
    @. refrac = max(refrac - dt, z)
    _decay_all!(integ.syns)                                # advance each projection's synaptic state
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
struct DewdropSolution{ST, C, T, R}
    state::ST
    spike_count::C
    nsteps::Int
    dt::T
    tspan::Tuple{T, T}
    record::R         # NamedTuple{names}(::RecordResult...) keyed by monitor name
end
function DewdropSolution(integ::DewdropIntegrator)
    return DewdropSolution(
        integ.state, integ.spike_count, integ.n, integ.dt,
        (integ.t - integ.n * integ.dt, integ.t), map(_result, integ.monitors),
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

"""
    raster(sol; name=nothing) -> (times, ids)

Extract spike events from a solution recorded with a `Spikes()` monitor: `times[k]` and `ids[k]`
are the time and neuron index of the k-th spike (host-side analysis helper). `name` picks a
particular spike monitor; by default the first one is used.
"""
function raster(sol::DewdropSolution; name = nothing)
    res = _find_spikes(sol.record, name)
    res === nothing && error("no spikes recorded --- pass `record = (spikes = Spikes(),)` to `solve`")
    idx = findall(res.data)                          # CartesianIndex(recorded-row, column)
    times = [I[2] * res.every * sol.dt for I in idx]
    ids = [_neuronid(res.idx, I[1]) for I in idx]
    return times, ids
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
