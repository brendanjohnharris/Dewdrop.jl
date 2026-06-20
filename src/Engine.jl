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
    r, w = promote(rate, weight)
    return PoissonDrive(r, w, UInt64(seed))
end
export PoissonDrive

"""
    DewdropNetwork(model, N; input, tspan, arch=CPU(), schedule=default_schedule(), projection=nothing)

A simulation problem: `N` units of neuron `model` driven by external `input` (a scalar
constant current, or a per-unit array) over `tspan`. An optional recurrent [`Projection`](@ref)
adds synaptic coupling.
"""
struct DewdropNetwork{M <: AbstractNeuronModel, In, A <: AbstractArchitecture, S <: Schedule, T, P, DR}
    model::M
    n::Int
    input::In
    tspan::Tuple{T, T}
    arch::A
    schedule::S
    projection::P
    drive::DR
end
function DewdropNetwork(model::AbstractNeuronModel, N::Integer; input, tspan,
        arch::AbstractArchitecture = CPU(), schedule::Schedule = default_schedule(),
        projection = nothing, drive = nothing)
    T = float_type(model)
    in_ = on_architecture(arch, input)        # per-unit input arrays move to the architecture
    return DewdropNetwork(model, Int(N), in_, (T(tspan[1]), T(tspan[2])), arch, schedule, projection, drive)
end
export DewdropNetwork

"""
    FixedStep(dt)

The fixed-step, clock-driven algorithm with time step `dt`.
"""
struct FixedStep{T}
    dt::T
end
FixedStep(; dt) = FixedStep(dt)
export FixedStep

"""
    DewdropIntegrator

Mutable, concretely-typed integrator cache (the CommonSolve `integrator`). Holds the SoA
state and preallocated buffers; only the step counter `n` and time `t` mutate.
"""
mutable struct DewdropIntegrator{M, ST, In, A, S, T, B, C, SY, SR, VR, DR}
    const model::M
    const state::ST
    const input::In
    const dt::T
    const decay::T
    n::Int
    t::T
    const tend::T
    const arch::A
    const schedule::S
    const spiked::B
    const spike_count::C
    const syn::SY                # SynapseState, or `nothing` for an unconnected population
    const spike_rec::SR          # (N, nsteps) Bool raster, or `nothing`
    const voltage_rec::VR        # (N, nsteps) V trace, or `nothing`
    const drive::DR              # PoissonDrive, or `nothing`
end
# Device-movable (GPU-readiness contract): adapt the SoA state + buffers, leave scalars.
Adapt.@adapt_structure DewdropIntegrator

function CommonSolve.init(prob::DewdropNetwork, alg::FixedStep;
        record_spikes::Bool = false, record_voltage::Bool = false)
    arch = prob.arch
    T = float_type(prob.model)
    N = prob.n
    state = Population(arch, prob.model, N)                 # type-stable (names from model type)
    fill!(state.state.V, prob.model.EL)                     # initialise V = EL; refrac = 0
    decay = propagator_decay(prob.model, T(alg.dt))
    spiked = fill!(allocate(arch, Bool, N), false)
    spike_count = fill!(allocate(arch, Int, N), 0)
    syn = _init_synapse(arch, prob.projection, T, N, T(alg.dt))
    nsteps = round(Int, (prob.tspan[2] - prob.tspan[1]) / T(alg.dt))
    spike_rec = record_spikes ? fill!(allocate(arch, Bool, N, nsteps), false) : nothing
    voltage_rec = record_voltage ? fill!(allocate(arch, T, N, nsteps), zero(T)) : nothing
    return DewdropIntegrator(
        prob.model, state, prob.input, T(alg.dt), decay,
        0, prob.tspan[1], prob.tspan[2], arch, prob.schedule, spiked, spike_count, syn,
        spike_rec, voltage_rec, prob.drive,
    )
end

_init_synapse(arch, ::Nothing, ::Type{T}, N, dt) where {T} = nothing
_init_synapse(arch, proj::Projection, ::Type{T}, N, dt) where {T} =
    _make_synstate(arch, proj.synapse, proj.conn, T, N, dt)

function _make_synstate(arch, syn::CurrentSynapse, conn, ::Type{T}, N, dt) where {T}
    Isyn = fill!(allocate(arch, T, N), zero(T))
    buf = DelayBuffer(arch, T, N, maximum(conn.delay))
    return SynapseState(Isyn, buf, conn, synapse_decay(syn, dt))
end
function _make_synstate(arch, ::DeltaSynapse, conn, ::Type{T}, N, dt) where {T}
    buf = DelayBuffer(arch, T, N, maximum(conn.delay))
    return DeltaSynapseState(buf, conn)
end

# --- Within-step phases (Val-dispatched; dense phases are fused SoA broadcasts). The
# synapse-coupled phases dispatch on the synaptic-state type, so an unconnected population
# (`syn === nothing`) compiles the synapse work away entirely. ---

@inline function run_phase!(::Val{:deliver}, integ::DewdropIntegrator)
    _deliver!(integ.syn, integ)                            # recurrent: ring-buffer due → Isyn
    _apply_drive!(integ.drive, integ)                      # external Poisson kicks → V
    return nothing
end
@inline _deliver!(::Nothing, integ) = nothing
@inline function _deliver!(syn::SynapseState, integ)
    deliver_due!(syn.Isyn, syn.buf, integ.n)               # add due increments into Isyn, in place
    return nothing
end
@inline function _deliver!(syn::DeltaSynapseState, integ)
    deliver_due!(integ.state.state.V, syn.buf, integ.n)    # delta: voltage jumps applied directly to V
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

@inline run_phase!(::Val{:propagate}, integ::DewdropIntegrator) = _propagate!(integ.syn, integ)
@inline _propagate!(::Nothing, integ) = nothing
@inline function _propagate!(syn::AbstractSynapseState, integ)
    scatter!(syn.buf, syn.conn, integ.spiked, integ.n)     # scatter this step's spikes into the buffer
    return nothing
end

@inline run_phase!(::Val{:integrate}, integ::DewdropIntegrator) = _integrate!(integ.syn, integ)

@inline function _integrate!(::Nothing, integ)
    st = integ.state.state
    V, refrac = st.V, st.refrac
    m = integ.model
    decay = integ.decay
    dt = integ.dt
    Iext = integ.input
    Vr = reset_value(m)
    z = zero(eltype(refrac))
    # refractory units are clamped to Vr (cannot integrate); others take the exact
    # linear-propagator step toward the input-dependent fixed point V∞ = EL + R·Iext.
    @. V = ifelse(refrac > z, Vr, subthreshold_step(V, asymptote(m, Iext), decay))
    @. refrac = max(refrac - dt, z)
    return nothing
end

@inline _integrate!(syn::DeltaSynapseState, integ) = _integrate!(nothing, integ)  # delta: no synaptic current term

@inline function _integrate!(syn::SynapseState, integ)
    st = integ.state.state
    V, refrac = st.V, st.refrac
    m = integ.model
    decay = integ.decay
    dt = integ.dt
    Iext = integ.input
    Isyn = syn.Isyn
    Vr = reset_value(m)
    z = zero(eltype(refrac))
    # input current is external + synaptic; the synaptic current then decays (its own propagator).
    @. V = ifelse(refrac > z, Vr, subthreshold_step(V, asymptote(m, Iext + Isyn), decay))
    @. refrac = max(refrac - dt, z)
    @. Isyn *= syn.decay
    return nothing
end

@inline function run_phase!(::Val{:threshold}, integ::DewdropIntegrator)
    st = integ.state.state
    Vθ = integ.model.Vθ
    z = zero(eltype(st.refrac))
    @. integ.spiked = (st.refrac ≤ z) & (st.V ≥ Vθ)
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
    @. integ.spike_count += integ.spiked
    _record_spikes!(integ.spike_rec, integ)
    _record_voltage!(integ.voltage_rec, integ)
    return nothing
end
@inline _record_spikes!(::Nothing, integ) = nothing
@inline function _record_spikes!(rec, integ)
    @inbounds if integ.n < size(rec, 2)
        rec[:, integ.n + 1] .= integ.spiked          # column dotview write (GPU-safe)
    end
    return nothing
end
@inline _record_voltage!(::Nothing, integ) = nothing
@inline function _record_voltage!(rec, integ)
    @inbounds if integ.n < size(rec, 2)
        rec[:, integ.n + 1] .= integ.state.state.V
    end
    return nothing
end

# Compile-time unroll of the schedule (carried in its type parameter) into a straight-line
# sequence of phase calls --- no runtime Symbol comparison, no dynamic dispatch.
@generated function run_phases!(::Schedule{P}, integ::DewdropIntegrator) where {P}
    calls = [:(run_phase!(Val($(QuoteNode(ph))), integ)) for ph in P]
    return Expr(:block, calls..., :(return nothing))
end

function CommonSolve.step!(integ::DewdropIntegrator)
    run_phases!(integ.schedule, integ)
    integ.n += 1
    integ.t += integ.dt
    return integ
end

function CommonSolve.solve!(integ::DewdropIntegrator)
    while integ.t < integ.tend - integ.dt / 2
        step!(integ)
    end
    return DewdropSolution(integ)
end

"""
    DewdropSolution

The result of a run: final SoA `state`, per-unit `spike_count`, number of steps, `dt`,
and `tspan`. See [`firing_rate`](@ref).
"""
struct DewdropSolution{ST, C, T, SP, VO}
    state::ST
    spike_count::C
    nsteps::Int
    dt::T
    tspan::Tuple{T, T}
    spikes::SP        # (N, nsteps) Bool raster if recorded, else `nothing`
    voltages::VO      # (N, nsteps) V trace if recorded, else `nothing`
end
function DewdropSolution(integ::DewdropIntegrator)
    return DewdropSolution(
        integ.state, integ.spike_count, integ.n, integ.dt,
        (integ.t - integ.n * integ.dt, integ.t), integ.spike_rec, integ.voltage_rec,
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
    raster(sol) -> (times, ids)

Extract spike events from a solution recorded with `record_spikes=true`: `times[k]` and
`ids[k]` are the time and neuron index of the k-th spike (host-side analysis helper).
"""
function raster(sol::DewdropSolution)
    sol.spikes === nothing && error("no spikes recorded --- pass `record_spikes = true` to `solve`")
    idx = findall(sol.spikes)                       # CartesianIndex(neuron, step)
    times = [I[2] * sol.dt for I in idx]
    ids = [I[1] for I in idx]
    return times, ids
end
export raster
