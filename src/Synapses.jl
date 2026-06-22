# * Synapse models (M1b). A synapse model defines how a delivered presynaptic spike affects
# the postsynaptic state and how that state decays between spikes. The exact exponential
# decay is the linear propagator for the synaptic subsystem (same structure as the neuron's
# subthreshold step), kept distinct from the spike-triggered jump (the delivered weight).

"""
    AbstractSynapseModel

A synapse model carries its parameters and provides [`synapse_decay`](@ref) (the
per-step propagator coefficient) and a rule for how a delivered weight enters the
postsynaptic state. Models broadcast as scalars.
"""
abstract type AbstractSynapseModel end
Base.Broadcast.broadcastable(s::AbstractSynapseModel) = Ref(s)

"""
    CurrentSynapse(; τ)

Current-based (CUBA) exponential synapse: a delivered spike of weight `w` adds `w` to the
postsynaptic synaptic current, which decays with time constant `τ` (`τ dI/dt = -I`) and
feeds the neuron's input current directly. The simplest synapse; conductance-based (COBA)
synapses, which couple to `V` via a reversal potential, arrive in M2.
"""
struct CurrentSynapse{T} <: AbstractSynapseModel
    τ::T
end
CurrentSynapse(; τ) = CurrentSynapse(to_time(τ))
export CurrentSynapse

"""
    synapse_decay(model, dt) -> decay

The exact per-step decay coefficient `exp(-dt/τ)` for the synaptic state (the linear
propagator of the synaptic current between spikes).
"""
@inline synapse_decay(s::CurrentSynapse, dt) = exp(-dt / s.τ)

"""
    DeltaSynapse()

Instantaneous (delta) synapse: a delivered spike of weight `w` adds `w` directly to the
postsynaptic membrane potential (a voltage jump), with no synaptic time constant. The
classic Brunel (2000) synapse, where the weight IS the PSP amplitude.
"""
struct DeltaSynapse <: AbstractSynapseModel end
export DeltaSynapse

"""
    ConductanceSynapse(; τ, Erev)

Conductance-based (COBA) exponential synapse: a delivered spike of weight `w` increments a
postsynaptic conductance that decays with time constant `τ`; the resulting synaptic current
is voltage-dependent, `g·(Erev − V)`, with reversal potential `Erev` (e.g. 0 mV for
excitation, −80 mV for inhibition).
"""
struct ConductanceSynapse{T} <: AbstractSynapseModel
    τ::T
    Erev::T
end
function ConductanceSynapse(; τ, Erev)
    t, e = promote(to_time(τ), to_voltage(Erev))
    return ConductanceSynapse(t, e)
end
export ConductanceSynapse

@inline synapse_decay(s::ConductanceSynapse, dt) = exp(-dt / s.τ)

"""
    DualExpSynapse(; τr, τd, Erev)

Conductance-based (COBA) dual-exponential synapse (the WRCircuit `DualExponV2` kinetic): a delivered
spike of weight `w` kicks two accumulators `g_rise`/`g_decay`, which decay with rise time `τr` and
decay time `τd`; the conductance is `g(t) = a·(g_decay − g_rise)` --- a difference of exponentials
(rise then decay) --- normalised by `a` so the peak conductance equals the delivered weight `w`. The
synaptic current is voltage-dependent, `g·(Erev − V)`. Requires `τr ≠ τd`.
"""
struct DualExpSynapse{T} <: AbstractSynapseModel
    τr::T
    τd::T
    Erev::T
end
function DualExpSynapse(; τr, τd, Erev)
    r, d, e = promote(to_time(τr), to_time(τd), to_voltage(Erev))
    r == d && throw(ArgumentError("DualExpSynapse requires τr ≠ τd (got τr = τd = $r); use an alpha synapse for equal time constants"))
    return DualExpSynapse(r, d, e)
end
export DualExpSynapse

# peak-normalising coefficient: `a·(e^{-t/τd} − e^{-t/τr})` has continuous peak 1, so the delivered
# weight is the peak conductance change (BrainPy's default `A`).
@inline _dualexp_a(τr, τd) = (τd / (τd - τr)) * (τr / τd)^(τr / (τr - τd))
