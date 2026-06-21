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
