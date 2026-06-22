# * Neuron models (M1) --- "model as code": a small isbits parameter struct plus pure,
# scalar, allocation-free functions for its dynamics, threshold and reset.
#
# The subthreshold update is deliberately structured as the EXACT linear propagator
# (Rotter--Diesmann) for the linear part of the dynamics, kept distinct from the
# discontinuous reset. For models with nonlinear coupling (e.g. AdEx's adaptation
# variable) the extension point is a symplectic-Euler coupling step layered on top of
# this propagator (Baronig et al. 2025) --- LIF, being fully linear, needs only the
# propagator.

"""
    AbstractNeuronModel

A neuron model is an isbits struct of parameters with methods [`statevars`](@ref),
[`float_type`](@ref), [`threshold`](@ref), [`reset_value`](@ref), [`refractory`](@ref)
and a subthreshold propagator. Models broadcast as scalars.
"""
abstract type AbstractNeuronModel end
Base.Broadcast.broadcastable(m::AbstractNeuronModel) = Ref(m)

"""
    statevars(model) -> NTuple{K,Symbol}

The per-unit state variable names (the SoA column names) the model needs. Defined per
model TYPE so the names are available at compile time for type-stable state allocation;
the instance form delegates to the type form.
"""
function statevars end
statevars(m::AbstractNeuronModel) = statevars(typeof(m))

"""
    float_type(model) -> Type

The floating-point type of the model's parameters and state.
"""
function float_type end

"""
    LIF(; τ, EL, Vθ, Vr, R, tref)

Leaky integrate-and-fire neuron: `τ dV/dt = -(V - EL) + R·I`; spike when `V ≥ Vθ`;
reset to `Vr`; absolute refractory period `tref`. Parameters share a float type `T`
(plain floats --- units are handled at the API boundary). State variables: `V`, `refrac`.
"""
struct LIF{T} <: AbstractNeuronModel
    τ::T
    EL::T
    Vθ::T
    Vr::T
    R::T
    tref::T
end
LIF(; τ, EL, Vθ, Vr, R, tref) = LIF(promote(
    to_time(τ), to_voltage(EL), to_voltage(Vθ), to_voltage(Vr), to_resistance(R), to_time(tref))...)
export LIF

statevars(::Type{<:LIF}) = (:V, :refrac)
float_type(::LIF{T}) where {T} = T

# --- Linear subsystem: the EXACT propagator (exact for LIF over dt at constant input) ---
"""
    asymptote(model, I) -> V∞

The steady-state membrane potential `V∞` for constant input `I` (LIF: `EL + R·I`).
"""
@inline asymptote(m::LIF, I) = m.EL + m.R * I

"""
    propagator_decay(model, dt) -> decay

The precomputable propagator coefficient `exp(-dt/τ)` for the linear subsystem.
"""
@inline propagator_decay(m::LIF, dt) = exp(-dt / m.τ)

"""
    subthreshold_step(V, V∞, decay) -> V

One exact linear-propagator update toward the fixed point `V∞`:
`V ← V∞ + (V - V∞)·decay`.
"""
@inline subthreshold_step(V, V∞, decay) = V∞ + (V - V∞) * decay

# COBA-capable exact subthreshold step (shared by LIF and the adaptation models): conductances
# set an effective leak (`denom`) and reversal drive, while `itot` carries every current term
# (external input, synaptic current, and --- for the adaptation models --- the adaptation current
# `-w` and AdEx's exponential term). With no conductance (gtot = 0) this is the plain exact
# propagator. Lives here (not in Engine.jl) so the adaptation models in Adaptation.jl can reuse it.
@inline function _coba_step(V, EL, R, τ, gtot, itot, dt)
    denom = 1 + R * gtot
    V∞ = (EL + R * itot) / denom
    return V∞ + (V - V∞) * exp(-dt * denom / τ)
end

# --- Threshold / reset / refractory ---
@inline threshold(m::LIF, V) = V ≥ m.Vθ
@inline reset_value(m::LIF) = m.Vr
@inline refractory(m::LIF) = m.tref

# Type-stable state allocation: a SoA `Population` with one zero-initialised column per
# `statevars(model)`. The names are carried in a `Val` so they reach the constructor as a TYPE
# parameter (a concrete StructArray, not `Population{S} where S`); constant propagation folds
# `statevars(M)` for a concrete `M` at the call site. Unlike a `@generated` body --- whose
# generator runs in this module's world and so cannot see a `@neuron`-defined model's
# `statevars` --- this resolves at the call world, so user models work too.
function Population(arch::AbstractArchitecture, model::AbstractNeuronModel, N::Integer)
    return _build_population(arch, float_type(model), Int(N), Val(statevars(typeof(model))))
end
@inline function _build_population(arch, ::Type{T}, N::Int, ::Val{names}) where {T, names}
    cols = ntuple(_ -> fill!(allocate(arch, T, N), zero(T)), Val(length(names)))
    return Population(StructArray(NamedTuple{names}(cols)))
end
