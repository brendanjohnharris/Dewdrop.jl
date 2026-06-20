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
LIF(; τ, EL, Vθ, Vr, R, tref) = LIF(promote(τ, EL, Vθ, Vr, R, tref)...)
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

# --- Threshold / reset / refractory ---
@inline threshold(m::LIF, V) = V ≥ m.Vθ
@inline reset_value(m::LIF) = m.Vr
@inline refractory(m::LIF) = m.tref

# Type-stable state allocation: a SoA `Population` with one zero-initialised column per
# `statevars(model)`, the column names resolved at COMPILE time from the model type (so
# `init` and the hot loop see a concrete StructArray, not `Population{S} where S`).
@generated function Population(arch::AbstractArchitecture, model::AbstractNeuronModel, N::Integer)
    names = statevars(model)               # `model` is the TYPE inside the generator
    # Build the column tuple explicitly (a @generated return AST may not contain a closure,
    # comprehension or generator, which rules out `ntuple(_ -> ..., K)`).
    colexpr = :(fill!(allocate(arch, T, N), zero(T)))
    cols = Expr(:tuple, fill(colexpr, length(names))...)
    return quote
        T = float_type(model)
        return Population(StructArray(NamedTuple{$names}($cols)))
    end
end
