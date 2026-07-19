# * Synapse models. A synapse model defines how a delivered presynaptic spike affects
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
synapses, which couple to `V` via a reversal potential, are defined below.
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

Conductance-based (COBA) dual-exponential synapse: a delivered
spike of weight `w` kicks two accumulators `g_rise`/`g_decay`, which decay with rise time `τr` and
decay time `τd`; the conductance is `g(t) = a·(g_decay − g_rise)` (a difference of exponentials,
rise then decay) normalised by `a` so the peak conductance equals the delivered weight `w`. The
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

"""
    FrozenDualExpSynapse(; τr, τd, Erev)

Frozen-current variant of [`DualExpSynapse`](@ref): identical dual-exponential conductance kinetics
`g(t) = a·(g_decay − g_rise)`, but the synaptic current `g·(Erev − V)` is evaluated with `V` FROZEN at
its pre-update value and injected as an ordinary current; it does NOT enter the effective leak, so it
does not shunt the membrane time constant. This reproduces the BrainPy `sum_current_inputs`/`COBA`
integration. Exact COBA ([`DualExpSynapse`](@ref)) is the more accurate scheme (the conductance shunts);
use this only to reproduce frozen-current dynamics. A drop-in for `DualExpSynapse`. Requires `τr ≠ τd`.
"""
struct FrozenDualExpSynapse{T} <: AbstractSynapseModel
    τr::T
    τd::T
    Erev::T
end
function FrozenDualExpSynapse(; τr, τd, Erev)
    r, d, e = promote(to_time(τr), to_time(τd), to_voltage(Erev))
    r == d && throw(ArgumentError("FrozenDualExpSynapse requires τr ≠ τd (got τr = τd = $r); use an alpha synapse for equal time constants"))
    return FrozenDualExpSynapse(r, d, e)
end
export FrozenDualExpSynapse

# * Single-source synapse descriptor. Every execution path (serial broadcast, fused megakernel,
# batched (N,B) ensemble) is generated generically from five tiny trait methods per synapse; adding a
# synapse is these methods, not four hand-copied kernels. Each method reproduces the prior code
# byte-for-byte. The forward-compatible superset: `_syn_accumulators` allows K per-target channels,
# `_syn_membrane` reads `v` (V-dependent/nonlinear currents, e.g. NMDA), `_syn_couple` marks the
# coupling mode. Coefficient eltype wrapping matches the old `_make_synstate` exactly (a byte-identity
# requirement: CUBA/COBA keep `synapse_decay` unwrapped; the dual-exp family wraps in `T`).

# Per-target accumulator field names (`()` for a stateless voltage-jump synapse). K = length.
_syn_accumulators(::Type{<:CurrentSynapse}) = (:Isyn,)
_syn_accumulators(::Type{<:ConductanceSynapse}) = (:g,)
_syn_accumulators(::Type{DeltaSynapse}) = ()
_syn_accumulators(::Type{<:DualExpSynapse}) = (:g_rise, :g_decay)
_syn_accumulators(::Type{<:FrozenDualExpSynapse}) = (:g_rise, :g_decay)

# Coupling mode: how a delivered event enters the cell. Drives the serial deliver skeleton and the
# fused voltage-jump short-circuit. `:current` feeds itot only; `:conductance` feeds gtot + itot;
# `:jump` is an instantaneous voltage jump with no accumulator (delta).
_syn_couple(::Type{<:CurrentSynapse}) = Val(:current)
_syn_couple(::Type{<:ConductanceSynapse}) = Val(:conductance)
_syn_couple(::Type{DeltaSynapse}) = Val(:jump)
_syn_couple(::Type{<:DualExpSynapse}) = Val(:conductance)
_syn_couple(::Type{<:FrozenDualExpSynapse}) = Val(:current)   # frozen current g·(Erev−V), no shunt

# Per-step derived coefficients, wrapped to EXACTLY the current stored eltype (byte-identity).
_syn_coeffs(s::CurrentSynapse, dt, ::Type{T}) where {T} = (; decay = synapse_decay(s, dt))
_syn_coeffs(s::ConductanceSynapse, dt, ::Type{T}) where {T} = (; decay = synapse_decay(s, dt), Erev = T(s.Erev))
_syn_coeffs(::DeltaSynapse, dt, ::Type{T}) where {T} = (;)
_syn_coeffs(s::DualExpSynapse, dt, ::Type{T}) where {T} =
    (; decay_r = T(exp(-dt / s.τr)), decay_d = T(exp(-dt / s.τd)), a = T(_dualexp_a(s.τr, s.τd)), Erev = T(s.Erev))
_syn_coeffs(s::FrozenDualExpSynapse, dt, ::Type{T}) where {T} =
    (; decay_r = T(exp(-dt / s.τr)), decay_d = T(exp(-dt / s.τd)), a = T(_dualexp_a(s.τr, s.τd)), Erev = T(s.Erev))

# Membrane coupling: the (Δgtot, Δitot) this synapse contributes given its CURRENT accumulator values
# `acc` (a tuple), coefficients `c`, and the (frozen) membrane potential `v`. `v` is threaded so a
# V-dependent current (frozen COBA today; NMDA later) needs no new seam. The frozen-vs-exact dual split
# is these two methods; `false` is a strong zero (gtot untouched, bit-identical + type-preserving).
@inline _syn_membrane(::CurrentSynapse, acc, c, v) = (false, acc[1])
@inline _syn_membrane(s::ConductanceSynapse, acc, c, v) = (acc[1], acc[1] * c.Erev)
@inline _syn_membrane(::DeltaSynapse, acc, c, v) = (false, false)   # jump mode; never called in the inner loop
@inline function _syn_membrane(s::DualExpSynapse, acc, c, v)
    g = c.a * (acc[2] - acc[1])
    return (g, g * c.Erev)
end
@inline function _syn_membrane(s::FrozenDualExpSynapse, acc, c, v)
    g = c.a * (acc[2] - acc[1])
    return (false, g * (c.Erev - v))
end

# Per-accumulator decay: the new accumulator tuple. Diagonal (each channel × its own coefficient) for
# every current synapse; a coupled `propagate` (GABA-B, alpha) would override this. Called with the real
# accumulator values (fused path) or a unit tuple (serial path extracts the diagonal coefficients).
@inline _syn_decay(::CurrentSynapse, acc, c) = (acc[1] * c.decay,)
@inline _syn_decay(::ConductanceSynapse, acc, c) = (acc[1] * c.decay,)
@inline _syn_decay(::DeltaSynapse, acc, c) = ()
@inline _syn_decay(::Union{DualExpSynapse, FrozenDualExpSynapse}, acc, c) = (acc[1] * c.decay_r, acc[2] * c.decay_d)
