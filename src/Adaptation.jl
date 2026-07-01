# * Adaptation neurons --- models that carry a spike-triggered adaptation current `w`
# alongside the membrane potential `V`. `AdaptLIF` is fully linear; `AdEx` adds the exponential
# spike-initiation term. Both advance `(V, w)` by the "w-first" symplectic split --- `w` from the
# OLD `V` (exact exponential relaxation), then `V` from the OLD V and the NEW `w` (the exact
# COBA propagator, with `-w` and AdEx's `Iexp` folded into the current). This split makes the
# broadcast path (two `@.` passes) and the per-neuron fused/batched kernels bit-identical, and
# it needs no refractory special-case for `w` (a refractory unit's old V is already clamped to Vr).
#
# The whole multi-state machinery is dispatched on whether the model carries `w` (`_has_w`), so
# LIF and every linear `@neuron` model take the V-only fast path UNCHANGED (the empty-aux methods
# below compile to exactly the prior code). The spike-triggered `w += b` lives in the `:reset`
# phase (broadcast) / inline after reset (kernels), as Schedule.jl's `:reset` doc reserves.

# AdEx's exp term is capped before evaluation so a unit between VT and Vpeak cannot overflow to
# Inf/NaN in the single step before the Vpeak cutoff fires (kept well inside Float32 range).
const _ADEX_EXP_CAP = 50.0

# --- AdaptLIF: linear adaptation (aLIF) ----------------------------------------------------------
"""
    AdaptLIF(; τ, EL, Vθ, Vr, R, tref, a, b, τw)

Adaptive LIF: `τ dV/dt = -(V - EL) + R·(I - w)`, `τw dw/dt = a·(V - EL) - w`; spike when `V ≥ Vθ`,
reset `V ← Vr` and increment `w ← w + b`. `a` is the subthreshold adaptation conductance, `b` the
spike-triggered current increment, `τw` the adaptation time constant. State: `V`, `refrac`, `w`.
"""
struct AdaptLIF{T} <: AbstractNeuronModel
    τ::T; EL::T; Vθ::T; Vr::T; R::T; tref::T
    a::T; b::T; τw::T
end
AdaptLIF(; τ, EL, Vθ, Vr, R, tref, a, b, τw) = AdaptLIF(
    promote(
        to_time(τ), to_voltage(EL), to_voltage(Vθ), to_voltage(Vr), to_resistance(R), to_time(tref),
        to_conductance(a), to_current(b), to_time(τw)
    )...
)
export AdaptLIF

statevars(::Type{<:AdaptLIF}) = (:V, :refrac, :w)
float_type(::AdaptLIF{T}) where {T} = T
@inline threshold(m::AdaptLIF, V) = V ≥ m.Vθ
@inline reset_value(m::AdaptLIF) = m.Vr
@inline refractory(m::AdaptLIF) = m.tref
@inline _tau(m::AdaptLIF) = m.τ
@inline spike_increment(m::AdaptLIF) = m.b

# `V` from the old V and the new w (`-w` is an outward current folded into `itot`, so conductance
# synapses via `gtot` still work); `w` itself uses the shared `_step_w` below (defined after AdEx).
@inline _step_V(m::AdaptLIF, V, w, gtot, itot, dt) = _coba_step(V, m.EL, m.R, m.τ, gtot, itot - w, dt)

# --- AdEx: adaptive exponential integrate-and-fire (Brette & Gerstner 2005) ----------------------
"""
    AdEx(; C, gL, EL, VT, ΔT, Vr, Vpeak, a, b, τw, tref=0)

Adaptive exponential IF: `C dV/dt = -gL(V - EL) + gL·ΔT·exp((V - VT)/ΔT) + I - w`,
`τw dw/dt = a·(V - EL) - w`; the exponential drives `V` past `Vpeak` (the numerical spike cutoff,
not `VT`), then reset `V ← Vr`, `w ← w + b`. The subthreshold step is exponential-Euler: the
exp term is a forcing current at the pre-step `V`, the linear part uses the exact propagator
(`R = 1/gL`, `τ = C/gL`). State: `V`, `refrac`, `w`.
"""
struct AdEx{T} <: AbstractNeuronModel
    C::T; gL::T; EL::T; VT::T; ΔT::T; Vr::T; Vpeak::T
    a::T; b::T; τw::T; tref::T
end
AdEx(; C, gL, EL, VT, ΔT, Vr, Vpeak, a, b, τw, tref = 0.0) = AdEx(
    promote(
        to_capacitance(C), to_conductance(gL), to_voltage(EL), to_voltage(VT), to_voltage(ΔT),
        to_voltage(Vr), to_voltage(Vpeak), to_conductance(a), to_current(b), to_time(τw), to_time(tref)
    )...
)
export AdEx

statevars(::Type{<:AdEx}) = (:V, :refrac, :w)
float_type(::AdEx{T}) where {T} = T
@inline threshold(m::AdEx, V) = V ≥ m.Vpeak     # fires at the cutoff Vpeak (the exp has diverged), not VT
@inline reset_value(m::AdEx) = m.Vr
@inline refractory(m::AdEx) = m.tref
@inline _tau(m::AdEx) = m.C / m.gL
@inline spike_increment(m::AdEx) = m.b

# Shared adaptation-current relaxation (AdaptLIF & AdEx): exact exponential decay of `w` toward its
# fixpoint a·(V−EL) over τw, at the OLD V (the w-first split). FNSNeuron's gK uses its own `_step_w`.
@inline function _step_w(m::Union{AdaptLIF, AdEx}, V, w, dt)
    w∞ = m.a * (V - m.EL)
    return w∞ + (w - w∞) * exp(-dt / m.τw)
end
@inline function _step_V(m::AdEx, V, w, gtot, itot, dt)
    Iexp = m.gL * m.ΔT * exp(min((V - m.VT) / m.ΔT, oftype(V, _ADEX_EXP_CAP)))   # exp term as a forcing current
    Vn = _coba_step(V, m.EL, inv(m.gL), m.C / m.gL, gtot, itot + Iexp - w, dt)
    return ifelse(Vn ≥ m.Vpeak, m.Vpeak, Vn)                                     # clamp at the cutoff (no Inf)
end

# --- FNSNeuron: conductance-adaptation LIF (Treves-style) ------------------
"""
    FNSNeuron(; C, gL, VL, VK, Vθ, Vr, tref, τK, ΔgK)

Conductance-adaptation LIF: `C dV/dt = -gL(V - VL) - gK(V - VK) + I`, `τK dgK/dt = -gK`; spike when
`V ≥ Vθ`, reset `V ← Vr` and increment the adaptation conductance `gK ← gK + ΔgK`. Unlike `AdaptLIF`
(whose `w` is a current), the adaptation here is a CONDUCTANCE with reversal `VK`, folded into the
exact COBA propagator as an extra leak `gK` plus reversal drive `gK·VK`. `ΔgK = 0` gives a plain
conductance-LIF (the inhibitory population). State: `V`, `refrac`, `w` (the generic aux column holds
`gK`). Defaults follow the Treves-style FNS neuron.
"""
struct FNSNeuron{T} <: AbstractNeuronModel
    C::T; gL::T; VL::T; VK::T; Vθ::T; Vr::T; tref::T; τK::T; ΔgK::T
end
FNSNeuron(;
    C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0,
    tref = 4.0, τK = 80.0, ΔgK = 0.01
) = FNSNeuron(
    promote(
        to_capacitance(C), to_conductance(gL), to_voltage(VL), to_voltage(VK), to_voltage(Vθ), to_voltage(Vr),
        to_time(tref), to_time(τK), to_conductance(ΔgK)
    )...
)
export FNSNeuron

statevars(::Type{<:FNSNeuron}) = (:V, :refrac, :w)            # the generic aux column :w holds gK
float_type(::FNSNeuron{T}) where {T} = T
@inline threshold(m::FNSNeuron, V) = V ≥ m.Vθ
@inline reset_value(m::FNSNeuron) = m.Vr
@inline refractory(m::FNSNeuron) = m.tref
@inline _resting(m::FNSNeuron) = m.VL                         # rests at the leak reversal VL (no EL field)
@inline _tau(m::FNSNeuron) = m.C / m.gL
@inline spike_increment(m::FNSNeuron) = m.ΔgK

# gK relaxes to 0 with τK (no V-coupling, w∞ = 0); the membrane folds gK in as an extra conductance
# with reversal drive gK·VK (w-first split: gK from old, V from old V + new gK).
@inline _step_w(m::FNSNeuron, V, w, dt) = w * exp(-dt / m.τK)
@inline _step_V(m::FNSNeuron, V, w, gtot, itot, dt) =
    _coba_step(V, m.VL, inv(m.gL), m.C / m.gL, gtot + w, itot + w * m.VK, dt)

# --- The shared (V,w) advance, w-first --- used by every execution path ---
# `advance_state` is the per-neuron scalar step for an adaptation model: w from old V, V from new w.
@inline function advance_state(m, V, w, gtot, itot, dt)
    w2 = _step_w(m, V, w, dt)
    V2 = _step_V(m, V, w2, gtot, itot, dt)
    return (V2, w2)
end

# --- The aux-state seam: route on whether the model carries a `w` column ---
# `_has_w` is a compile-time bool from the model's statevars; the `st.w` accesses below appear ONLY
# in the carries-`w` methods, so a V-only model (LIF / @neuron) never instantiates them and keeps
# its prior code byte-for-byte. The carries-`w` value is `nothing` for V-only, a scalar otherwise.
@inline _has_w(::Type{M}) where {M} = (:w in statevars(M))

# --- per-neuron heterogeneity hooks (see Heterogeneous.jl). A scalar model resolves to
# itself per-neuron (bit-identical, zero-cost), is not heterogeneous, and rests at its leak reversal. ---
@inline _resolve(m::AbstractNeuronModel, i) = m
@inline _is_hetero(::AbstractNeuronModel) = false
@inline _resting(m::AbstractNeuronModel) = m.EL

# select the adaptation column for the kernels (or `nothing`); dispatched so `st.w` never compiles
# for a V-only model.
@inline _aux_col(st, m::AbstractNeuronModel) = _aux_col(st, Val(_has_w(typeof(m))))
@inline _aux_col(st, ::Val{false}) = nothing
@inline _aux_col(st, ::Val{true}) = st.w

@inline _aux_read(::Nothing, idx::Vararg{Integer}) = nothing
@inline _aux_read(w::AbstractArray, idx::Vararg{Integer}) = @inbounds w[idx...]
@inline _aux_write!(::Nothing, val, idx::Vararg{Integer}) = nothing
@inline _aux_write!(w::AbstractArray, val, idx::Vararg{Integer}) = (@inbounds w[idx...] = val; nothing)

# per-neuron subthreshold advance: V-only (w === nothing) is exactly `membrane_step`; otherwise the
# (V,w) co-step. Keeps the V-only fast path bit-identical.
@inline _advance_unit(m, V, ::Nothing, gtot, itot, dt) = (membrane_step(m, V, gtot, itot, dt), nothing)
@inline _advance_unit(m, V, w, gtot, itot, dt) = advance_state(m, V, w, gtot, itot, dt)

# spike-triggered increment on the auxiliary state (no-op for V-only)
@inline _spike_aux(m, ::Nothing, spiked) = nothing
@inline _spike_aux(m, w, spiked) = ifelse(spiked, w + spike_increment(m), w)

# --- Broadcast-path helpers (called from Engine.jl's :integrate / :reset phases) ---
# V-only: the exact prior single broadcast (byte-identical). Carries-w: the two-pass w-first split.
@inline _integrate_membrane!(m, st, gtot, itot, dt, refrac, Vr, z) =
    _integrate_membrane!(Val(_has_w(typeof(m))), m, st, gtot, itot, dt, refrac, Vr, z)
@inline function _integrate_membrane!(::Val{false}, m, st, gtot, itot, dt, refrac, Vr, z)
    V = st.V
    @. V = ifelse(refrac > z, Vr, membrane_step(m, V, gtot, itot, dt))
    return nothing
end
@inline function _integrate_membrane!(::Val{true}, m, st, gtot, itot, dt, refrac, Vr, z)
    @. st.w = _step_w(m, st.V, st.w, dt)                                        # w from old V
    @. st.V = ifelse(refrac > z, Vr, _step_V(m, st.V, st.w, gtot, itot, dt))    # V from old V, new w
    return nothing
end

# spike-triggered increment in the :reset phase (no-op for V-only models)
@inline _reset_aux!(m, st, spiked) = _reset_aux!(Val(_has_w(typeof(m))), m, st, spiked)
@inline _reset_aux!(::Val{false}, m, st, spiked) = nothing
@inline function _reset_aux!(::Val{true}, m, st, spiked)
    @. st.w = ifelse(spiked, st.w + spike_increment(m), st.w)
    return nothing
end
