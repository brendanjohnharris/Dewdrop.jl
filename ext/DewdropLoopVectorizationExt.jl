module DewdropLoopVectorizationExt

# The `Turbo` backend's SIMD kernels (Tier 2). `using LoopVectorization` activates this extension,
# which registers a `@turbo`-vectorised dense membrane+threshold+reset+count kernel for the built-in
# model families. The orchestration (deliver / accumulate / decay / scatter) stays in the core
# `_turbo_step!`; only the per-neuron dense update is vectorised here. The kernels REPLICATE the
# engine's scalar math exactly (the w-first split, the exact-propagator `_coba_step`, the AdEx cutoff)
# so results are spike-identical --- they differ only at the `exp` ULP (SLEEF SIMD exp vs scalar
# `libm`), which is why `Turbo` is opt-in and never an `Auto` default.
#
# To give a NEW model a Turbo specialization: define `Dewdrop.turbo_kernel(::Type{MyModel}) =
# my_turbo_kernel`, where `my_turbo_kernel(integ)` runs `@turbo for i in eachindex(V) … end` over the
# integrator's SoA (`integ.state.state`), reading the accumulated `integ.itot`/`integ.gtot` and
# writing `V`/aux/`refrac`/`spiked`/`spike_count`. Keep the body branch-free (`ifelse`, `min`) so it
# vectorises. See `_turbo_adex!`/`_turbo_lif!` below as templates.

using LoopVectorization
import Dewdrop
import Dewdrop: AdEx, LIF, turbo_kernel, _ADEX_EXP_CAP

# --- registration: which built-in models have a Turbo specialization ---
turbo_kernel(::Type{<:AdEx}) = _turbo_adex!
turbo_kernel(::Type{<:LIF}) = _turbo_lif!

# AdEx (V, w): the w-first split + the exact-propagator membrane with the exponential forcing term.
# Reads `itot`/`gtot` (so it serves CUBA gtot=0 AND COBA), writes V/w/refrac/spiked/spike_count.
function _turbo_adex!(integ)
    st = integ.state.state
    V, w, refrac, spiked, spk = st.V, st.w, st.refrac, integ.spiked, integ.spike_count
    itot, gtot = integ.itot, integ.gtot
    m = integ.model::AdEx
    dt = integ.dt
    C, gL, EL, VT, ΔT = m.C, m.gL, m.EL, m.VT, m.ΔT
    Vr, Vpeak, a, b, τw, tref = m.Vr, m.Vpeak, m.a, m.b, m.τw, m.tref
    R = inv(gL); τ = C / gL
    z = zero(eltype(V)); cap = eltype(V)(_ADEX_EXP_CAP)
    @turbo for i in eachindex(V)
        v = V[i]; wi = w[i]; r = refrac[i]; it = itot[i]; gt = gtot[i]
        # _step_w: w from the OLD V (exact relaxation toward a·(V−EL))
        w∞ = a * (v - EL)
        w2 = w∞ + (wi - w∞) * exp(-dt / τw)
        # _step_V: exponential forcing term + the COBA exact propagator (with the Vpeak cutoff)
        Iexp = gL * ΔT * exp(min((v - VT) / ΔT, cap))
        denom = 1 + R * gt
        V∞ = (EL + R * (it + Iexp - w2)) / denom
        vc = V∞ + (v - V∞) * exp(-dt * denom / τ)
        v_adv = ifelse(vc ≥ Vpeak, Vpeak, vc)
        # refractory clamp → threshold (at Vpeak) → reset + spike-triggered w increment
        v1 = ifelse(r > z, Vr, v_adv)
        r1 = max(r - dt, z)
        s = (r1 ≤ z) & (v1 ≥ Vpeak)
        V[i] = ifelse(s, Vr, v1)
        w[i] = ifelse(s, w2 + b, w2)
        refrac[i] = ifelse(s, tref, r1)
        spiked[i] = s
        spk[i] += s
    end
    return nothing
end

# LIF (V only): the COBA-capable exact propagator + hard threshold/reset.
function _turbo_lif!(integ)
    st = integ.state.state
    V, refrac, spiked, spk = st.V, st.refrac, integ.spiked, integ.spike_count
    itot, gtot = integ.itot, integ.gtot
    m = integ.model::LIF
    dt = integ.dt
    EL, R, τ, Vθ, Vr, tref = m.EL, m.R, m.τ, m.Vθ, m.Vr, m.tref
    z = zero(eltype(V))
    @turbo for i in eachindex(V)
        v = V[i]; r = refrac[i]; it = itot[i]; gt = gtot[i]
        denom = 1 + R * gt
        V∞ = (EL + R * it) / denom
        v_adv = V∞ + (v - V∞) * exp(-dt * denom / τ)        # membrane_step = _coba_step
        v1 = ifelse(r > z, Vr, v_adv)
        r1 = max(r - dt, z)
        s = (r1 ≤ z) & (v1 ≥ Vθ)
        V[i] = ifelse(s, Vr, v1)
        refrac[i] = ifelse(s, tref, r1)
        spiked[i] = s
        spk[i] += s
    end
    return nothing
end

end # module DewdropLoopVectorizationExt
