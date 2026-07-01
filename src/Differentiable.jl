# * Differentiable execution backend (surrogate-gradient SNN).
#
# `backend = Differentiable()` makes a CPU forward pass automatically differentiable so a scalar loss
# can be back-propagated to the model parameters through the whole time loop (gradient-based fitting and
# surrogate-gradient training). It is a SEPARATE step path: the default `_fused_unit!` / `_tight_step!`
# are left byte-for-byte untouched, so every other backend stays bit-identical BY CONSTRUCTION.
#
# Only two things change versus the bit-identical engine:
#   1. the discontinuous spike (hard threshold + reset + integer count) becomes a smooth fast-sigmoid
#      surrogate + soft reset + real-valued count (this file);
#   2. `spiked` / `spike_count` take the state float type so a `Dual`/`Active` eltype flows (Backends.jl
#      `_spiked_eltype` / `_count_eltype`, wired through `init`).
# The eltype-genericity needed for AD is already in the engine: `float_type(model)` flows to every state
# column via `Population`/`allocate`, so a `Dual`-typed model gives a `Dual`-typed run with no further
# change. Pair with `ForwardDiff` (few parameters) or `Enzyme` reverse-mode (many; the scalable path).
#
# NEXT STEPS (designed, not yet built): a surrogate-WEIGHTED scatter (`accum += weight·s`) to make
# synaptic weights trainable in recurrent nets; an Enzyme weak-dep extension with the atomic-scatter
# reverse rule; and the GPU path (Enzyme through the KA megakernel — the open hardware question).

# the surrogate: a smooth, in-[0,1] replacement for the Heaviside threshold at the model's Vθ
"""
    threshold_voltage(model) -> V

The voltage the surrogate spike centres on (the model's spike threshold). Default models: `LIF`/`AdaptLIF`
fire at `Vθ`, `AdEx` at `Vpeak`. Define a method for a custom model to make it trainable.
"""
function threshold_voltage end
@inline threshold_voltage(m::LIF) = m.Vθ
@inline threshold_voltage(m::AdaptLIF) = m.Vθ
@inline threshold_voltage(m::AdEx) = m.Vpeak

"""
    surrogate_spike(model, V, β) -> s ∈ (0, 1)

The fast-sigmoid surrogate `1 / (1 + exp(-β·(V - threshold_voltage(model))))`: a smooth, differentiable
stand-in for the spike indicator, with steepness `β`. As `β → ∞` it approaches the true Heaviside.
"""
@inline surrogate_spike(m, V, β) = inv(one(V) + exp(-β * (V - threshold_voltage(m))))

# soft spike-triggered adaptation bump — the real-valued-`s` mirror of `_spike_aux`: `w + s·Δ`
# (with `s ∈ {0,1}` this equals the hard `ifelse(spiked, w + Δ, w)`).
@inline _diff_spike_aux(m, ::Nothing, s) = nothing
@inline _diff_spike_aux(m, w, s) = w + s * spike_increment(m)

# One surrogate per-neuron step. Identical to `_fused_unit!` up to the spike block (marked below), which
# swaps the hard threshold + reset + integer count for the smooth surrogate + soft reset + real count.
@inline function _diff_unit!(V, refrac, spiked, spike_count, input, syns, m, dt, n, drive, noise, aux, β, i)
    @inbounds begin
        v = V[i]
        r = refrac[i]
        z = zero(r)
        gtot = zero(eltype(V))
        itot = oftype(gtot, _inputval(input, i))
        v, gtot, itot = _syn_contribute(syns, i, n, v, gtot, itot)   # no-op for an unconnected population
        v += _drive_kick(drive, n, i, dt)
        m_i = _resolve(m, i)
        w0 = _aux_read(aux, i)
        v_adv, w_adv = _advance_unit(m_i, v, w0, gtot, itot, dt)
        v = ifelse(r > z, reset_value(m_i), v_adv + _noise_kick(noise, n, i, dt, m_i))
        r = max(r - dt, z)
        # surrogate spike block (the ONLY difference from `_fused_unit!`)
        s = oftype(v, r ≤ z) * surrogate_spike(m_i, v, β)            # refractory-gated smooth spike ∈ [0,1)
        v = v - s * (v - reset_value(m_i))                           # soft reset (→ hard reset as s → 1)
        w_new = _diff_spike_aux(m_i, w_adv, s)                       # soft adaptation increment
        r = r + s * (refractory(m_i) - r)                           # soft refractory arm
        V[i] = v
        refrac[i] = r
        _aux_write!(aux, w_new, i)
        spiked[i] = s
        spike_count[i] += s                                          # real-valued surrogate count
    end
    return nothing
end

# Surrogate dense pass: single-threaded (clean for AD; the differentiable nets are small), then the
# propagate (a no-op for an unconnected population — the seam where a surrogate-weighted scatter will go)
# and monitors. Mirrors `_tight_step!`.
function _diff_step!(integ::DewdropIntegrator)
    st = integ.state.state
    m = integ.model
    V, refrac, spiked, spike_count = st.V, st.refrac, integ.spiked, integ.spike_count
    input, syns, dt, n = integ.input, integ.syns, integ.dt, integ.n
    drive, noise, aux = integ.drive, integ.noise, _aux_col(st, m)
    β = integ.backend.β
    @inbounds for i in 1:length(V)
        _diff_unit!(V, refrac, spiked, spike_count, input, syns, m, dt, n, drive, noise, aux, β, i)
    end
    _propagate_step!(integ.compaction, integ)                       # no-op (unconnected); scatter seam
    _record_all!(integ.monitors, integ)
    return nothing
end

# route the canonical-schedule CPU step to the surrogate path for `backend = Differentiable()`.
@inline _step_backend!(::Differentiable, ::_KA.CPU, integ::DewdropIntegrator) = _diff_step!(integ)
