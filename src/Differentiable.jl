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
# The surrogate-WEIGHTED scatter (`slots += weight·s_pre`, `_surrogate_scatter!` below) makes synaptic
# weights trainable in CONNECTED / recurrent nets: build the connectome with a `Dual`/`Active` weight eltype
# and the gradient flows to the weights (and, through `s_pre`, back to the presynaptic voltages). NEXT STEPS
# (designed, not yet built): an Enzyme weak-dep extension for scalable reverse-mode (many weights); and the
# GPU path (Enzyme through the KA megakernel — the open hardware question).

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
@inline function _diff_unit!(V, refrac, spiked, spike_count, stimuli, syns, m, dt, n, t, aux, β, i)
    @inbounds begin
        v = V[i]
        r = refrac[i]
        z = zero(r)
        gtot = zero(eltype(V))
        x0 = stim_ctx(m, v, i, 1, n, t, dt, 0)
        itot = oftype(gtot, _stim_itot(stimuli, gtot, x0))
        v, gtot, itot = _syn_contribute(syns, i, n, v, gtot, itot)   # no-op for an unconnected population
        Δg, Δi = _stim_gtot(stimuli, x0)
        gtot = _addcond(gtot, Δg)
        itot = _addcond(itot, Δi)
        v += _stim_kick(stimuli, x0)
        m_i = _resolve(m, i)
        w0 = _aux_read(aux, i)
        v_adv, w_adv = _advance_unit(m_i, v, w0, gtot, itot, dt)
        v = ifelse(r > z, reset_value(m_i), v_adv + _stim_noise(stimuli, stim_ctx(m_i, v_adv, i, 1, n, t, dt, 0)))
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

# Surrogate-weighted scatter: the connected-training seam. Deposit `weight·s_pre` into the delay ring,
# UNGATED — a serial per-edge walk (the differentiable path is CPU-only and single-threaded, so no atomics,
# AD-safe). The regular `scatter!` gates on `spiked[pre] || continue`, which a real-valued / `Dual` surrogate
# spike cannot take; here `iszero(s)` short-circuits the exact-zero (refractory / far-subthreshold) case,
# recovering the spike sparsity without a boolean gate. Differentiable through BOTH `weight[e]` (→ trainable
# synapses, when the connectome weight is a `Dual`/`Active` eltype) AND the presynaptic surrogate spike
# `s_pre` (→ gradients flow back to V_pre); the scatter's adjoint is a gather, handled implicitly by the AD.
@inline function _surrogate_scatter!(buf::DelayBuffer, conn::SparseCSR, spiked, now::Integer)
    slots, L = buf.slots, buf.L
    rowptr, post, weight, delay = conn.rowptr, conn.post, conn.weight, conn.delay
    n = Int(now)
    @inbounds for pre in eachindex(spiked)
        s = spiked[pre]
        iszero(s) && continue
        for e in rowptr[pre]:(rowptr[pre + 1] - 1)
            slots[post[e], mod(n + delay[e], L) + 1] += weight[e] * s
        end
    end
    return nothing
end
@inline _surrogate_propagate!(syn::AbstractSynapseState, integ) = (_surrogate_scatter!(syn.buf, syn.conn, integ.spiked, integ.n); nothing)
@inline _surrogate_propagate_all!(::Tuple{}, integ) = nothing
@inline _surrogate_propagate_all!(s::Tuple, integ) = (_surrogate_propagate!(first(s), integ); _surrogate_propagate_all!(Base.tail(s), integ))

# Surrogate dense pass: single-threaded (clean for AD; the differentiable nets are small), then the
# surrogate-weighted scatter (a no-op for an unconnected population; the trainable-synapse seam for a
# connected one) and monitors. Mirrors `_tight_step!`.
function _diff_step!(integ::DewdropIntegrator)
    st = integ.state.state
    m = integ.model
    V, refrac, spiked, spike_count = st.V, st.refrac, integ.spiked, integ.spike_count
    stimuli, syns, dt, n = integ.stimuli, integ.syns, integ.dt, integ.n
    t, aux = _step_time(integ), _aux_col(st, m)
    β = integ.backend.β
    @inbounds for i in 1:length(V)
        _diff_unit!(V, refrac, spiked, spike_count, stimuli, syns, m, dt, n, t, aux, β, i)
    end
    _surrogate_propagate_all!(integ.syns, integ)                    # ungated weighted deposit (connected training)
    _record_all!(integ.monitors, integ)
    return nothing
end

# route the canonical-schedule CPU step to the surrogate path for `backend = Differentiable()`.
@inline _step_backend!(::Differentiable, ::_KA.CPU, integ::DewdropIntegrator) = _diff_step!(integ)
