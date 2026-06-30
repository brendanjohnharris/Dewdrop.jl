# * Fused device step (M6 Tier-1) --- the launch-bound fix for the GPU hot path.
#
# At small/medium N the per-step cost is dominated by KERNEL LAUNCHES, not compute: the
# broadcast-per-phase engine issues ~10 dense launches per step (deliver, drive, accumulate,
# membrane, refrac, decay, threshold, reset, count) plus the sparse scatter and the monitors,
# and the device `scatter!`/`_aggregate!` each `synchronize` (a host round-trip) every step.
#
# This module collapses the DENSE per-neuron phases of the canonical schedule into ONE
# KernelAbstractions megakernel and removes the per-step host synchronisation, so the step
# sequence pipelines on a single device stream (host syncs only at the windowed monitor flush
# and at host reads). The SPARSE scatter stays a separate kernel (folding it in would serialise
# it), and the monitors stay separate (their windowed device→host machinery is unchanged).
#
# It is a DEVICE-ONLY fast path: dispatched on `get_backend(V)`, the CPU backend keeps the
# tuned broadcast phases (CPU is not launch-bound, and this avoids any change to the CPU
# validation path). The fused kernel is model-generic (it calls the same `membrane_step` /
# `threshold` / `reset_value` / `refractory` hooks), so `@neuron` models fuse too, and it is
# numerically identical to the broadcast path (same operations, same order) up to the `exp` ULP
# difference between CPU `libm` and CUDA `libdevice` already accounted for in the GPU tests.

import KernelAbstractions as _KA
using KernelAbstractions: @kernel, @index

# Per-projection synaptic contribution for neuron `i`, unrolled over the projection tuple at
# compile time (mirrors the broadcast path's deliver → accumulate → decay, fused per neuron):
#   - read this projection's ring-buffer slot due at step `n` and clear it (the deliver),
#   - fold it into the running conductance `gtot` / input-current `itot` (the accumulate),
#   - decay the per-neuron synaptic state for the next step.
# CUBA contributes a decaying current; COBA a conductance (effective leak + reversal drive);
# delta an instantaneous voltage jump (added straight to `v`, no accumulator, no decay).
@inline _syn_contribute(::Tuple{}, i, n, v, gtot, itot) = (v, gtot, itot)
@inline function _syn_contribute(syns::Tuple, i, n, v, gtot, itot)
    v, gtot, itot = _syn_one(first(syns), i, n, v, gtot, itot)
    return _syn_contribute(Base.tail(syns), i, n, v, gtot, itot)
end

@inline function _syn_one(s::SynapseState, i, n, v, gtot, itot)        # CUBA (current)
    L = s.buf.L
    slot = mod(n, L) + 1
    @inbounds due = s.buf.slots[i, slot]
    @inbounds s.buf.slots[i, slot] = zero(due)
    @inbounds isyn = s.Isyn[i] + due                                  # deliver
    itot += isyn                                                      # accumulate
    @inbounds s.Isyn[i] = isyn * s.decay                             # decay
    return (v, gtot, itot)
end
@inline function _syn_one(s::COBAState, i, n, v, gtot, itot)           # COBA (conductance)
    L = s.buf.L
    slot = mod(n, L) + 1
    @inbounds due = s.buf.slots[i, slot]
    @inbounds s.buf.slots[i, slot] = zero(due)
    @inbounds g = s.g[i] + due                                        # deliver
    gtot += g                                                         # effective leak
    itot += g * s.Erev                                               # reversal drive
    @inbounds s.g[i] = g * s.decay                                   # decay
    return (v, gtot, itot)
end
@inline function _syn_one(s::DeltaSynapseState, i, n, v, gtot, itot)   # delta (voltage jump)
    L = s.buf.L
    slot = mod(n, L) + 1
    @inbounds due = s.buf.slots[i, slot]
    @inbounds s.buf.slots[i, slot] = zero(due)
    v += due
    return (v, gtot, itot)
end
@inline function _syn_one(s::DualExpCOBAState, i, n, v, gtot, itot)    # dual-exp COBA (rise + decay)
    L = s.buf.L
    slot = mod(n, L) + 1
    @inbounds due = s.buf.slots[i, slot]
    @inbounds s.buf.slots[i, slot] = zero(due)
    @inbounds gr = s.g_rise[i] + due                                  # deliver (both accumulators)
    @inbounds gd = s.g_decay[i] + due
    g = s.a * (gd - gr)                                               # conductance
    gtot += g                                                         # effective leak
    itot += g * s.Erev                                              # reversal drive
    @inbounds s.g_rise[i] = gr * s.decay_r                           # decay
    @inbounds s.g_decay[i] = gd * s.decay_d
    return (v, gtot, itot)
end
@inline function _syn_one(s::FrozenDualExpCOBAState, i, n, v, gtot, itot)   # dual-exp frozen-current (no shunt)
    L = s.buf.L
    slot = mod(n, L) + 1
    @inbounds due = s.buf.slots[i, slot]
    @inbounds s.buf.slots[i, slot] = zero(due)
    @inbounds gr = s.g_rise[i] + due                                  # deliver (both accumulators)
    @inbounds gd = s.g_decay[i] + due
    g = s.a * (gd - gr)                                               # conductance
    itot += g * (s.Erev - v)                                         # frozen current g·(Erev − V); gtot untouched
    @inbounds s.g_rise[i] = gr * s.decay_r                           # decay
    @inbounds s.g_decay[i] = gd * s.decay_d
    return (v, gtot, itot)
end

# External input current for neuron `i` (scalar constant, or a per-unit array).
@inline _inputval(input::Number, i) = input
@inline _inputval(input::AbstractArray, i) = @inbounds input[i]

# External Poisson drive kick for neuron `i` at step `n` (a strong-zero `false` when no drive,
# so it adds nothing and preserves `v`'s type). Matches the broadcast `_apply_drive!`.
@inline _drive_kick(::Nothing, n, i, dt) = false
@inline _drive_kick(d::PoissonDrive, n, i, dt) = d.weight * draw_poisson(d.rate * dt, d.seed, n, i)

# SDE noise increment for neuron `i` at step `n` (a strong-zero `false` when no noise, so it adds
# nothing and preserves `v`'s type --- the deterministic path stays bit-identical). Matches the
# broadcast `_apply_noise!`: the exact-OU scale times one counter-based normal.
@inline _noise_kick(::Nothing, n, i, dt, m) = false
@inline function _noise_kick(noise::WhiteNoise, n, i, dt, m)
    s = _noise_scale(noise, m, dt)
    return s * draw_normal(typeof(s), noise.seed, n, i)
end

# The per-neuron fused step body for neuron `i`: deliver + drive + accumulate + membrane + decay +
# threshold + reset + count. Written ONCE as a plain `@inline` function so it drives BOTH the GPU
# megakernel (`_fused_kernel!` below, one thread per neuron) AND the CPU tight loop (`_tight_step!`,
# a plain Julia `for`/`@threads` loop). Each neuron writes only its own state, so the loop is
# embarrassingly parallel → bit-identical regardless of thread count or driver.
@inline function _fused_unit!(V, refrac, spiked, spike_count, input, itotarr, gtotarr, syns, m, dt, n, drive, noise, aux, i)
    @inbounds begin
        v = V[i]
        r = refrac[i]
        z = zero(r)
        gtot = zero(eltype(V))
        itot = oftype(gtot, _inputval(input, i))
        # synaptic deliver + accumulate + decay (delta also kicks `v`), then external drive
        v, gtot, itot = _syn_contribute(syns, i, n, v, gtot, itot)
        # materialise the membrane accumulators so `Trace(:itot)`/`Trace(:gtot)` record real values under
        # the fused/GPU path too (bit-identical to what the Serial `_accum_all!` writes); one store each,
        # negligible against the membrane `exp`.
        itotarr[i] = itot
        gtotarr[i] = gtot
        v += _drive_kick(drive, n, i, dt)
        # resolve the per-neuron model (Phase 3): `_resolve(m, i) = m` for a scalar model (bit-identical),
        # the i-th override values for a Heterogeneous one.
        m_i = _resolve(m, i)
        # subthreshold (V, aux) advance + SDE noise (refractory clamps V to reset, no noise). For a
        # V-only model `aux` is `nothing` and this is exactly the prior membrane_step (bit-identical).
        w0 = _aux_read(aux, i)
        v_adv, w_adv = _advance_unit(m_i, v, w0, gtot, itot, dt)
        v = ifelse(r > z, reset_value(m_i), v_adv + _noise_kick(noise, n, i, dt, m_i))
        r = max(r - dt, z)
        # threshold (respecting refractory), then reset + arm refractory + spike-triggered adaptation
        s = (r ≤ z) & threshold(m_i, v)
        v = ifelse(s, reset_value(m_i), v)
        w_new = _spike_aux(m_i, w_adv, s)
        r = ifelse(s, refractory(m_i), r)
        V[i] = v
        refrac[i] = r
        _aux_write!(aux, w_new, i)
        spiked[i] = s
        spike_count[i] += s                                          # always-on count (firing_rate)
    end
    return nothing
end

# The GPU megakernel: one thread per neuron, `offset` shifts a group's launch into its flat range.
# (`@index(Global)` on its own line --- inlining the `+ offset` breaks KA's cartesian-context macro.)
@kernel function _fused_kernel!(V, refrac, spiked, spike_count, input, itotarr, gtotarr, syns, m, dt, n, drive, noise, aux, offset)
    g = @index(Global)
    i = g + offset
    _fused_unit!(V, refrac, spiked, spike_count, input, itotarr, gtotarr, syns, m, dt, n, drive, noise, aux, i)
end

# Launch the fused kernel (no synchronisation --- it pipelines with the scatter and monitors on
# the device stream; host syncs happen at the windowed flush and at host reads). A single (scalar)
# model is one launch over `1:N` (offset 0); a `MultiModel` launches the SAME kernel once per group
# over the group's range with the group's concrete model (so each launch is monomorphic). The aux
# column is selected per group --- `nothing` for a V-only group keeps its byte-identical fast path.
function _launch_fused!(integ::DewdropIntegrator)
    st = integ.state.state
    backend = get_backend(st.V)
    _launch_groups!(backend, integ.model, integ, st)
    return nothing
end

@inline _launch_groups!(backend, m::AbstractNeuronModel, integ, st) =
    _launch_group!(backend, m, integ, st, 0, length(st.V))
@inline _launch_groups!(backend, mm::MultiModel, integ, st) =
    _launch_grouped!(backend, mm.models, mm.ranges, integ, st)
@inline _launch_grouped!(backend, ::Tuple{}, ::Tuple{}, integ, st) = nothing
@inline function _launch_grouped!(backend, models::Tuple, ranges::Tuple, integ, st)
    r = first(ranges)
    _launch_group!(backend, first(models), integ, st, first(r) - 1, length(r))
    _launch_grouped!(backend, Base.tail(models), Base.tail(ranges), integ, st)
end
@inline function _launch_group!(backend, m, integ, st, offset::Int, len::Int)
    _fused_kernel!(backend)(
        st.V, st.refrac, integ.spiked, integ.spike_count, integ.input, integ.itot, integ.gtot,
        integ.syns, m, integ.dt, integ.n, integ.drive, integ.noise, _aux_col(st, m), offset;
        ndrange = len,
    )
    return nothing
end

# Scatter without the per-step host synchronisation (the launch-bound round-trip). Stream
# ordering keeps the next step's deliver correct; the host only reads at flush / solve-end.
@inline _propagate_nosync!(syn::AbstractSynapseState, integ) =
    (scatter!(syn.buf, syn.conn, integ.spiked, integ.n; sync = false); nothing)
@inline _propagate_all_nosync!(::Tuple{}, integ) = nothing
@inline _propagate_all_nosync!(s::Tuple, integ) =
    (_propagate_nosync!(first(s), integ); _propagate_all_nosync!(Base.tail(s), integ))

# One fused device step: dense megakernel → sparse scatter → monitors (the count is already in
# the megakernel, so `:record` here is just the monitors).
function _fused_step!(integ::DewdropIntegrator)
    _synprestep_all!(integ.syns, integ)         # streaming drives (host-side; the fused path has no global :deliver)
    _launch_fused!(integ)
    _propagate_step!(integ.compaction, integ)   # edge-parallel, or compacted if scatter = :compacted
    _record_all!(integ.monitors, integ)
    return nothing
end

# --- CPU tight step (backend = Fused): the SAME `_fused_unit!` body as the megakernel, driven by a
# plain Julia loop instead of KA --- it drops the per-workitem KA machinery (measured ~2× over the
# KA.CPU megakernel and the multi-pass broadcast). Threaded when >1 thread; each neuron writes only
# its own state, so the dense loop is bit-identical regardless of thread count (only the existing
# threaded atomic scatter is order-dependent, unchanged). One range per group (MultiModel).
function _tight_step!(integ::DewdropIntegrator)
    st = integ.state.state
    _synprestep_all!(integ.syns, integ)         # streaming drives (the fused tight loop has no global :deliver)
    _tight_groups!(integ.model, integ, st)
    _propagate_step!(integ.compaction, integ)
    _record_all!(integ.monitors, integ)
    return nothing
end
@inline _tight_groups!(m::AbstractNeuronModel, integ, st) = _tight_range!(m, integ, st, 1, length(st.V))
@inline _tight_groups!(mm::MultiModel, integ, st) = _tight_grouped!(mm.models, mm.ranges, integ, st)
@inline _tight_grouped!(::Tuple{}, ::Tuple{}, integ, st) = nothing
@inline function _tight_grouped!(models::Tuple, ranges::Tuple, integ, st)
    r = first(ranges)
    _tight_range!(first(models), integ, st, first(r), last(r))
    _tight_grouped!(Base.tail(models), Base.tail(ranges), integ, st)
end
# Thread the dense loop only with enough work per thread to amortise the per-step `@threads` dispatch
# (~tens of µs/step of task spawn+sync). Below this, the dispatch dominates and a small net runs FASTER
# single-threaded --- the small-N "floor" (e.g. ~0.16 s flat for N ≤ 4000 at 16 threads). Heuristic:
# ≥ `_TIGHT_MIN_PER_THREAD` neurons per thread. Bit-identical either way (the dense update is
# per-neuron-independent), so this only changes speed, never results.
const _TIGHT_MIN_PER_THREAD = 256

function _tight_range!(m, integ::DewdropIntegrator, st, lo::Int, hi::Int)
    V, refrac, spiked, spike_count = st.V, st.refrac, integ.spiked, integ.spike_count
    input, syns, dt, n = integ.input, integ.syns, integ.dt, integ.n
    itotarr, gtotarr = integ.itot, integ.gtot
    drive, noise, aux = integ.drive, integ.noise, _aux_col(st, m)
    if Threads.nthreads() > 1 && (hi - lo + 1) ≥ Threads.nthreads() * _TIGHT_MIN_PER_THREAD
        Threads.@threads for i in lo:hi
            _fused_unit!(V, refrac, spiked, spike_count, input, itotarr, gtotarr, syns, m, dt, n, drive, noise, aux, i)
        end
    else
        @inbounds for i in lo:hi
            _fused_unit!(V, refrac, spiked, spike_count, input, itotarr, gtotarr, syns, m, dt, n, drive, noise, aux, i)
        end
    end
    return nothing
end

# `backend = Turbo`: the deliver / accumulate / decay / scatter / monitor work stays scalar (it is not
# vectorisable --- ring buffers, the projection tuple, the sparse scatter), and ONLY the dense
# membrane + threshold + reset + count phase is replaced by a SIMD `@turbo` kernel chosen per model
# (`turbo_kernel`, registered by the LoopVectorization extension). So this orchestration lives in the
# core; the vectorised kernels live in the extension. `_check_backend` (init) guarantees a supported
# model + a loaded extension, so `turbo_kernel(...)` is non-`nothing` here.
function _turbo_step!(integ::DewdropIntegrator)
    run_phase!(Val(:deliver), integ)                       # ring → synaptic accumulators / V (delta), drive → V
    st = integ.state.state
    itot, gtot = integ.itot, integ.gtot
    itot .= integ.input                                    # base input (scalar or per-unit)
    fill!(gtot, zero(eltype(gtot)))
    _accum_all!(integ.syns, gtot, itot, st.V)              # per-neuron conductance + current
    turbo_kernel(typeof(integ.model))(integ)               # the SIMD dense membrane/threshold/reset/count kernel
    _decay_all!(integ.syns)                                # advance each projection's synaptic state
    _propagate_step!(integ.compaction, integ)              # sparse scatter (scalar / threaded as usual)
    _record_all!(integ.monitors, integ)
    return nothing
end

# Canonical schedule: route on the resolved execution backend × architecture. Any non-canonical
# schedule falls through to the generic `_run_step!` (broadcast phases).
_run_step!(::Schedule{DEFAULT_PHASES}, integ::DewdropIntegrator) =
    _step_backend!(integ.backend, get_backend(integ.state.state.V), integ)
@inline _step_backend!(::SimBackend, ::_KA.GPU, integ::DewdropIntegrator) = _fused_step!(integ)  # GPU = megakernel
@inline _step_backend!(::Serial, ::_KA.CPU, integ::DewdropIntegrator) = run_phases!(integ.schedule, integ)
@inline _step_backend!(::Fused, ::_KA.CPU, integ::DewdropIntegrator) = _tight_step!(integ)
@inline _step_backend!(::Turbo, ::_KA.CPU, integ::DewdropIntegrator) = _turbo_step!(integ)

# --- Periodic device synchronisation (safety valve for long runs) ---
# The fused/batched device steps pipeline on one stream with NO per-step host sync, so a long
# run with no monitor flush (which would otherwise drain the stream every window) can deep-queue
# kernels and pressure the driver's command buffer. Draining every `sync_every` steps bounds the
# queue at negligible cost (one round-trip per ~1000 steps). `sync_every = 0` disables it; on the
# CPU backend the broadcast step is synchronous, so there is nothing to drain (true no-op).
@inline _device_sync!(::_KA.CPU) = nothing
@inline _device_sync!(backend) = (applicable(synchronize, backend) && synchronize(backend); nothing)
@inline function _maybe_sync!(integ)
    se = integ.sync_every
    (se > 0 && integ.n % se == 0) || return nothing
    _device_sync!(get_backend(integ.state.state.V))
    return nothing
end
