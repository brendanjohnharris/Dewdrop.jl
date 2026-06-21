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

# External input current for neuron `i` (scalar constant, or a per-unit array).
@inline _inputval(input::Number, i) = input
@inline _inputval(input::AbstractArray, i) = @inbounds input[i]

# External Poisson drive kick for neuron `i` at step `n` (a strong-zero `false` when no drive,
# so it adds nothing and preserves `v`'s type). Matches the broadcast `_apply_drive!`.
@inline _drive_kick(::Nothing, n, i, dt) = false
@inline _drive_kick(d::PoissonDrive, n, i, dt) = d.weight * draw_poisson(d.rate * dt, d.seed, n, i)

# The fused dense step: deliver + drive + accumulate + membrane + decay + threshold + reset +
# count, for every neuron, in one launch. `syns` is the projection-state tuple; `m` the (isbits)
# neuron model; `drive` a PoissonDrive or nothing.
@kernel function _fused_kernel!(V, refrac, spiked, spike_count, input, syns, m, dt, n, drive)
    i = @index(Global)
    @inbounds begin
        v = V[i]
        r = refrac[i]
        z = zero(r)
        gtot = zero(eltype(V))
        itot = oftype(gtot, _inputval(input, i))
        # synaptic deliver + accumulate + decay (delta also kicks `v`), then external drive
        v, gtot, itot = _syn_contribute(syns, i, n, v, gtot, itot)
        v += _drive_kick(drive, n, i, dt)
        # subthreshold membrane step (refractory units clamp to the reset value)
        v = ifelse(r > z, reset_value(m), membrane_step(m, v, gtot, itot, dt))
        r = max(r - dt, z)
        # threshold (respecting refractory), then reset + arm refractory
        s = (r ≤ z) & threshold(m, v)
        v = ifelse(s, reset_value(m), v)
        r = ifelse(s, refractory(m), r)
        V[i] = v
        refrac[i] = r
        spiked[i] = s
        spike_count[i] += s                                          # always-on count (firing_rate)
    end
end

# Launch the fused kernel (no synchronisation --- it pipelines with the scatter and monitors on
# the device stream; host syncs happen at the windowed flush and at host reads).
function _launch_fused!(integ::DewdropIntegrator)
    st = integ.state.state
    V = st.V
    backend = get_backend(V)
    _fused_kernel!(backend)(
        V, st.refrac, integ.spiked, integ.spike_count, integ.input,
        integ.syns, integ.model, integ.dt, integ.n, integ.drive;
        ndrange = length(V),
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
    _launch_fused!(integ)
    _propagate_step!(integ.compaction, integ)   # edge-parallel, or compacted if scatter = :compacted
    _record_all!(integ.monitors, integ)
    return nothing
end

# Canonical schedule: pick the fused device path or the broadcast CPU path by backend. Any
# non-canonical schedule falls through to the generic `_run_step!` (broadcast phases).
_run_step!(::Schedule{DEFAULT_PHASES}, integ::DewdropIntegrator) =
    _step_on_backend!(get_backend(integ.state.state.V), integ)
@inline _step_on_backend!(::_KA.CPU, integ::DewdropIntegrator) = run_phases!(integ.schedule, integ)
@inline _step_on_backend!(::_KA.GPU, integ::DewdropIntegrator) = _fused_step!(integ)

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
