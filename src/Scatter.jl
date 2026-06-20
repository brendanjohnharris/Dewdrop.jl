# * Event-driven sparse scatter (M1c) --- the spike-propagation hot path, written ONCE as a
# KernelAbstractions kernel so the same source runs on CPU (`@threads`) and GPU (PTX/AIR).
# One thread per presynaptic neuron; spiking neurons walk their CSR row and deposit each
# synapse's weight into the delay ring buffer at (now + per-synapse delay), accumulating with
# `Atomix.@atomic` (several presynaptic spikes may hit the same target+slot in a step).
# Order-independent atomic accumulation also keeps the operator deterministic. Only spiking
# neurons do work, so cost scales with spikes, not with the synapse count (the BrainPy lesson).

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
import Atomix

@kernel function _scatter_kernel!(
        slots, @Const(spiked), @Const(rowptr), @Const(post), @Const(weight), @Const(delay), now, L
    )
    pre = @index(Global)
    @inbounds if spiked[pre]
        for e in rowptr[pre]:(rowptr[pre + 1] - 1)
            slot = mod(now + delay[e], L) + 1
            Atomix.@atomic slots[post[e], slot] += weight[e]
        end
    end
end

"""
    scatter!(buf, conn, spiked, now)

Scatter the spikes in `spiked` (a per-presynaptic-neuron mask) through connectivity `conn`
into the delay ring buffer `buf` at the current step `now`. Each spiking neuron's synapses
deposit their weight into the postsynaptic target's slot `(now + delay) mod L`. Runs on
whatever backend owns `buf.slots` (CPU or device) via `get_backend`.
"""
function scatter!(buf::DelayBuffer, conn::SparseCSR, spiked, now::Integer)
    backend = get_backend(buf.slots)
    _scatter_kernel!(backend)(
        buf.slots, spiked, conn.rowptr, conn.post, conn.weight, conn.delay, Int(now), buf.L;
        ndrange = length(spiked),
    )
    # Some backends (e.g. the JLArrays reference backend) run kernels synchronously and
    # define no `synchronize`; only wait where the backend actually needs it (CPU, GPU).
    applicable(synchronize, backend) && synchronize(backend)
    return nothing
end
