# * Event-driven sparse scatter (M1c) --- the spike-propagation hot path, written ONCE as a
# KernelAbstractions kernel so the same source runs on CPU (`@threads`) and GPU (PTX/AIR).
# One thread per presynaptic neuron; spiking neurons walk their CSR row and deposit each
# synapse's weight into the delay ring buffer at (now + per-synapse delay), accumulating with
# `Atomix.@atomic` (several presynaptic spikes may hit the same target+slot in a step).
# Order-independent atomic accumulation also keeps the operator deterministic. Only spiking
# neurons do work, so cost scales with spikes, not with the synapse count (the BrainPy lesson).

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize
import Atomix

# EDGE-PARALLEL scatter (the GPU occupancy fix): ONE thread per synapse, not per presynaptic
# neuron. The old per-neuron kernel exposed only (spiking-neuron) parallelism and walked each row
# serially (a few hundred busy threads, the rest idle --- the profiled 97-99% bottleneck). Here
# every synapse is an independent thread with uniform work (one conditional atomic), so the device
# is saturated. The presynaptic source is read from the materialised `src` array (sorted, so the
# idle-thread reads stay coalesced) --- measured 1.7-4x over per-neuron, and far better than a
# per-edge binary search of rowptr (whose O(log npre) work dominates at large nedges).
@kernel function _scatter_edge_kernel!(
        slots, @Const(spiked), @Const(src), @Const(post), @Const(weight), @Const(delay), now, L
    )
    e = @index(Global)
    @inbounds begin
        pre = src[e]
        if spiked[pre]
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
whatever backend owns `buf.slots` (CPU or device) via `get_backend`; the device path is
edge-parallel (one thread per synapse), the CPU path a serial per-neuron walk.
"""
function scatter!(buf::DelayBuffer, conn::SparseCSR, spiked, now::Integer; sync::Bool = true)
    backend = get_backend(buf.slots)
    ne = nedges(conn)
    if ne > 0
        _scatter_edge_kernel!(backend)(
            buf.slots, spiked, conn.src, conn.post, conn.weight, conn.delay, Int(now), buf.L;
            ndrange = ne,
        )
    end
    # Some backends (e.g. the JLArrays reference backend) run kernels synchronously and
    # define no `synchronize`; only wait where the backend actually needs it (CPU, GPU). The
    # fused device step passes `sync = false` so steps pipeline on one stream (M6 Tier-1); the
    # next deliver/read on the same stream still sees this scatter's writes.
    sync && applicable(synchronize, backend) && synchronize(backend)
    return nothing
end

# CPU fast path. The per-step KernelAbstractions launch + `synchronize` round-trip dominates
# the small-N regime (the launch-bound bottleneck); a plain serial walk over the spiking
# neurons' CSR rows removes it entirely. Serial accumulation visits each (target, slot) in a
# fixed presynaptic order, so it is deterministic and needs no atomics. Dispatched on plain
# `Array` storage; the device path keeps the kernel above. Result matches the kernel for any
# input (same set of additions).
function scatter!(
        buf::DelayBuffer{<:Array},
        conn::SparseCSR{<:Array, <:Array, <:Array, <:Array},
        spiked::AbstractArray, now::Integer; sync::Bool = true,   # `sync` ignored: a serial CPU walk
    )
    slots, L = buf.slots, buf.L
    rowptr, post, weight, delay = conn.rowptr, conn.post, conn.weight, conn.delay
    n = Int(now)
    if Threads.nthreads() == 1
        # serial: contributions to a slot accumulate in a fixed presynaptic order, so the result
        # is bit-reproducible (no atomics needed).
        @inbounds for pre in eachindex(spiked)
            spiked[pre] || continue
            for e in rowptr[pre]:(rowptr[pre + 1] - 1)
                slots[post[e], mod(n + delay[e], L) + 1] += weight[e]
            end
        end
    else
        # partitioned over presynaptic neurons across threads; disjoint rows, but two threads may
        # hit the same (target, slot), so accumulate atomically. Order-independent and correct,
        # though (like the GPU path) not bit-identical across thread counts --- run single-threaded
        # for bit-reproducibility, multi-threaded for speed (statistically identical).
        @inbounds Threads.@threads for pre in eachindex(spiked)
            spiked[pre] || continue
            for e in rowptr[pre]:(rowptr[pre + 1] - 1)
                Atomix.@atomic slots[post[e], mod(n + delay[e], L) + 1] += weight[e]
            end
        end
    end
    return nothing
end
