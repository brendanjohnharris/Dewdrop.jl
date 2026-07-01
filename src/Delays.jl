# * Per-synapse conduction delays: a NEST-style ring buffer of postsynaptic
# accumulators. A spike scattered at step `now` along a synapse with integer delay `d` is
# deposited into ring slot (now + d) mod L for its postsynaptic target, and delivered when
# the clock reaches that step. Because the delay is read per synapse, arbitrarily distinct
# delays cost the same O(1) delivery as a single global delay (a homogeneous delay is the
# degenerate case). Sized L = maxdelay + 1.
#
# Layout is (N_post, L): a fixed column per time slot, so collecting the due increments is a
# contiguous column read (coalescing-friendly on GPU). Deposit is a scatter-add into a
# column: a host loop here, a single Atomix.@atomic kernel on the device.

"""
    DelayBuffer(arch, T, N, maxdelay)

A ring buffer of postsynaptic increment accumulators of element type `T` for `N`
postsynaptic units, holding delays up to `maxdelay` integer steps (`L = maxdelay + 1`
slots). See [`deposit!`](@ref) and [`collect_due!`](@ref).
"""
struct DelayBuffer{M <: AbstractMatrix}
    slots::M     # (N_post, L): slots[i, s] = pending increment for neuron i at ring slot s
    L::Int
end
Adapt.@adapt_structure DelayBuffer

function DelayBuffer(arch::AbstractArchitecture, ::Type{T}, N::Integer, maxdelay::Integer) where {T}
    L = Int(maxdelay) + 1
    slots = fill!(allocate(arch, T, Int(N), L), zero(T))
    return DelayBuffer(slots, L)
end

"""
    maxdelay(buf)

The largest delay (in integer steps) the buffer can hold.
"""
maxdelay(buf::DelayBuffer) = buf.L - 1

@inline _slotof(t::Integer, L::Integer) = mod(t, L) + 1   # 1-based ring slot for step `t`

"""
    deposit!(buf, now, target, value, delay)

Add `value` to postsynaptic neuron `target`, to be delivered `delay` integer steps after
the current step `now` (`delay` must be ≤ `maxdelay(buf)`). The fractional-offset hook for
sub-`dt` spike timing rides on the same slot indexing; the engine always uses integer delays.
"""
@inline function deposit!(buf::DelayBuffer, now::Integer, target::Integer, value, delay::Integer)
    @inbounds buf.slots[target, _slotof(now + delay, buf.L)] += value
    return nothing
end

"""
    collect_due!(buf, now)

Return the vector of increments due at step `now`, and clear that slot so it is reusable
for step `now + L`.
"""
function collect_due!(buf::DelayBuffer, now::Integer)
    s = _slotof(now, buf.L)
    @inbounds due = buf.slots[:, s]
    @inbounds buf.slots[:, s] .= zero(eltype(buf.slots))
    return due
end

"""
    deliver_due!(target, buf, now)

Add the increments due at step `now` into `target` in place, then clear that slot. The
allocation-free form of [`collect_due!`](@ref) for the engine's deliver phase.
"""
function deliver_due!(target, buf::DelayBuffer, now::Integer)
    s = _slotof(now, buf.L)
    @inbounds col = @view buf.slots[:, s]
    target .+= col
    col .= zero(eltype(buf.slots))
    return nothing
end

# CPU fast path: add-and-clear in a single pass over the due column (the generic form reads the
# column twice). Dispatched on plain `Array` storage; the device path keeps the broadcasts.
function deliver_due!(target::AbstractVector, buf::DelayBuffer{<:Array}, now::Integer)
    s = _slotof(now, buf.L)
    slots = buf.slots
    z = zero(eltype(slots))
    @inbounds for i in eachindex(target)
        target[i] += slots[i, s]
        slots[i, s] = z
    end
    return nothing
end

"""
    deliver_due_dual!(a, b, buf, now)

Add the increments due at step `now` into BOTH `a` and `b` in place, then clear that slot; the
deliver for a dual-state synapse (e.g. the dual-exponential's `g_rise`/`g_decay`, which receive the
same kick). Reads the due column once.
"""
function deliver_due_dual!(a, b, buf::DelayBuffer, now::Integer)
    s = _slotof(now, buf.L)
    @inbounds col = @view buf.slots[:, s]
    a .+= col
    b .+= col
    col .= zero(eltype(buf.slots))
    return nothing
end
function deliver_due_dual!(a::AbstractVector, b::AbstractVector, buf::DelayBuffer{<:Array}, now::Integer)
    s = _slotof(now, buf.L)
    slots = buf.slots
    z = zero(eltype(slots))
    @inbounds for i in eachindex(a)
        d = slots[i, s]
        a[i] += d
        b[i] += d
        slots[i, s] = z
    end
    return nothing
end
