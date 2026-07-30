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

# * Fixed-point accumulation --------------------------------------------------------------------
# The ring accumulates FIXED-POINT COUNTS, not floats. Deposits land through `Atomix.@atomic`, and
# several presynaptic spikes may hit the same (target, slot) in one step, so the order in which they
# are summed varies with thread count, backend and scatter strategy. Float addition is not
# associative, so a float ring gives a different last bit run to run; INTEGER addition is, so an
# integer ring is bit-identical no matter how the deposits are interleaved.
#
# That matters far more than the last bit suggests: a spiking network is chaotic, so a one-ulp
# difference sits dormant until it flips which side of threshold a neuron lands on, and the network
# then decorrelates completely within a step or two. Measured on the spatial E/I model, the flip came
# at 3.4 s of a 25 s run --- i.e. every run was an independent realisation.
#
# The weights are quantised in the kernel (`w * scale`, one multiply) rather than stored twice. The
# weights are already `Float32`, so quantising at `Float32` relative precision is no worse than how
# they are held today, while the SUM becomes exact --- strictly better than float accumulation, which
# rounds at every add.
const FPCount = Int64

# Headroom between the worst case a slot can hold and `typemax(FPCount)`.
const _FP_SAFETY = 1024

# Fallback scale when the ring is built without a connectome to size it from (`DelayBuffer(arch, T,
# N, maxdelay)`): 2^30 counts per unit weight, i.e. ~1e-9 resolution and ~8.6e9 of range.
const _FP_DEFAULT_SCALE = 2.0^30

# Fixed point applies to plain IEEE floats. Anything else --- a `Dual`/`Active` from the
# differentiable backend --- keeps a VALUE ring: rounding is not differentiable, and those paths are
# CPU-serial, so they are already order-deterministic and gain nothing from quantisation.
_ring_eltype(::Type{<:Base.IEEEFloat}) = FPCount
_ring_eltype(::Type{T}) where {T} = T
_ring_scale(::Type{<:Base.IEEEFloat}, scale) = scale
_ring_scale(::Type{T}, scale) where {T} = one(T)

@inline _fp_quantise(::Type{I}, w, scale) where {I <: Integer} = round(I, w * scale)
@inline _fp_quantise(::Type, w, scale) = w                       # value ring: store as-is
@inline _fp_value(c::Integer, scale) = c * inv(scale)
@inline _fp_value(c, scale) = c                                  # value ring: already physical

"""
    fixedpoint_scale(post, weight, N; safety = $(_FP_SAFETY))

Counts per unit weight for a ring fed by the edges `(post, weight)` over `N` targets: the largest
power of two that keeps the worst case a single slot can hold --- every incoming synapse of one
target depositing in the same step --- inside `$(FPCount)` with `safety` to spare. A power of two
makes `w * scale` exact in binary, so the only quantisation is the weight's own mantissa round.
"""
function fixedpoint_scale(post, weight, N::Integer; safety = _FP_SAFETY)
    # A value ring (`Dual`/`Active` weights, the differentiable backend) has no scale to size, and
    # those weights do not convert to `Float64` anyway. The eltype is known at compile time, so this
    # costs nothing on the ordinary path.
    eltype(weight) <: Base.IEEEFloat || return _FP_DEFAULT_SCALE
    isempty(weight) && return _FP_DEFAULT_SCALE
    w, p = Array(weight), Array(post)
    isempty(p) && return _FP_DEFAULT_SCALE
    # Sized from the targets actually named by `post`, not from `N`: a projection's ring is built for
    # its own postsynaptic population, and `post` is not guaranteed to index into `1:N`.
    tot = zeros(Float64, max(Int(N), Int(maximum(p))))
    for e in eachindex(w)
        tot[p[e]] += abs(Float64(w[e]))
    end
    peak = maximum(tot)
    peak > 0 || return _FP_DEFAULT_SCALE
    # Capped at 2^62 so the scale stays finite in `Float32` (a vanishingly small `peak` would
    # otherwise ask for a scale past `floatmax(Float32)`); the cap only costs resolution nobody needs.
    scale = min(2.0^62, 2.0^floor(log2(typemax(FPCount) / (safety * peak))))
    # A weight that quantises to zero would be silently dropped. That needs a dynamic range of ~10^18
    # within one projection to happen, but warn rather than lose spikes in silence.
    wmin = minimum(x -> (x == 0 ? Inf : abs(Float64(x))), w)
    if isfinite(wmin) && wmin * scale < 1
        @warn "fixed-point delay ring: smallest weight quantises to zero; increase the weight scale " *
            "or split the projection" wmin peak scale
    end
    return scale
end

"""
    DelayBuffer(arch, T, N, maxdelay; scale)

A ring buffer of postsynaptic increment accumulators for `N` postsynaptic units, holding delays up to
`maxdelay` integer steps (`L = maxdelay + 1` slots). Increments accumulate as `$(FPCount)` fixed-point
counts at `scale` counts per unit weight so that atomic accumulation is order-independent and hence
bit-reproducible (see the note above); `T` is the float type increments are delivered back in. Size
`scale` from the connectome with [`fixedpoint_scale`](@ref). See [`deposit!`](@ref),
[`collect_due!`](@ref) and [`slotvalues`](@ref).
"""
struct DelayBuffer{M <: AbstractMatrix, T}
    slots::M     # (N_post, L): slots[i, s] = pending increment for neuron i at ring slot s, in counts
    L::Int
    scale::T     # counts per unit weight; a physical increment is `counts / scale`
end
Adapt.@adapt_structure DelayBuffer

function DelayBuffer(
        arch::AbstractArchitecture, ::Type{T}, N::Integer, maxdelay::Integer;
        scale::Real = _FP_DEFAULT_SCALE
    ) where {T}
    L = Int(maxdelay) + 1
    E = _ring_eltype(T)
    slots = fill!(allocate(arch, E, Int(N), L), zero(E))
    return DelayBuffer(slots, L, _ring_scale(T, T(scale)))
end

"""
    slotvalues(buf)

The pending increments in PHYSICAL units, as a plain array --- the ring itself holds fixed-point
counts, so read it through this rather than touching `buf.slots` directly.
"""
slotvalues(buf::DelayBuffer) = Array(buf.slots) .* inv(buf.scale)

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
    @inbounds buf.slots[target, _slotof(now + delay, buf.L)] += _fp_quantise(eltype(buf.slots), value, buf.scale)
    return nothing
end

"""
    collect_due!(buf, now)

Return the vector of increments due at step `now`, and clear that slot so it is reusable
for step `now + L`.
"""
function collect_due!(buf::DelayBuffer, now::Integer)
    s = _slotof(now, buf.L)
    @inbounds due = _fp_value.(buf.slots[:, s], buf.scale)
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
    inv_scale = inv(buf.scale)
    @inbounds col = @view buf.slots[:, s]
    target .+= col .* inv_scale
    col .= zero(eltype(buf.slots))
    return nothing
end

# CPU fast path: add-and-clear in a single pass over the due column (the generic form reads the
# column twice). Dispatched on plain `Array` storage; the device path keeps the broadcasts.
function deliver_due!(target::AbstractVector, buf::DelayBuffer{<:Array}, now::Integer)
    s = _slotof(now, buf.L)
    slots = buf.slots
    inv_scale = inv(buf.scale)
    @inbounds for i in eachindex(target)
        target[i] += slots[i, s] * inv_scale
        slots[i, s] = zero(eltype(slots))
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
    inv_scale = inv(buf.scale)
    @inbounds col = @view buf.slots[:, s]
    a .+= col .* inv_scale
    b .+= col .* inv_scale
    col .= zero(eltype(buf.slots))
    return nothing
end
function deliver_due_dual!(a::AbstractVector, b::AbstractVector, buf::DelayBuffer{<:Array}, now::Integer)
    s = _slotof(now, buf.L)
    slots = buf.slots
    inv_scale = inv(buf.scale)
    @inbounds for i in eachindex(a)
        d = slots[i, s] * inv_scale
        a[i] += d
        b[i] += d
        slots[i, s] = zero(eltype(slots))
    end
    return nothing
end
