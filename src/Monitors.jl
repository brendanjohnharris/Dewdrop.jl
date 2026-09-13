# * Monitor (recording) framework.
# A network is recorded by a NamedTuple of monitors, materialised from a `record = (...)` spec
# and held in the integrator. The `:record` phase unrolls them (tuple-recursion, dispatch-free,
# like the projection tuple). Each monitor stages into an arch-resident window buffer that
# flushes to a host store in windows: O(1) host transfers per window, the GPU-resident
# recording mechanism. Four axes: what (state/synaptic/accumulator var, spikes,
# or a Probe fn) × where (`:all` or an index subset) × how (per-unit, or a scalar Aggregate) ×
# when (stride). The `:record` slot runs after `:reset`, so traces are the post-reset state.

# Source descriptors: what to read (type-stable; the field/projection is a type parameter)
struct StateSrc{V} end                      # state.state.<V> (a model statevar column)
struct SynSrc{P, V} end                     # syns[P].acc.<V> (accumulator var of projection P)
struct AccumSrc{V} end                      # integ.<V>       (:gtot / :itot)
struct SpikeSrc end                         # integ.spiked
struct ProbeSrc{F}                          # f(integ) → a vector
    f::F
end
@inline _read(::StateSrc{V}, integ) where {V} = getproperty(integ.state.state, V)   # StructArray column
@inline _syn_acc(s) = s.acc                # plastic wrappers (PlasticState/BatchedSTDState) unwrap to their transmission state
@inline _read(::SynSrc{P, V}, integ) where {P, V} = getproperty(_syn_acc(getfield(integ.syns, P)), V)   # accumulator array under `.acc`
@inline _read(::AccumSrc{:gtot}, integ) = integ.gtot
@inline _read(::AccumSrc{:itot}, integ) = integ.itot
@inline _read(::SpikeSrc, integ) = integ.spiked
@inline _read(s::ProbeSrc, integ) = s.f(integ)

@inline _select(arr, ::Colon) = arr
@inline _select(arr, idx) = @view arr[idx]

# WindowBuffer: an arch-resident (n_out × Wcols) staging window flushed to a host
# (n_out × ncols) store every Wcols columns (the host store is the final result).
const _DEFAULT_WINDOW = 1024

mutable struct WindowBuffer{W <: AbstractMatrix, H <: AbstractMatrix}
    const window::W      # arch-resident staging (device on GPU, host on CPU)
    const store::H       # host-resident full result
    const Wcols::Int
    filled::Int          # columns staged in the current window
    flushed::Int         # columns already copied to the store
    host::Union{Nothing, H}   # reusable landing buffer for a device window; allocated on first flush
end
# Adapt only the staging window onto the target architecture; the `store` is the host-resident
# result and must stay a host `Array` (else the flush writes into a device store via scalar
# `setindex!`). This is the device-window/host-store split.
Adapt.adapt_structure(to, wb::WindowBuffer) =
    WindowBuffer(adapt(to, wb.window), wb.store, wb.Wcols, wb.filled, wb.flushed, wb.host)
function WindowBuffer(arch, ::Type{E}, n_out::Integer, ncols::Integer) where {E}
    Wcols = min(ncols, _DEFAULT_WINDOW)
    window = fill!(allocate(arch, E, Int(n_out), Wcols), zero(E))
    store = fill!(Array{E}(undef, Int(n_out), Int(ncols)), zero(E))
    return WindowBuffer(window, store, Wcols, 0, 0, nothing)
end

@inline _wcol(wb::WindowBuffer) = wb.filled + 1     # next free column in the window

# Flush the staged window columns to the host store (O(1) host transfers per window). The device
# window is moved host-side in bulk: a `copyto!` between a host view and a device view (or
# `Array` of a view) falls back to element-wise scalar indexing (forbidden on CUDA; broken under
# JLArrays' `allowscalar(false)`), so transfer the whole contiguous window with a single
# `copyto!(::Array, ::GPUArray)` and slice host-side.
@inline _windowslice(wb::WindowBuffer{<:Array}) = @inbounds view(wb.window, :, 1:wb.filled)
function _windowslice(wb::WindowBuffer)                     # device → host
    # one landing buffer per window, reused: allocating a window-sized host array on every flush is
    # steady GC pressure for a long run
    wb.host === nothing && (wb.host = similar(wb.store, size(wb.window)))
    copyto!(wb.host, wb.window)                             # one bulk DtoH of the full window
    return @inbounds view(wb.host, :, 1:wb.filled)
end
function flush!(wb::WindowBuffer)
    wb.filled == 0 && return wb
    @inbounds copyto!(
        view(wb.store, :, (wb.flushed + 1):(wb.flushed + wb.filled)),
        _windowslice(wb)
    )
    wb.flushed += wb.filled
    wb.filled = 0
    return wb
end
@inline function _advance!(wb::WindowBuffer)
    wb.filled += 1
    wb.filled == wb.Wcols && flush!(wb)
    return nothing
end

# Runtime monitors
# Per-unit: stage the selected source values verbatim (Trace / Spikes / Probe).
struct PerUnitMonitor{S, I, B <: WindowBuffer}
    src::S
    idx::I               # `:` (all) or an index vector
    buf::B
    every::Int
    bin::Int             # >1: each column reduces `bin` steps (see `record!`)
end
Adapt.@adapt_structure PerUnitMonitor
# Aggregate: reduce the selected source values to a scalar per step (R = :sum | :mean).
struct AggMonitor{S, I, B <: WindowBuffer, R, A}
    src::S
    idx::I
    buf::B
    every::Int
    n::Int               # number of selected units (for :mean)
    bin::Int
    acc::A               # 1-element arch-resident scratch: the device reduction's destination
end
# custom (the reducer `R` is a phantom type param @adapt_structure cannot reconstruct)
function Adapt.adapt_structure(to, m::AggMonitor{S, I, B, R}) where {S, I, B, R}
    src, idx, buf, acc = adapt(to, m.src), adapt(to, m.idx), adapt(to, m.buf), adapt(to, m.acc)
    return AggMonitor{typeof(src), typeof(idx), typeof(buf), R, typeof(acc)}(
        src, idx, buf, m.every, m.n, m.bin, acc
    )
end

# Does any monitor read `integ.itot` / `integ.gtot`? A pure function of the monitor NamedTuple's type,
# so it folds to a constant and lets the fused step skip materialising the accumulators when nothing
# records them (see `_accum_sinks`, Fused.jl).
@inline _reads_accum(::Type{<:PerUnitMonitor{<:AccumSrc}}) = true
@inline _reads_accum(::Type{<:AggMonitor{<:AccumSrc}}) = true
@inline _reads_accum(::Type{<:PerUnitMonitor{<:ProbeSrc}}) = true   # `f` is opaque: assume it reads them
@inline _reads_accum(::Type) = false
@inline _any_reads_accum(::Type{NT}) where {NT <: NamedTuple} = any(_reads_accum, fieldtypes(NT))
# the batched monitor types are declared later (Batch.jl) and add their own `_reads_accum` methods

@inline _due(m, integ) = (integ.n % m.every == 0) && _room(m)
@inline _room(m) = m.buf.flushed + m.buf.filled < size(m.buf.store, 2)

# Closing a bin: events sum (the column already holds the count), signals average (a box filter).
# Dispatched on the source, so a count buffer never meets a division.
@inline _closebin!(w, ::SpikeSrc, bin) = nothing
@inline _closebin!(w, src, bin) = (w ./= bin; nothing)

@inline function record!(m::PerUnitMonitor, integ)
    m.bin > 1 && return _record_binned!(m, integ)
    _due(m, integ) || return nothing
    @inbounds view(m.buf.window, :, _wcol(m.buf)) .= _select(_read(m.src, integ), m.idx)
    _advance!(m.buf)
    return nothing
end

# Binned: accumulate every step into the open column and close it on the bin's last step, so no
# sample is skipped. `integ.n` is 0-based, so a bin spans `n % bin ∈ 0:(bin-1)`.
@inline function _record_binned!(m::PerUnitMonitor, integ)
    _room(m) || return nothing
    r = integ.n % m.bin
    w = @inbounds view(m.buf.window, :, _wcol(m.buf))
    vals = _select(_read(m.src, integ), m.idx)
    r == 0 ? (w .= vals) : (w .+= vals)
    if r == m.bin - 1
        _closebin!(w, m.src, m.bin)
        _advance!(m.buf)
    end
    return nothing
end

@inline function record!(m::AggMonitor, integ)
    m.bin > 1 && return _record_binned!(m, integ)
    _due(m, integ) || return nothing
    _aggregate!(m.buf, _select(_read(m.src, integ), m.idx), m, _wcol(m.buf), false)
    _advance!(m.buf)
    return nothing
end

@inline function _record_binned!(m::AggMonitor, integ)
    _room(m) || return nothing
    r = integ.n % m.bin
    col = _wcol(m.buf)
    _aggregate!(m.buf, _select(_read(m.src, integ), m.idx), m, col, r != 0)
    if r == m.bin - 1
        _closebin!(@inbounds(view(m.buf.window, :, col)), m.src, m.bin)
        _advance!(m.buf)
    end
    return nothing
end

# scalar finalize for the supported reducers
@inline _finalize(::AggMonitor{S, I, B, :sum}, acc) where {S, I, B} = acc
@inline _finalize(m::AggMonitor{S, I, B, :mean}, acc) where {S, I, B} = acc / m.n

# CPU fast path: a plain serial reduction + direct slot write (no kernel launch, no host sync).
# `accum` adds into the open column instead of overwriting it: the temporal half of a binned
# aggregate; the spatial reduction is the same either way.
function _aggregate!(buf::WindowBuffer{<:Array}, vals, m::AggMonitor, col, accum::Bool = false)
    acc = zero(eltype(buf.window))
    @inbounds @simd for k in eachindex(vals)
        acc += vals[k]
    end
    v = _finalize(m, acc)
    @inbounds buf.window[1, col] = accum ? buf.window[1, col] + v : v
    return nothing
end

# Device path: `sum!` is the backend's own tree reduction, so the spatial fold is parallel rather than
# the O(n) walk a single thread would do (an `Aggregate` over `of = :all` reduces the whole population
# every step). It lands in the monitor's one-element scratch, and a second, one-element broadcast scales
# by `:mean` and folds into the open column. The scratch is what makes this safe: `sum!(slot; init =
# false)` would accumulate the destination once per block, so the reduction can never read the running
# slot directly. Both launches are queued on the stream, so there is still no per-step host round-trip;
# the slot is read only at the windowed flush (a DtoH copy that synchronises).
function _aggregate!(buf::WindowBuffer, vals, m::AggMonitor{S, I, B, R}, col, accum::Bool = false) where {S, I, B, R}
    T = eltype(buf.window)
    sum!(m.acc, vals)                                       # init = true: the only shape `sum!` gets right here
    slot = @inbounds view(buf.window, 1:1, col)
    # Divide rather than scale by the reciprocal: `x * inv(n)` differs from `x / n` in the last ulp for
    # about a third of all `x`, and the CPU `_finalize` above divides. It is one element, so the
    # division is free, and the two paths then differ only by the reduction order.
    if R === :mean
        d = T(m.n)
        accum ? (slot .+= m.acc ./ d) : (slot .= m.acc ./ d)
    else
        accum ? (slot .+= m.acc) : (slot .= m.acc)
    end
    return nothing
end

# Compile-time unroll over the monitors (dispatch-free + allocation-free, like the projection
# tuple). Recurse on the values tuple, not the NamedTuple: `Base.tail` on a NamedTuple of
# non-isbits monitors materialises intermediate NamedTuples (heap); on the backing tuple the
# compiler elides them when inlined.
@inline _record_all!(ms::NamedTuple, integ) = _rec_tuple!(values(ms), integ)
@inline _rec_tuple!(ms::Tuple, integ) = _foreach_tuple!(record!, ms, integ)

@inline _finalize_all!(ms::NamedTuple) = _fin_tuple!(values(ms))
@inline _fin_tuple!(ms::Tuple) = _foreach_tuple!(_finalize!, ms)
# Default finalize: flush the monitor's windowed store. Streaming reducers (no window) override to a no-op.
@inline _finalize!(m) = (flush!(m.buf); nothing)

# Specs (user-facing) → materialised at `init` into runtime monitors

# Time-axis reduction, mirroring the unit axis: `every =` selects steps as `of =` selects units,
# `bin =` reduces them as `Aggregate` does. They are alternatives, never combined. Striding an event
# record is refused outright: it thins spikes rather than coarse-graining them, and binning costs the
# same while keeping `sum(record) == sum(spike_count)`.
function _check_reduction(every::Int, bin::Int, isevent::Bool, what::AbstractString)
    (every ≥ 1 && bin ≥ 1) || throw(ArgumentError("$what: `every` and `bin` must be ≥ 1"))
    (every > 1 && bin > 1) && throw(ArgumentError(
        "$what: pass `every` (keep one step in k) OR `bin` (reduce each window of k steps), not both"))
    (isevent && every > 1) && throw(ArgumentError(
        "$what: `every = $every` would stride a spike record, keeping the spike bit at one step in " *
            "$every and discarding roughly $(round(100 * (1 - 1 / every); digits = 1))% of the spikes. " *
            "Use `bin = $every` instead: same memory, every spike counted."))
    return nothing
end
# The inner spec must be a per-unit source to reduce over. `Probe` carries `every`/`bin` too, so a
# stride check alone admitted it, and it then died on `inner.of` inside `_materialize`.
function _check_inner(inner, what)
    inner isa Union{Trace, Spikes} || throw(
        ArgumentError(
            "$what takes a `Trace` or `Spikes` as its inner spec; got $(nameof(typeof(inner))). " *
                "A `Probe` reduces for itself: return the scalar you want from its `f`."
        )
    )
    (inner.every == 1 && inner.bin == 1) || throw(
        ArgumentError(
            "$what: set `every`/`bin` on the $what, not on its inner spec (the inner one is ignored)"
        )
    )
    return nothing
end

"""
    Trace(var; of=:all, projection=nothing, every=1, bin=1)

Record the per-unit values of a state variable (`:V`, `:refrac`, …), a synaptic variable
(`:Isyn`/`:g` with `projection=i`), or an accumulator (`:gtot`/`:itot`), for the selected units.

`every = k` keeps one step in `k` (a stride: no filtering, so everything above the new Nyquist
folds into the band). `bin = k` instead averages each window of `k` steps, which is a box filter
and the better choice whenever the record will be spectrally analysed. The two are exclusive.
"""
struct Trace
    var::Symbol
    of::Any
    projection::Union{Nothing, Int}
    every::Int
    bin::Int
end
function Trace(var::Symbol; of = :all, projection = nothing, every::Integer = 1, bin::Integer = 1)
    _check_reduction(Int(every), Int(bin), false, "Trace(:$var)")
    return Trace(var, of, projection, Int(every), Int(bin))
end

"""
    Spikes(; of=:all, bin=1)

Record spikes for the selected units: a `Neuron × Time` boolean raster at `bin = 1`, or per-window
spike counts (`UInt16`) when `bin > 1`, each column summing `bin` steps.

`every > 1` is refused here; see `bin`. A `bin > 1` discards spike times within a window,
so [`raster`](@ref) and the statistics built on it require an unbinned recording.
"""
struct Spikes
    of::Any
    every::Int
    bin::Int
end
function Spikes(; of = :all, every::Integer = 1, bin::Integer = 1)
    _check_reduction(Int(every), Int(bin), true, "Spikes")
    return Spikes(of, Int(every), Int(bin))
end

"""
    Aggregate(inner, reducer; every=1, bin=1)

Reduce an inner [`Trace`](@ref)/[`Spikes`](@ref) over its selected units to one scalar per step.
`reducer` is `sum` (or `:sum`) or `:mean`: e.g. `Aggregate(Spikes(), sum)` is the population
spike count per step; `Aggregate(Trace(:V), :mean)` the population mean V. Arbitrary reductions go
through [`Probe`](@ref).

`bin = k` additionally reduces over time: spikes are summed over the window (so the column is the
population count in the bin, losing nothing), other sources are averaged. `every = k` strides
instead, and is refused over a spike source.
"""
struct Aggregate{S}
    inner::S
    reducer::Symbol
    every::Int
    bin::Int
end
function Aggregate(inner, reducer; every::Integer = 1, bin::Integer = 1)
    _check_reduction(Int(every), Int(bin), inner isa Spikes, "Aggregate")
    _check_inner(inner, "Aggregate")
    return Aggregate(inner, _reducer_sym(reducer), Int(every), Int(bin))
end
# Validate here: the reducer becomes a type parameter, and the two paths disagree on an unsupported
# one. The CPU `_finalize` has no method (a MethodError mid-solve); the GPU kernel tests only
# `R === :mean`, so anything else silently computes a sum.
_reducer_sym(s::Symbol) = s in (:sum, :mean) ? s : throw(
    ArgumentError(
        "Aggregate reducer must be :sum or :mean (or the function `sum`); got :$s. " *
            "Arbitrary reductions go through `Probe`."
    )
)
_reducer_sym(::typeof(sum)) = :sum
_reducer_sym(f) = throw(
    ArgumentError("Aggregate reducer must be :sum, :mean or the function `sum`; got $f. Use `Probe` for anything else.")
)

"""
    Probe(f; n, every=1, bin=1)

Record an arbitrary derived quantity: `f(integrator)` must return a length-`n` vector each step
(must be GPU-kernel-safe, i.e. broadcast/reduction, no scalar indexing, when run on a device).
`bin = k` averages each window of `k` steps; `every = k` strides.
"""
struct Probe{F}
    f::F
    n::Int
    every::Int
    bin::Int
end
function Probe(f; n::Integer, every::Integer = 1, bin::Integer = 1)
    _check_reduction(Int(every), Int(bin), false, "Probe")
    return Probe(f, Int(n), Int(every), Int(bin))
end

# selector + column count
_resolve_idx(arch, ::Colon) = Colon()
_resolve_idx(arch, of::Symbol) = of === :all ? Colon() : error("unknown selector $of (use :all or an index vector)")
_resolve_idx(arch, of::AbstractVector{<:Integer}) = on_architecture(arch, collect(Int, of))

# Resolve a monitor's `of` selector, additionally honouring named-subpopulation symbols against the
# registry (`of = :E`). `:all` and index vectors keep their meaning; a registered subpop resolves to
# an index vector over its range; an unknown symbol errors with the available names.
_resolve_of(arch, of::Colon, subpops) = Colon()
_resolve_of(arch, of::AbstractVector{<:Integer}, subpops) = _resolve_idx(arch, of)
function _resolve_of(arch, of::Symbol, subpops)
    of === :all && return Colon()                                     # full-N fast path (no index vector)
    (subpops !== nothing && haskey(subpops, of)) && return _resolve_idx(arch, subpops[of])
    avail = subpops === nothing ? ":all" : join(keys(subpops), ", ")
    error("unknown subpopulation/selector :$of (available: $avail, or an index vector)")
end
_nsel(N, ::Colon) = N
_nsel(N, idx) = length(idx)
# Columns of the store. A stride keeps `cld` (the first step is always recorded); a bin keeps only
# complete windows, so the trailing remainder is discarded, matching `coarsegrain`.
function _ncols(nsteps, every, bin = 1)
    bin > nsteps && throw(ArgumentError(
        "bin = $bin exceeds the run length ($nsteps steps): no complete window would close, so the record would be empty"))
    return bin > 1 ? fld(nsteps, bin) : cld(nsteps, every)
end

_srcof(t::Trace) = t.projection === nothing ? (t.var in (:gtot, :itot) ? AccumSrc{t.var}() : StateSrc{t.var}()) : SynSrc{t.projection, t.var}()

# materialise a spec → a runtime monitor (called per entry of the `record` NamedTuple at init).
# `subpops` is the network's subpopulation registry, so `of = :E` resolves against it.
function _materialize(spec::Trace, arch, ::Type{T}, N, nsteps, subpops) where {T}
    idx = _resolve_of(arch, spec.of, subpops)
    buf = WindowBuffer(arch, T, _nsel(N, idx), _ncols(nsteps, spec.every, spec.bin))
    return PerUnitMonitor(_srcof(spec), idx, buf, spec.every, spec.bin)
end
# A binned spike column holds a count, which `Bool` cannot: `UInt16` is one byte more per element
# than the raster it replaces and cannot overflow at any usable bin width (refractoriness bounds a
# neuron's count at bin/t_ref).
function _materialize(spec::Spikes, arch, ::Type{T}, N, nsteps, subpops) where {T}
    idx = _resolve_of(arch, spec.of, subpops)
    buf = WindowBuffer(arch, spec.bin > 1 ? UInt16 : Bool, _nsel(N, idx),
                       _ncols(nsteps, spec.every, spec.bin))
    return PerUnitMonitor(SpikeSrc(), idx, buf, spec.every, spec.bin)
end
function _materialize(spec::Aggregate, arch, ::Type{T}, N, nsteps, subpops) where {T}
    idx = _resolve_of(arch, spec.inner.of, subpops)
    buf = WindowBuffer(arch, T, 1, _ncols(nsteps, spec.every, spec.bin))
    src = spec.inner isa Spikes ? SpikeSrc() : _srcof(spec.inner)
    acc = fill!(allocate(arch, T, 1), zero(T))
    return AggMonitor{typeof(src), typeof(idx), typeof(buf), spec.reducer, typeof(acc)}(
        src, idx, buf, spec.every, _nsel(N, idx), spec.bin, acc)
end
function _materialize(spec::Probe, arch, ::Type{T}, N, nsteps, subpops) where {T}
    buf = WindowBuffer(arch, T, spec.n, _ncols(nsteps, spec.every, spec.bin))
    return PerUnitMonitor(ProbeSrc(spec.f), Colon(), buf, spec.every, spec.bin)
end

# build the monitor NamedTuple from the `record` spec NamedTuple
_make_monitors(::Nothing, arch, ::Type{T}, N, nsteps, subpops = nothing) where {T} = NamedTuple()
function _make_monitors(record::NamedTuple, arch, ::Type{T}, N, nsteps, subpops = nothing) where {T}
    return map(spec -> _materialize(spec, arch, T, N, nsteps, subpops), record)
end

# the recorded host result + metadata for a monitor (consumed by raster/firing_rate + the ext)
struct RecordResult{D}
    data::D              # host (n_out × ncols)
    idx::Any             # selected neuron indices (`Colon`/vector), or `nothing` for aggregates/probes
    every::Int           # steps per column (the bin width when binned), so `every * dt` is the sample period
    kind::Symbol         # :trace | :spikes | :aggregate | :probe
    bin::Int             # 1 = one step per column; >1 = each column reduces `bin` steps
    var::Symbol          # the recorded variable (:V, :g, :gtot, …), so a result can be looked up by it
end
RecordResult(data, idx, every, kind) = RecordResult(data, idx, every, kind, 1, kind)
RecordResult(data, idx, every, kind, bin) = RecordResult(data, idx, every, kind, bin, kind)

# The recorded variable, read off the source descriptor's type parameter.
@inline _srcvar(::StateSrc{V}) where {V} = V
@inline _srcvar(::SynSrc{P, V}) where {P, V} = V
@inline _srcvar(::AccumSrc{V}) where {V} = V
@inline _srcvar(::SpikeSrc) = :spikes
@inline _srcvar(::ProbeSrc) = :probe
# Time of recorded column `j`. Recording runs at the end of the step, so a strided column holds the
# state after step `(j-1)*every`, at `t0 + ((j-1)*every + 1)*dt`; a binned column reduces steps
# `(j-1)*bin` through `j*bin-1` and is stamped at its right edge, `t0 + j*bin*dt`. Both reduce to
# `t0 + j*dt` unstrided. Every time axis (raster, the TimeseriesBase and Makie extensions) uses this.
@inline _coltime(res::RecordResult, j::Integer, dt, t0) =
    res.bin > 1 ? t0 + j * res.bin * dt : t0 + ((j - 1) * res.every + 1) * dt
# A range, not a vector: both formulas are affine in `j` with step `every*dt`, and a regular range is
# what TimeseriesBase needs to give the time axis a `step`.
_coltimes(res::RecordResult, dt, t0) =
    range(_coltime(res, 1, dt, t0); step = res.every * dt, length = size(res.data, 2))

_result(m::PerUnitMonitor{<:SpikeSrc}) = RecordResult(m.buf.store, m.idx, max(m.every, m.bin), :spikes, m.bin, :spikes)
_result(m::PerUnitMonitor{<:ProbeSrc}) = RecordResult(m.buf.store, nothing, max(m.every, m.bin), :probe, m.bin, :probe)
_result(m::PerUnitMonitor) = RecordResult(m.buf.store, m.idx, max(m.every, m.bin), :trace, m.bin, _srcvar(m.src))
_result(m::AggMonitor) = RecordResult(m.buf.store, nothing, max(m.every, m.bin), :aggregate, m.bin, _srcvar(m.src))

# Spike times do not survive binning, so every statistic that reconstructs them refuses a binned
# record rather than quantising ISIs onto the bin grid.
function _require_unbinned(res::RecordResult, what::AbstractString)
    res.bin > 1 && error(
        "$what needs unbinned spikes, but this monitor used `bin = $(res.bin)`: each column is a " *
            "per-window COUNT, so individual spike times no longer exist. Re-record with `bin = 1`, " *
            "or work with the counts directly."
    )
    return res
end

export Trace, Spikes, Aggregate, Probe
