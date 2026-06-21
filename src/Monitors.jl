# * Monitor (recording) framework (M4).
# A network is recorded by a NamedTuple of monitors, materialised from a `record = (...)` spec
# and held in the integrator. The `:record` phase unrolls them (tuple-recursion, dispatch-free,
# like the projection tuple). Each monitor stages into an arch-resident WINDOW buffer that
# flushes to a host store in windows --- O(1) host transfers per window, the GPU-resident
# recording substrate (M0 contract 8). Four axes: WHAT (state/synaptic/accumulator var, spikes,
# or a Probe fn) × WHERE (`:all` or an index subset) × HOW (per-unit, or a scalar Aggregate) ×
# WHEN (stride). The `:record` slot runs after `:reset`, so traces are the post-reset state.

using KernelAbstractions: @kernel, @index, @Const, get_backend, synchronize

# --- Source descriptors: WHAT to read (type-stable; the field/projection is a type parameter) ---
struct StateSrc{V} end                      # state.state.<V> (a model statevar column)
struct SynSrc{P, V} end                     # syns[P].<V>     (synaptic state of projection P)
struct AccumSrc{V} end                      # integ.<V>       (:gtot / :itot)
struct SpikeSrc end                         # integ.spiked
struct ProbeSrc{F}                          # f(integ) → a vector
    f::F
end
@inline _read(::StateSrc{V}, integ) where {V} = getproperty(integ.state.state, V)   # StructArray column
@inline _read(::SynSrc{P, V}, integ) where {P, V} = getfield(getfield(integ.syns, P), V)
@inline _read(::AccumSrc{:gtot}, integ) = integ.gtot
@inline _read(::AccumSrc{:itot}, integ) = integ.itot
@inline _read(::SpikeSrc, integ) = integ.spiked
@inline _read(s::ProbeSrc, integ) = s.f(integ)

@inline _select(arr, ::Colon) = arr
@inline _select(arr, idx) = @view arr[idx]

# --- WindowBuffer: an arch-resident (n_out × Wcols) staging window flushed to a host
# (n_out × ncols) store every Wcols columns (the host store is the final result). ---
const _DEFAULT_WINDOW = 1024

mutable struct WindowBuffer{W <: AbstractMatrix, H <: AbstractMatrix}
    const window::W      # arch-resident staging (device on GPU, host on CPU)
    const store::H       # host-resident full result
    const Wcols::Int
    filled::Int          # columns staged in the current window
    flushed::Int         # columns already copied to the store
end
Adapt.@adapt_structure WindowBuffer
function WindowBuffer(arch, ::Type{E}, n_out::Integer, ncols::Integer) where {E}
    Wcols = min(ncols, _DEFAULT_WINDOW)
    window = fill!(allocate(arch, E, Int(n_out), Wcols), zero(E))
    store = fill!(Array{E}(undef, Int(n_out), Int(ncols)), zero(E))
    return WindowBuffer(window, store, Wcols, 0, 0)
end

@inline _wcol(wb::WindowBuffer) = wb.filled + 1     # next free column in the window

# Flush the staged window columns to the host store (O(1) host transfers per window).
function flush!(wb::WindowBuffer)
    wb.filled == 0 && return wb
    @inbounds copyto!(view(wb.store, :, (wb.flushed + 1):(wb.flushed + wb.filled)),
        view(wb.window, :, 1:wb.filled))
    wb.flushed += wb.filled
    wb.filled = 0
    return wb
end
@inline function _advance!(wb::WindowBuffer)
    wb.filled += 1
    wb.filled == wb.Wcols && flush!(wb)
    return nothing
end

# --- Runtime monitors ---
# Per-unit: stage the selected source values verbatim (Trace / Spikes / Probe).
struct PerUnitMonitor{S, I, B <: WindowBuffer}
    src::S
    idx::I               # `:` (all) or an index vector
    buf::B
    every::Int
end
Adapt.@adapt_structure PerUnitMonitor
# Aggregate: reduce the selected source values to a scalar per step (R = :sum | :mean).
struct AggMonitor{S, I, B <: WindowBuffer, R}
    src::S
    idx::I
    buf::B
    every::Int
    n::Int               # number of selected units (for :mean)
end
# custom (the reducer `R` is a phantom type param @adapt_structure cannot reconstruct)
function Adapt.adapt_structure(to, m::AggMonitor{S, I, B, R}) where {S, I, B, R}
    src, idx, buf = adapt(to, m.src), adapt(to, m.idx), adapt(to, m.buf)
    return AggMonitor{typeof(src), typeof(idx), typeof(buf), R}(src, idx, buf, m.every, m.n)
end

@inline _due(m, integ) = (integ.n % m.every == 0) && (m.buf.flushed + m.buf.filled < size(m.buf.store, 2))

@inline function record!(m::PerUnitMonitor, integ)
    _due(m, integ) || return nothing
    @inbounds view(m.buf.window, :, _wcol(m.buf)) .= _select(_read(m.src, integ), m.idx)
    _advance!(m.buf)
    return nothing
end

@inline function record!(m::AggMonitor, integ)
    _due(m, integ) || return nothing
    _aggregate!(m.buf, _select(_read(m.src, integ), m.idx), m, _wcol(m.buf))
    _advance!(m.buf)
    return nothing
end

# scalar finalize for the supported reducers
@inline _finalize(::AggMonitor{S, I, B, :sum}, acc) where {S, I, B} = acc
@inline _finalize(m::AggMonitor{S, I, B, :mean}, acc) where {S, I, B} = acc / m.n

# CPU fast path: a plain serial reduction + direct slot write (no kernel launch, no host sync).
function _aggregate!(buf::WindowBuffer{<:Array}, vals, m::AggMonitor, col)
    acc = zero(eltype(buf.window))
    @inbounds @simd for k in eachindex(vals)
        acc += vals[k]
    end
    @inbounds buf.window[1, col] = _finalize(m, acc)
    return nothing
end

# GPU path: a single-thread in-kernel reduction writing the device window slot directly (no
# host round-trip --- the GPU-resident aggregate). A parallel reduction is the M6 refinement.
@kernel function _agg_kernel!(window, col, @Const(vals), n, mean::Bool)
    acc = zero(eltype(window))
    @inbounds for k in eachindex(vals)
        acc += vals[k]
    end
    @inbounds window[1, col] = mean ? acc / n : acc
end
function _aggregate!(buf::WindowBuffer, vals, m::AggMonitor{S, I, B, R}, col) where {S, I, B, R}
    backend = get_backend(buf.window)
    _agg_kernel!(backend)(buf.window, col, vals, m.n, R === :mean; ndrange = 1)
    applicable(synchronize, backend) && synchronize(backend)
    return nothing
end

# Compile-time unroll over the monitors (dispatch-free + allocation-free, like the projection
# tuple). Recurse on the VALUES TUPLE, not the NamedTuple --- `Base.tail` on a NamedTuple of
# non-isbits monitors materialises intermediate NamedTuples (heap); on the backing tuple the
# compiler elides them when inlined.
@inline _record_all!(ms::NamedTuple, integ) = _rec_tuple!(values(ms), integ)
@inline _rec_tuple!(::Tuple{}, integ) = nothing
@inline _rec_tuple!(ms::Tuple, integ) = (record!(first(ms), integ); _rec_tuple!(Base.tail(ms), integ))

@inline _finalize_all!(ms::NamedTuple) = _fin_tuple!(values(ms))
@inline _fin_tuple!(::Tuple{}) = nothing
@inline _fin_tuple!(ms::Tuple) = (flush!(first(ms).buf); _fin_tuple!(Base.tail(ms)))

# --- Specs (user-facing) → materialised at `init` into runtime monitors ---
"""
    Trace(var; of=:all, projection=nothing, every=1)

Record the per-unit values of a state variable (`:V`, `:refrac`, …), a synaptic variable
(`:Isyn`/`:g` with `projection=i`), or an accumulator (`:gtot`/`:itot`), for the selected units.
"""
struct Trace
    var::Symbol
    of::Any
    projection::Union{Nothing, Int}
    every::Int
end
Trace(var::Symbol; of = :all, projection = nothing, every::Integer = 1) = Trace(var, of, projection, Int(every))

"""
    Spikes(; of=:all, every=1)

Record the spike mask (a `Neuron × Time` boolean raster) for the selected units.
"""
struct Spikes
    of::Any
    every::Int
end
Spikes(; of = :all, every::Integer = 1) = Spikes(of, Int(every))

"""
    Aggregate(inner, reducer; every=1)

Reduce an inner [`Trace`](@ref)/[`Spikes`](@ref) over its selected units to one scalar per step.
`reducer` is `sum` (or `:sum`) or `:mean` --- e.g. `Aggregate(Spikes(), sum)` is the population
spike count per step; `Aggregate(Trace(:V), :mean)` the population mean V. Arbitrary reductions go
through [`Probe`](@ref).
"""
struct Aggregate{S}
    inner::S
    reducer::Symbol
    every::Int
end
Aggregate(inner, reducer; every::Integer = 1) = Aggregate(inner, _reducer_sym(reducer), Int(every))
_reducer_sym(s::Symbol) = s
_reducer_sym(::typeof(sum)) = :sum

"""
    Probe(f; n, every=1)

Record an arbitrary derived quantity: `f(integrator)` must return a length-`n` vector each step
(must be GPU-kernel-safe --- broadcast/reduction, no scalar indexing --- when run on a device).
"""
struct Probe{F}
    f::F
    n::Int
    every::Int
end
Probe(f; n::Integer, every::Integer = 1) = Probe(f, Int(n), Int(every))

# selector + column count
_resolve_idx(arch, ::Colon) = Colon()
_resolve_idx(arch, of::Symbol) = of === :all ? Colon() : error("unknown selector $of (use :all or an index vector)")
_resolve_idx(arch, of::AbstractVector{<:Integer}) = on_architecture(arch, collect(Int, of))
_nsel(N, ::Colon) = N
_nsel(N, idx) = length(idx)
_ncols(nsteps, every) = cld(nsteps, every)

_srcof(t::Trace) = t.projection === nothing ? (t.var in (:gtot, :itot) ? AccumSrc{t.var}() : StateSrc{t.var}()) : SynSrc{t.projection, t.var}()

# materialise a spec → a runtime monitor (called per entry of the `record` NamedTuple at init)
function _materialize(spec::Trace, arch, ::Type{T}, N, nsteps) where {T}
    idx = _resolve_idx(arch, spec.of)
    buf = WindowBuffer(arch, T, _nsel(N, idx), _ncols(nsteps, spec.every))
    return PerUnitMonitor(_srcof(spec), idx, buf, spec.every)
end
function _materialize(spec::Spikes, arch, ::Type{T}, N, nsteps) where {T}
    idx = _resolve_idx(arch, spec.of)
    buf = WindowBuffer(arch, Bool, _nsel(N, idx), _ncols(nsteps, spec.every))
    return PerUnitMonitor(SpikeSrc(), idx, buf, spec.every)
end
function _materialize(spec::Aggregate, arch, ::Type{T}, N, nsteps) where {T}
    idx = _resolve_idx(arch, spec.inner.of)
    buf = WindowBuffer(arch, T, 1, _ncols(nsteps, spec.every))
    src = spec.inner isa Spikes ? SpikeSrc() : _srcof(spec.inner)
    return AggMonitor{typeof(src), typeof(idx), typeof(buf), spec.reducer}(src, idx, buf, spec.every, _nsel(N, idx))
end
function _materialize(spec::Probe, arch, ::Type{T}, N, nsteps) where {T}
    buf = WindowBuffer(arch, T, spec.n, _ncols(nsteps, spec.every))
    return PerUnitMonitor(ProbeSrc(spec.f), Colon(), buf, spec.every)
end

# build the monitor NamedTuple from the `record` spec NamedTuple
_make_monitors(::Nothing, arch, ::Type{T}, N, nsteps) where {T} = NamedTuple()
function _make_monitors(record::NamedTuple, arch, ::Type{T}, N, nsteps) where {T}
    return map(spec -> _materialize(spec, arch, T, N, nsteps), record)
end

# the recorded host result + metadata for a monitor (consumed by raster/firing_rate + the ext)
struct RecordResult{D}
    data::D              # host (n_out × ncols)
    idx::Any             # selected neuron indices (`Colon`/vector), or `nothing` for aggregates/probes
    every::Int
    kind::Symbol         # :trace | :spikes | :aggregate | :probe
end
_result(m::PerUnitMonitor{<:SpikeSrc}) = RecordResult(m.buf.store, m.idx, m.every, :spikes)
_result(m::PerUnitMonitor{<:ProbeSrc}) = RecordResult(m.buf.store, nothing, m.every, :probe)
_result(m::PerUnitMonitor) = RecordResult(m.buf.store, m.idx, m.every, :trace)
_result(m::AggMonitor) = RecordResult(m.buf.store, nothing, m.every, :aggregate)

export Trace, Spikes, Aggregate, Probe
