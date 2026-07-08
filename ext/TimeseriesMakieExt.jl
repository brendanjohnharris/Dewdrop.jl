module TimeseriesMakieExt

# Weak-dependency plotting layer. The base neural recipes (`spikeraster`/`psth`/`ratemap`) live in
# TimeseriesMakie; here we only specialise them for Dewdrop's solution types (via `convert_arguments`
# + `plottype`, so `plot(sol)` rasters) and provide the trace/phase/position/connectivity adapters
# declared in src/Plotting.jl. Triggered by `["TimeseriesMakie", "Makie"]`: loading TimeseriesMakie
# always loads Makie, so this activates whenever TimeseriesMakie is present.

import Dewdrop
import Dewdrop: DewdropSolution, SubSolution, DewdropNetwork, Projection, SparseCSR
using Makie
import TimeseriesMakie: SpikeRaster, PSTH, RateMap, traces, traces!, trajectory, trajectory!

# ─────────────────────────── L3: core recipe adapters ───────────────────────────
# A solution feeds the base recipes as plain arrays: `raster` (times, ids), `_spike_raster` (the
# Neuron × Time mask) and the recorded-column time axis. No TimeseriesBase needed.

# spikeraster: a solution IS a raster. Force concrete eltypes: with no spikes `raster` yields an empty
# `Vector{Any}`, which Makie's argument conversion cannot reduce over.
_typed_raster(sol; kw...) = (r = Dewdrop.raster(sol; kw...); (Float64.(r[1]), Int.(r[2])))
Makie.convert_arguments(::Type{<:SpikeRaster}, sol::DewdropSolution) = _typed_raster(sol)
Makie.convert_arguments(::Type{<:SpikeRaster}, ss::SubSolution) = _typed_raster(ss.parent; of = ss.name)
Makie.plottype(::DewdropSolution) = SpikeRaster       # bare `plot(sol)` → raster
Makie.plottype(::SubSolution) = SpikeRaster

# psth: the pooled spike times
Makie.convert_arguments(::Type{<:PSTH}, sol::DewdropSolution) = (first(_typed_raster(sol)),)
Makie.convert_arguments(::Type{<:PSTH}, ss::SubSolution) = (first(_typed_raster(ss.parent; of = ss.name)),)

# ratemap: the Neuron × Time spike mask + the real (recorded) time axis
Makie.convert_arguments(::Type{<:RateMap}, sol::DewdropSolution) = (_rec_times(sol), Dewdrop._spike_raster(sol))
Makie.convert_arguments(::Type{<:RateMap}, ss::SubSolution) =
    (_rec_times(ss.parent), Dewdrop._spike_raster(ss.parent; of = ss.name))

# recorded-column times: column c of a monitor sampled every `e` steps → time c·e·dt (matches `raster`)
function _rec_times(sol)
    res = Dewdrop._find_spikes(sol.record, nothing)
    res === nothing && error("no spikes recorded; pass `record = (spikes = Spikes(),)` to `solve`")
    return (1:size(res.data, 2)) .* (res.every * sol.dt)
end

# ─────────────────────────── traces (reuse TimeseriesMakie `traces`) ───────────────────────────
function Dewdrop.traceplot(sol::DewdropSolution, name::Symbol = :V; of = :all, kwargs...)
    x, colorby, Z = _trace_args(sol, name, of)
    return traces(x, colorby, Z; kwargs...)
end
function Dewdrop.traceplot!(ax, sol::DewdropSolution, name::Symbol = :V; of = :all, kwargs...)
    x, colorby, Z = _trace_args(sol, name, of)
    return traces!(ax, x, colorby, Z; kwargs...)
end

# (time vector, per-trace color values = neuron ids, time×unit matrix) from a per-unit Trace monitor
function _trace_args(sol, name, of)
    res = sol.record[name]
    res.kind === :aggregate && error("`traceplot` needs a per-unit `Trace` monitor; `$name` is an aggregate")
    rows, neurons = _sub_rows(sol, res, of)
    data = rows === Colon() ? res.data : res.data[rows, :]        # (unit, time)
    x = (1:size(data, 2)) .* (res.every * sol.dt)
    return (collect(x), collect(float.(neurons)), permutedims(data))   # Z: (time, unit)
end

# rows of a per-unit monitor for subpop `of` (Colon for :all), with the global neuron ids they carry
function _sub_rows(sol, res, of)
    of === :all && return (Colon(), res.idx isa Colon ? collect(1:size(res.data, 1)) : collect(res.idx))
    r = Dewdrop._subrange(sol.subpops, of)
    res.idx isa Colon || error("subpop `of = :$of` needs a full recording (idx = :all)")
    return (r, collect(r))
end

# ─────────────────────────── phase plane (reuse TimeseriesMakie `trajectory`) ───────────────────────────
function Dewdrop.phaseplane(sol::DewdropSolution; vars = (:V, :w), neuron::Integer = 1, kwargs...)
    a, b = _phase_args(sol, vars, neuron)
    return trajectory(a, b; kwargs...)
end
function Dewdrop.phaseplane!(ax, sol::DewdropSolution; vars = (:V, :w), neuron::Integer = 1, kwargs...)
    a, b = _phase_args(sol, vars, neuron)
    return trajectory!(ax, a, b; kwargs...)
end
function _phase_args(sol, vars, neuron)
    length(vars) == 2 || throw(ArgumentError("`vars` must be a pair of record names, e.g. (:V, :w)"))
    return (_trace_row(sol, vars[1], neuron), _trace_row(sol, vars[2], neuron))
end
function _trace_row(sol, name, neuron)
    res = sol.record[name]
    row = res.idx isa Colon ? Int(neuron) : findfirst(==(neuron), res.idx)
    row === nothing && error("neuron $neuron not recorded in monitor `$name`")
    return collect(res.data[row, :])
end

# ─────────────────────────── positions ───────────────────────────
function Dewdrop.positionplot(sol::DewdropSolution; color = :rate, kwargs...)
    pts = _pos_points(sol)
    fig = Figure()
    ax = eltype(pts) <: Point3 ? Axis3(fig[1, 1]) : Axis(fig[1, 1])
    p = scatter!(ax, pts; color = _pos_color(sol, color), kwargs...)
    return Makie.FigureAxisPlot(fig, ax, p)
end
Dewdrop.positionplot!(ax, sol::DewdropSolution; color = :rate, kwargs...) =
    scatter!(ax, _pos_points(sol); color = _pos_color(sol, color), kwargs...)

function _pos_points(sol)
    pos = sol.positions
    pos === nothing && error("this network has no positions; build it with `positions = ...`")
    return length(first(pos)) == 3 ? [Point3f(p...) for p in pos] : [Point2f(p...) for p in pos]
end
function _pos_color(sol, color)
    color === :rate && return Dewdrop.firing_rate(sol)
    color === :type && return _subpop_index(sol)
    return color                                    # a per-neuron vector or a fixed color
end
# per-neuron integer subpopulation id (skipping the implicit `:all`), for `color = :type`
function _subpop_index(sol)
    idx = ones(Int, length(sol.spike_count))
    k = 0
    for (name, r) in pairs(sol.subpops)
        name === :all && continue
        k += 1
        idx[r] .= k
    end
    return idx
end

# ─────────────────────────── connectivity ───────────────────────────
Dewdrop.connectivity(x; kwargs...) = heatmap(_weight_matrix(x); kwargs...)
Dewdrop.connectivity!(ax, x; kwargs...) = heatmap!(ax, _weight_matrix(x); kwargs...)

_weight_matrix(proj::Projection) = _dense_csr(proj.conn)
_weight_matrix(csr::SparseCSR) = _dense_csr(csr)
function _weight_matrix(net::DewdropNetwork)
    isempty(net.projections) && error("network has no projections to show")
    Ms = [_dense_csr(p.conn; maxdim = typemax(Int)) for p in net.projections]   # bin the sum, not each
    shp = size(first(Ms))
    all(size(M) == shp for M in Ms) || return _maybe_bin(first(Ms))
    return _maybe_bin(reduce(+, Ms))
end

# densify a CSR into a `post × pre` weight matrix (host-side), then bound its size
function _dense_csr(csr::SparseCSR; maxdim::Integer = 2048)
    post = collect(csr.post); src = collect(csr.src); w = collect(csr.weight)
    M = zeros(eltype(w), csr.npost, csr.npre)
    @inbounds for e in eachindex(post)
        M[post[e], src[e]] = w[e]
    end
    return _maybe_bin(M, maxdim)
end

# block-mean downsample so neither dimension exceeds `maxdim` (keeps a dense connectome plottable)
function _maybe_bin(M::AbstractMatrix, maxdim::Integer = 2048)
    (size(M, 1) <= maxdim && size(M, 2) <= maxdim) && return M
    @warn "connectivity: binning a $(size(M)) weight matrix to ≤ $(maxdim)²" maxlog = 1
    r = cld(size(M, 1), maxdim); c = cld(size(M, 2), maxdim)
    nr = size(M, 1) ÷ r; nc = size(M, 2) ÷ c
    out = zeros(float(eltype(M)), nr, nc)
    @inbounds for j in 1:nc, i in 1:nr
        acc = 0.0
        for ii in ((i - 1) * r + 1):(i * r), jj in ((j - 1) * c + 1):(j * c)
            acc += M[ii, jj]
        end
        out[i, j] = acc / (r * c)
    end
    return out
end

end # module TimeseriesMakieExt
