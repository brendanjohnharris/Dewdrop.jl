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

# * L3: core recipe adapters
# A solution feeds the base recipes as plain arrays: `raster` (times, ids), `_spike_raster` (the
# Neuron × Time mask) and the recorded-column time axis. No TimeseriesBase needed.

# spikeraster: a solution is a raster. Force concrete eltypes: with no spikes `raster` yields an empty
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

function _rec_times(sol)
    res = Dewdrop._find_spikes(sol.record, nothing)
    res === nothing && error("no spikes recorded; pass `record = (spikes = Spikes(),)` to `solve`")
    return Dewdrop._coltimes(res, sol.dt, sol.tspan[1])
end

# * traces (reuse TimeseriesMakie `traces`)
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
    x = Dewdrop._coltimes(res, sol.dt, sol.tspan[1])
    return (collect(x), collect(float.(neurons)), permutedims(data))   # Z: (time, unit)
end

# rows of a per-unit monitor for subpop `of` (Colon for :all), with the global neuron ids they carry
function _sub_rows(sol, res, of)
    of === :all && return (Colon(), res.idx isa Colon ? collect(1:size(res.data, 1)) : collect(res.idx))
    r = Dewdrop._subrange(sol.subpops, of)
    res.idx isa Colon || error("subpop `of = :$of` needs a full recording (idx = :all)")
    return (r, collect(r))
end

# * phase plane (reuse TimeseriesMakie `trajectory`)
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

# * positions
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

# * connectivity
Dewdrop.connectivity(x; kwargs...) = heatmap(_weight_matrix(x); kwargs...)
Dewdrop.connectivity!(ax, x; kwargs...) = heatmap!(ax, _weight_matrix(x); kwargs...)

_weight_matrix(proj::Projection) = _dense_csr(proj.conn)
_weight_matrix(csr::SparseCSR) = _dense_csr(csr)
function _weight_matrix(net::DewdropNetwork)
    isempty(net.projections) && error("network has no projections to show")
    Ms = [_dense_csr(p.conn) for p in net.projections]   # each binned onto the same grid
    shp = size(first(Ms))
    # Returning `first(Ms)` here would draw one projection while looking like the whole network: a plot
    # that silently omits connections is worse than no plot.
    all(size(M) == shp for M in Ms) || error(
        "connectivity(network): the projections bin to different shapes $(unique(size.(Ms))), so they " *
            "cannot be summed onto one grid. Plot a single projection instead, e.g. " *
            "`connectivity(net.projections[1])`."
    )
    return reduce(+, Ms)
end

# Densify a CSR into a `post × pre` weight matrix, block-mean binned so neither dimension exceeds
# `maxdim`. Binning happens while walking the edges, so the full `npost × npre` dense form is never
# allocated (it is 8 GB at N = 32k). The trailing partial block is kept and divided by its true size,
# so no neuron is dropped from the edge of the plot.
function _dense_csr(csr::SparseCSR; maxdim::Integer = 2048)
    post, src, w = collect(csr.post), collect(csr.src), collect(csr.weight)
    r = max(cld(csr.npost, maxdim), 1)
    c = max(cld(csr.npre, maxdim), 1)
    nr, nc = cld(csr.npost, r), cld(csr.npre, c)
    M = zeros(float(eltype(w)), nr, nc)
    @inbounds for e in eachindex(post)
        M[cld(post[e], r), cld(src[e], c)] += w[e]
    end
    if r > 1 || c > 1
        @warn "connectivity: binning a $(csr.npost)×$(csr.npre) weight matrix to $(nr)×$(nc)" maxlog = 1
        for j in 1:nc, i in 1:nr        # divide by the block's true size (the last one is partial)
            @inbounds M[i, j] /= (min(i * r, csr.npost) - (i - 1) * r) * (min(j * c, csr.npre) - (j - 1) * c)
        end
    end
    return M
end

end # module TimeseriesMakieExt
