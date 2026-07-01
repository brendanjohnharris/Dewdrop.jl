module TimeseriesBaseExt

# Optional labeled outputs: wrap a `DewdropSolution`'s recorded arrays in TimeseriesBase
# `ToolsArray`s carrying meaningful dimensions (`Time`, `Neuron`), so traces and rasters display,
# plot and analyse with proper axes. The core stays plain-array; `using TimeseriesBase` activates
# this. Custom `Neuron`/`Synapse` dimensions follow the TimeseriesBase `Var`/`Obs` pattern (a
# `ToolsDim` subtype, so the results are proper `ToolsArray`s, not bare `DimArray`s).

# `import` (not `using`) Dewdrop so the local `Population` dimension below does not collide with
# Dewdrop's exported SoA-state `Population` struct; the ext refers to Dewdrop names qualified.
import Dewdrop
import TimeseriesBase: Timeseries, spiketrain, ToolsArray, ToolsDim, 𝑡, Var
import DimensionalData

# Custom Dewdrop dimensions
abstract type NeuronDim{T} <: ToolsDim{T} end
DimensionalData.@dim Neuron NeuronDim "Neuron"
abstract type SynapseDim{T} <: ToolsDim{T} end
DimensionalData.@dim Synapse SynapseDim "Synapse"
# The `Population` LABELLED-OUTPUT dimension (matching the WRCircuit bpformat convention). Its name
# clashes with Dewdrop's core SoA `Population` struct, so it is NOT injected into Dewdrop's namespace
# reference it on a result by its name symbol instead, e.g. `dims(X, :Population)`.
abstract type PopulationDim{T} <: ToolsDim{T} end
DimensionalData.@dim Population PopulationDim "Population"

# Expose the per-unit dims in Dewdrop's namespace (so `Dewdrop.Neuron(...)` / `import Dewdrop: Neuron`
# work once TimeseriesBase is loaded). Done at load time, not precompile, to avoid mutating the parent
# module during its precompilation. (`Var`/`Obs`/`𝑡` come from TimeseriesBase itself; `Population` is
# referenced by symbol, see above.)
#
# The `jl_generating_output` guard is load-bearing: when a DOWNSTREAM package that depends on both
# Dewdrop and TimeseriesBase is precompiled, this ext's `__init__` runs while that package's output is
# being generated, and evaluating into the (sealed) `Dewdrop` module then throws "Evaluation into the
# closed module Dewdrop breaks incremental compilation". Skipping the injection during ANY precompile
# output generation avoids that; the names are still injected at interactive/runtime load (when they
# are actually used), and a downstream package that wants the `Neuron`/`Synapse` dims defines its own.
function __init__()
    ccall(:jl_generating_output, Cint, ()) == 1 && return nothing
    isdefined(Dewdrop, :Neuron) || Core.eval(Dewdrop, :(const Neuron = $Neuron))
    isdefined(Dewdrop, :Synapse) || Core.eval(Dewdrop, :(const Synapse = $Synapse))
    return nothing
end

# recorded column c of a monitor sampled every `e` steps → time c·e·dt (matching `raster`)
_times(res, sol) = (1:size(res.data, 2)) .* (res.every * sol.dt)
_neurons(res) = res.idx isa Colon ? collect(1:size(res.data, 1)) : collect(res.idx)

# rows of a per-unit monitor restricted to subpopulation `of` (and the global neuron indices they
# carry). `of = :all` keeps everything; a named subpop needs a full recording (idx = :all) so its
# range maps to rows directly.
function _sub_rows(sol, res, of)
    of === :all && return (Colon(), _neurons(res))
    r = Dewdrop._subrange(sol.subpops, of)
    res.idx isa Colon || error("subpop labelling (`of = :$of`) needs a full recording (idx = :all)")
    return (r, collect(r))
end

"""
    Timeseries(sol::DewdropSolution, name=:V; of=:all)

The named recorded monitor as a labeled `Timeseries`: a per-unit `Trace`/`Probe` → `Time × Neuron`
(the `Neuron` axis carries the actual recorded indices); an `Aggregate` → a univariate `Time`
series. `of` (a subpopulation symbol, e.g. `:E`) restricts the `Neuron` axis to that subpopulation.
Use [`spiketrain`](@ref) for a `Spikes` monitor.
"""
function Timeseries(sol::Dewdrop.DewdropSolution, name::Symbol = :V; of = :all, lazy::Bool = false)
    res = sol.record[name]
    if res.kind === :aggregate
        of === :all || error("aggregate monitors have no neuron axis to restrict with `of`")
        return ToolsArray(vec(res.data), 𝑡(_times(res, sol)); name = name)
    end
    rows, neurons = _sub_rows(sol, res, of)
    data = rows === Colon() ? res.data : (lazy ? view(res.data, rows, :) : res.data[rows, :])
    # per-unit: stored (unit, time); transpose to time-first so it is a proper Timeseries. `lazy = true`
    # keeps the transpose a `PermutedDimsArray` VIEW (no copy of the whole trace): e.g. for `bpformat`
    # over a large population × long run, where an eager `permutedims` would double the (already large)
    # recorded data. The default stays an eager copy (a contiguous, standalone array).
    mat = lazy ? PermutedDimsArray(data, (2, 1)) : permutedims(data)
    return ToolsArray(mat, (𝑡(_times(res, sol)), Neuron(neurons)); name = name)
end

"""
    Timeseries(sol::DewdropSolution, populations; vars=[:V])

The recorded traces as a `Population × Var` nested `ToolsArray`, each cell the per-population
`Time × Neuron` timeseries of one variable, matching the WRCircuit `bpsolve`/`bpformat` output
shape (the populations are named subpopulations, the variables recorded `Trace` monitor names).
"""
function Timeseries(sol::Dewdrop.DewdropSolution, populations::AbstractVector; vars = [:V])
    X = [Timeseries(sol, Symbol(v); of = Symbol(p)) for p in populations, v in vars]
    return ToolsArray(X, (Population(collect(Symbol.(populations))), Var(collect(Symbol.(vars)))))
end

# Extend TimeseriesBase's `spiketrain` (which builds a SpikeTrain from spike times) with a
# DewdropSolution method: a `Spikes` monitor as a labeled, binary `Neuron × Time` timeseries.
"""
    spiketrain(sol::DewdropSolution, name=nothing; of=:all)

A recorded `Spikes` monitor as a labeled binary `Neuron × Time` timeseries (`name` picks one;
default: the first spike monitor). `of` restricts the `Neuron` axis to a named subpopulation.
"""
function spiketrain(sol::Dewdrop.DewdropSolution, name = nothing; of = :all)
    res = Dewdrop._find_spikes(sol.record, name)
    res === nothing && error("no spikes recorded; pass `record = (spikes = Spikes(),)` to `solve`")
    rows, neurons = _sub_rows(sol, res, of)
    data = rows === Colon() ? res.data : res.data[rows, :]
    return ToolsArray(data, (Neuron(neurons), 𝑡(_times(res, sol))); name = :spikes)
end

end # module TimeseriesBaseExt
