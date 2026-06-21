module TimeseriesBaseExt

# Optional labeled outputs: wrap a `DewdropSolution`'s recorded arrays in TimeseriesBase
# `ToolsArray`s carrying meaningful dimensions (`Time`, `Neuron`), so traces and rasters display,
# plot and analyse with proper axes. The core stays plain-array; `using TimeseriesBase` activates
# this. Custom `Neuron`/`Synapse` dimensions follow the TimeseriesBase `Var`/`Obs` pattern (a
# `ToolsDim` subtype, so the results are proper `ToolsArray`s, not bare `DimArray`s).

using Dewdrop
import TimeseriesBase: Timeseries, spiketrain, ToolsArray, ToolsDim, 𝑡
import DimensionalData

# --- Custom Dewdrop dimensions ---
abstract type NeuronDim{T} <: ToolsDim{T} end
DimensionalData.@dim Neuron NeuronDim "Neuron"
abstract type SynapseDim{T} <: ToolsDim{T} end
DimensionalData.@dim Synapse SynapseDim "Synapse"

# Expose the dims in Dewdrop's namespace (so `Dewdrop.Neuron(...)` / `import Dewdrop: Neuron`
# work once TimeseriesBase is loaded). Done at load time, not precompile, to avoid mutating the
# parent module during its precompilation.
function __init__()
    isdefined(Dewdrop, :Neuron) || Core.eval(Dewdrop, :(const Neuron = $Neuron))
    isdefined(Dewdrop, :Synapse) || Core.eval(Dewdrop, :(const Synapse = $Synapse))
    return nothing
end

# recorded column c of a monitor sampled every `e` steps → time c·e·dt (matching `raster`)
_times(res, sol) = (1:size(res.data, 2)) .* (res.every * sol.dt)
_neurons(res) = res.idx isa Colon ? collect(1:size(res.data, 1)) : collect(res.idx)

"""
    Timeseries(sol::DewdropSolution, name=:V)

The named recorded monitor as a labeled `Timeseries`: a per-unit `Trace`/`Probe` → `Time × Neuron`
(the `Neuron` axis carries the actual recorded indices); an `Aggregate` → a univariate `Time`
series. Use [`spiketrain`](@ref) for a `Spikes` monitor.
"""
function Timeseries(sol::Dewdrop.DewdropSolution, name::Symbol = :V)
    res = sol.record[name]
    if res.kind === :aggregate
        return ToolsArray(vec(res.data), 𝑡(_times(res, sol)); name = name)
    end
    # per-unit: stored (unit, time); transpose to time-first so it is a proper Timeseries
    return ToolsArray(permutedims(res.data), (𝑡(_times(res, sol)), Neuron(_neurons(res))); name = name)
end

# Extend TimeseriesBase's `spiketrain` (which builds a SpikeTrain from spike times) with a
# DewdropSolution method: a `Spikes` monitor as a labeled, binary `Neuron × Time` timeseries.
"""
    spiketrain(sol::DewdropSolution, name=nothing)

A recorded `Spikes` monitor as a labeled binary `Neuron × Time` timeseries (`name` picks one;
default: the first spike monitor).
"""
function spiketrain(sol::Dewdrop.DewdropSolution, name = nothing)
    res = Dewdrop._find_spikes(sol.record, name)
    res === nothing && error("no spikes recorded --- pass `record = (spikes = Spikes(),)` to `solve`")
    return ToolsArray(res.data, (Neuron(_neurons(res)), 𝑡(_times(res, sol))); name = :spikes)
end

end # module TimeseriesBaseExt
