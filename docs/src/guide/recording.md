```@meta
CurrentModule = Dewdrop
```

# Recording and outputs

Recording is opt-in: by default a run keeps only the final state and an always-on per-unit spike
count (so [`firing_rate`](@ref) costs nothing). Anything time-resolved is requested with the
`record` keyword to [`solve`](@ref) (or [`init`](@ref)), a NamedTuple mapping a name you choose to a
monitor:

```julia
sol = solve(prob, FixedStep(0.1); record = (V = Trace(:V), spikes = Spikes()))
sol.record.V        # a Neuron × Time matrix of membrane potentials
sol.record.spikes   # a Neuron × Time boolean raster
```

Each monitor stages into an architecture-resident window buffer that flushes to a host store in
windows (an internal `WindowBuffer`), so recording on a [`GPU`](@ref) stays device-resident and
incurs O(1) host transfers per window rather than one per step. Recording therefore follows the
network's [architecture](backends.md) without you copying anything by hand.

## The four monitors

| monitor | records | shape of `sol.record.<name>` |
|---|---|---|
| [`Trace`](@ref)`(var; of, projection, every)` | a per-unit state, synaptic, or accumulator variable | `Neuron × Time` |
| [`Spikes`](@ref)`(; of, every)` | the spike mask | `Neuron × Time` (`Bool`) |
| [`Aggregate`](@ref)`(inner, reducer; every)` | an inner monitor reduced over its units, one scalar per step | `1 × Time` |
| [`Probe`](@ref)`(f; n, every)` | an arbitrary `f(integrator)` returning a length-`n` vector | `n × Time` |

[`Trace`](@ref) reads a state-variable column by symbol (`Trace(:V)`, `Trace(:refrac)`). The same
constructor reaches two other sources: the dense membrane accumulators with `Trace(:gtot)` /
`Trace(:itot)`, and a synaptic variable of projection `i` with `Trace(:g_decay; projection = i)`.
Accumulators are only materialised by the [`Serial`](@ref), [`Fused`](@ref), and [`Turbo`](@ref)
backends; recording them under another backend errors rather than returning zeros (synaptic
variables are materialised under every backend).

[`Aggregate`](@ref) wraps an inner [`Trace`](@ref) or [`Spikes`](@ref) and collapses its selected
units to one scalar per step. The `reducer` is `sum` (or `:sum`) or `:mean`; the reduction runs in
place on the architecture (a single in-kernel reduction on the GPU, no host round-trip). For example
`Aggregate(Spikes(), sum)` is the population spike count per step and `Aggregate(Trace(:V), :mean)`
the population mean voltage. Anything beyond sum/mean goes through [`Probe`](@ref).

[`Probe`](@ref) records a derived quantity: `f(integrator)` must return a length-`n` vector each
step and, when the run is on a device, must be GPU-kernel-safe (broadcast/reduction, no scalar
indexing).

## Selecting units and stride

Per-unit monitors take an `of` selector and every monitor takes an `every` stride:

- `of = :all` (the default) records every unit.
- `of = [1, 5, 9]` records an explicit index vector.
- `of = :E` records a named subpopulation; the symbol resolves against the registry built by the
  [fluent builder](networks.md), and an unknown name errors with the available names listed.
- `every = k` samples once every `k` steps, so the recorded array has `cld(nsteps, k)` columns.

```julia
sol = solve(prob, FixedStep(0.1);
    record = (Ve = Trace(:V; of = :E, every = 10),   # E voltages, downsampled 10×
              rate = Aggregate(Spikes(of = :E), sum),  # E spike count per step
              spikes = Spikes()))                      # full raster
```

## The solution

[`solve`](@ref) returns a [`DewdropSolution`](@ref). Its fields:

| field | meaning |
|---|---|
| `state` | the final SoA state (a `Population`) |
| `spike_count` | per-unit spike total (always recorded) |
| `nsteps`, `dt`, `tspan` | the fixed-step window |
| `record` | a NamedTuple of the requested monitors' results, keyed by your names |
| `subpops` | the named-subpopulation registry |
| `positions` | per-neuron positions, or `nothing` |

Helpers read these without you indexing the raw arrays:

```julia
firing_rate(sol)            # spike_count ./ duration(sol), per unit
firing_rate(sol, :E)        # restricted to a named subpopulation
duration(sol)               # nsteps · dt
times, ids = raster(sol)    # spike events from a Spikes() monitor
times, ids = raster(sol; of = :E)   # restricted to :E, ids rebased into 1:|E|
```

[`duration`](@ref) is the simulated time and [`firing_rate`](@ref) is per-unit rate in inverse units
of `dt`. [`raster`](@ref) flattens a recorded [`Spikes`](@ref) monitor into parallel event vectors;
with no `name` it uses the first spike monitor, `name` picks a specific one, and `of` restricts to a
subpopulation.

## Named subpopulations

Indexing a solution by a subpopulation symbol returns a [`SubSolution`](@ref), a lightweight view
(no copy) over that subpopulation's range:

```julia
sub = sol[:E]               # SubSolution over the E range
firing_rate(sub)            # per-unit rate of E
duration(sub)               # same window as the parent
```

[`SubSolution`](@ref) carries the subpopulation's `state` and `spike_count` views and works with
[`firing_rate`](@ref) and [`duration`](@ref).

## Labelled outputs (TimeseriesBase)

Loading TimeseriesBase activates an extension that wraps recorded arrays in `ToolsArray`s with real
axes (`Time`, `Neuron`), so traces and rasters carry units and plot with proper axes:

```julia
using TimeseriesBase
Timeseries(sol, :V)             # a per-unit Trace → Time × Neuron
Timeseries(sol, :V; of = :E)    # restricted to subpopulation E
Timeseries(sol, :rate)          # an Aggregate → a univariate Time series
spiketrain(sol, :spikes)        # a Spikes monitor → binary Neuron × Time
```

A `Population × Var` nested array (the WRCircuit `bpformat` shape) follows from passing a vector of
subpopulation names:

```julia
Timeseries(sol, [:E, :I]; vars = [:V])   # Population × Var of Time × Neuron cells
```

The `Population` dimension is referenced by its name symbol (e.g. `dims(X, :Population)`); its name
clashes with Dewdrop's core SoA `Population` struct, so the extension does not inject it into the
Dewdrop namespace.
