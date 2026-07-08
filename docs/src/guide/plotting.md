```@meta
CurrentModule = Dewdrop
```

# Plotting

Dewdrop's plots are a weak-dependency layer: the base neural recipes live in
[TimeseriesMakie](https://github.com/brendanjohnharris/TimeseriesMakie.jl), and Dewdrop specialises
them for its own solution types. Load a Makie backend together with TimeseriesMakie to activate
everything:

```julia
using Dewdrop
using CairoMakie, TimeseriesMakie      # CairoMakie for static plots; GLMakie for interactive
```

Nothing is pulled in unless you ask for it; the core package never depends on Makie. The recipes are
theme-agnostic, so [Fathom](https://github.com/brendanjohnharris/Fathom.jl) (or Foresight) styling
applies automatically once set:

```julia
using Fathom
set_theme!(fathom())                   # excitatory blue, inhibitory red, and the Fathom palette
```

The recipes below, on one spatial E/I network: the spike raster, population rate, per-neuron rate map,
and neuron positions.

```@setup plotting
using Dewdrop, CairoMakie, TimeseriesMakie, Fathom
set_theme!(fathom())
Dewdrop.set_advice!(false)
```

```@example plotting
N, NE = 200, 160
m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 2.0)
pos = random_positions(N, (1.0, 1.0); seed = UInt(11))
w(pre) = pre ≤ NE ? 0.4 : -1.6
conn = fixed_prob(CPU(), N, N, 0.1; weight = w, delay = steps(1), seed = 0x5, allow_self = false)
input = [17.0 + 9.0 * (i - 1) / (N - 1) for i in 1:N]
prob = DewdropNetwork(m, N; input = input, tspan = (0.0, 500.0),
    projection = Projection(DeltaSynapse(), conn), positions = pos,
    noise = WhiteNoise(6.0; seed = UInt64(23)), subpops = (E = 1:NE, I = (NE + 1):N))
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(), v = Trace(:V; every = 5)))

fig = FourPanel()
t, id = raster(sol)
cols = [i ≤ NE ? Fathom.baikal : Fathom.bermejo for i in id]
spikeraster!(Axis(fig[1, 1]; xlabel = "Time (ms)", ylabel = "Neuron", title = "Spike raster"),
             t, id; color = cols, markersize = 3)
psth!(Axis(fig[1, 2]; xlabel = "Time (ms)", ylabel = "Rate (per ms per neuron)", title = "Population rate"),
      sol; binwidth = 5.0, nneurons = N)
ratemap!(Axis(fig[2, 1]; xlabel = "Time (ms)", ylabel = "Neuron", title = "Rate map"), sol; binwidth = 50)
positionplot!(Axis(fig[2, 2]; xlabel = "x", ylabel = "y", title = "Positions (by type)"), sol; color = :type)
addlabels!(fig)
fig
```

## Spike rasters

With a [`Spikes`](@ref) recording, a [`DewdropSolution`](@ref) *is* a raster: `plot(sol)` scatters it
directly, or call `spikeraster` for control.

```julia
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),))

plot(sol)                              # bare plot → raster (via plottype)
spikeraster(sol)                       # the same, explicitly
spikeraster(sol; sortby = :rate)       # order neurons by firing rate (or :rev, a function, a vector)
spikeraster!(ax, sol[:E])              # a named subpopulation only (a SubSolution)
```

`spikeraster` also takes plain data: flat `(times, ids)` (as returned by [`raster`](@ref)), a vector
of per-neuron spike-time vectors, or a `Neuron × Time` boolean matrix.

## Population rate and rate maps

`psth` bins spikes into a peri-stimulus time histogram; `ratemap` shows the per-neuron time-resolved
rate as a `Neuron × Time` heatmap.

```julia
psth(sol; binwidth = 5.0)              # population rate over time (spikes per unit time)
psth(sol; binwidth = 5.0, normalization = :count, nneurons = 800)
ratemap(sol; binwidth = 10)            # coarse-grained per-neuron rate image
```

`firing_rate`, `psth`, and `mua` report rates in inverse units of `dt` (so `1/ms` when `dt` is in
milliseconds); multiply by 1000 for Hz.

## State traces and phase planes

[`traceplot`](@ref) overlays the recorded traces of a [`Trace`](@ref) monitor (reusing TimeseriesMakie's
`traces`); [`phaseplane`](@ref) draws one neuron's two-variable trajectory (reusing `trajectory`),
e.g. the `(V, w)` plane of an [`AdEx`](@ref) unit.

```julia
sol = solve(prob, FixedStep(0.1); record = (v = Trace(:V), w = Trace(:w)))

traceplot(sol, :v)                     # stacked V(t) traces, one per unit
traceplot(sol, :v; of = :E)            # restrict to a subpopulation
phaseplane(sol; vars = (:v, :w), neuron = 1)
```

`traceplot`/`phaseplane` take the monitor's *record key* (the name in the `record` NamedTuple), not
the state-variable symbol.

## Network structure

[`positionplot`](@ref) scatters a spatial network's neuron positions (2-D or 3-D), and
[`connectivity`](@ref) shows a weight matrix. These describe network structure rather than time, so
they are Dewdrop-only (not part of TimeseriesMakie).

```julia
positionplot(sol)                      # colored by firing rate (default)
positionplot(sol; color = :type)       # colored by named subpopulation
connectivity(prob)                     # weight matrix of a network (or a Projection / SparseCSR)
```

A large connectome is block-mean binned so the image stays bounded.

## Composing figures

Every recipe has a mutating form (`spikeraster!`, `psth!`, …) that draws into an existing axis, so
they compose into Fathom panel layouts:

```julia
fig = TwoPanel()
spikeraster!(Axis(fig[1, 1]; xlabel = "Time (ms)", ylabel = "Neuron"), sol)
psth!(Axis(fig[1, 2]; xlabel = "Time (ms)", ylabel = "Rate"), sol; binwidth = 5.0)
addlabels!(fig)
```

See the [reference](@ref "Reference") for the full signatures.
