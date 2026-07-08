```@meta
CurrentModule = Dewdrop
```

# A full tour

This page walks through the majority of Dewdrop in one narrative: build a network, simulate and record
it, sweep parameters as a batched ensemble, and train a connected network by gradient descent. Every
figure below is generated live by the code shown (run during the docs build), using
[CairoMakie](https://docs.makie.org) with the [Fathom](https://github.com/brendanjohnharris/Fathom.jl)
theme:

```@setup tour
using Dewdrop
Dewdrop.set_advice!(false)   # quieten the performance advisor during the doc build
```

```@example tour
using Dewdrop, ForwardDiff, Statistics
using CairoMakie, Fathom
set_theme!(fathom())
nothing # hide
```

## 1. Build a network

The fluent builder assembles a network from named subpopulations and typed projections. Here is a
balanced excitatory/inhibitory network with conductance-based ([`ConductanceSynapse`](@ref), COBA)
synapses, wired with random [`fixed_prob`](@ref) connectivity and sustained by an external
[`PoissonDrive`](@ref). The four projections give the classic Vogels–Abbott recurrent structure.

```@example tour
NE, NI = 320, 80
E = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 5.0)

nb = network(; tspan = (0.0, 600.0))
population!(nb, :E, E, NE; input = 0.0)
population!(nb, :I, E, NI; input = 0.0)
project!(nb, :E => :E, ConductanceSynapse(; τ = 5.0,  Erev = 0.0);   p = 0.02, weight = 0.9, delay = 1.0, seed = 0x01)
project!(nb, :E => :I, ConductanceSynapse(; τ = 5.0,  Erev = 0.0);   p = 0.02, weight = 0.9, delay = 1.0, seed = 0x02)
project!(nb, :I => :E, ConductanceSynapse(; τ = 10.0, Erev = -80.0); p = 0.02, weight = 6.7, delay = 1.0, seed = 0x03)
project!(nb, :I => :I, ConductanceSynapse(; τ = 10.0, Erev = -80.0); p = 0.02, weight = 6.7, delay = 1.0, seed = 0x04)
drive!(nb, :all, PoissonDrive(; rate = 6.0, weight = 0.1, seed = 0x07))

prob = build(nb)
nothing # hide
```

`build` returns a [`DewdropNetwork`](@ref). The named subpopulations `:E` and `:I` become contiguous
ranges the solution can address symbolically (`sol[:E]`, `firing_rate(sol, :E)`); see
[building networks](networks.md).

## 2. Simulate and record

[`solve`](@ref) runs the whole window. Ask for a [`Spikes`](@ref) monitor to record the raster, then
[`raster`](@ref) unpacks the events and [`firing_rate`](@ref) gives the per-neuron rate.

```@example tour
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),))
t, ids  = raster(sol)                        # spike times (ms) and neuron ids
rate_hz = 1000 * mean(firing_rate(sol))      # population mean rate in Hz (dt in ms)
round(rate_hz; digits = 1)
```

The network settles into an asynchronous-irregular state at ≈ 55 Hz. Plotting the raster (excitatory
neurons in blue, inhibitory in red) over the population rate:

```@setup tour
# population rate over time (5 ms bins)
edges = 0:5.0:600.0
counts = zeros(length(edges) - 1)
for tk in t
    b = clamp(searchsortedlast(edges, tk), 1, length(counts))
    counts[b] += 1
end
poprate = counts ./ ((NE + NI) * step(edges) / 1000)   # Hz per neuron
bincentres = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
```

```@example tour
fig = TwoPanel()
ax1 = Axis(fig[1, 1]; ylabel = "Neuron", title = "Asynchronous-irregular COBA network")
isE = ids .<= NE
scatter!(ax1, t[isE] ./ 1000,   ids[isE];   color = Fathom.baikal,  markersize = 2)   # E blue
scatter!(ax1, t[.!isE] ./ 1000, ids[.!isE]; color = Fathom.bermejo, markersize = 2)   # I red
hidexdecorations!(ax1; grid = false)
ax2 = Axis(fig[2, 1]; xlabel = "Time (s)", ylabel = "Population rate (Hz)")
lines!(ax2, bincentres ./ 1000, poprate; color = Fathom.ianthina)
rowsize!(fig.layout, 1, Relative(0.66))
addlabels!(fig)
fig
```

## 3. Batching: parameter sweeps

An **ensemble** co-executes `B` network instances over one shared connectome in a single fused solve,
`O(edges)` in memory. Pass `syn_overrides` (or `model_overrides`) to vary a parameter per member —
**physical parameters go in directly** and Dewdrop derives the per-member coefficients, so any synapse
or neuron parameter can be swept with no boilerplate. Here we sweep the excitatory synaptic time
constant `τ` across twelve members of a smaller single-population network:

```@setup tour
# a smaller single-population base network for the sweep
base = build(let nb = network(; tspan = (0.0, 400.0))
    population!(nb, :E, E, 200; input = 0.0)
    project!(nb, :E => :E, ConductanceSynapse(; τ = 2.0, Erev = 0.0); p = 0.05, weight = 1.2, delay = 1.0, seed = 0x11)
    drive!(nb, :all, PoissonDrive(; rate = 12.0, weight = 0.12, seed = 0x17))
    nb
end)
```

```@example tour
τ_sweep = collect(range(2.0, 12.0, length = 12))
B = length(τ_sweep)

bs = solve(base, FixedStep(0.1); batch = B, syn_overrides = Dict(1 => (; τ = τ_sweep)))
sweep_rate = [1000 * mean(bs.spike_count[:, m]) / (bs.nsteps * bs.dt) for m in 1:B]

fig = OnePanel()
ax = Axis(fig[1, 1]; xlabel = "Excitatory synaptic τ (ms)", ylabel = "Population rate (Hz)")
scatterlines!(ax, τ_sweep, sweep_rate; color = Fathom.qinghai, markersize = 10)
fig
```

The whole sweep is one launch: slower excitatory synapses integrate more input, so the rate rises with
`τ`. For distinct-topology sweeps and the `batch(base; param = values)` front-end, see
[batching & ensembles](batching.md).

## 4. Training a connected network

The [`Differentiable`](@ref) backend makes a run automatically differentiable: it replaces the hard
spike with a smooth surrogate and deposits `weight · s` through a surrogate-weighted scatter, so
gradients flow to the **synaptic weights of a connected network** (and to neuron parameters). Pair it
with any autodiff tool — `ForwardDiff` for a few parameters, `Enzyme` for many.

Here we train a recurrent network's synaptic weight, by gradient descent, to *raise* its population rate
to a target. Building the network inside the loss at the differentiated element type lets a
`ForwardDiff.Dual` flow through the whole time loop and reach the weight:

```@example tour
function netrate(; w, N = 40, β = 25.0, T = typeof(w))
    m = LIF(; τ = T(20.0), EL = T(0.0), Vθ = T(20.0), Vr = T(0.0), R = T(1.0), tref = T(0.0))
    conn = fixed_prob(CPU(), N, N, 0.3; weight = T(w), delay = steps(1), seed = UInt64(1), allow_self = false)
    p = DewdropNetwork(m, N; input = T(22.0), tspan = (0.0, 250.0),
                        projections = (Projection(DeltaSynapse(), conn),))
    s = solve(p, FixedStep(0.1); backend = Differentiable(; β = β))
    return 1000 * sum(s.spike_count) / (N * Dewdrop.duration(s))     # rate in Hz
end

target = 18.0                       # raise the rate to this target
w, η = 0.5, 3.0e-4
hist, whist = Float64[], Float64[]
for _ in 1:40
    push!(hist, netrate(; w = w)); push!(whist, w)   # record rate and weight each step
    g = ForwardDiff.derivative(ww -> (netrate(; w = ww) - target)^2, w)
    global w = clamp(w - η * g, 0.0, 1.4)
end
push!(hist, netrate(; w = w)); push!(whist, w)

fig = OnePanel()
ax = Axis(fig[1, 1]; xlabel = "Gradient-descent step", ylabel = "Population rate (Hz)")
lines!(ax, 0:(length(hist) - 1), hist; color = Fathom.seohae, label = "rate")
hlines!(ax, [target]; color = Fathom.chernoe, linestyle = :dash, label = "target")
axislegend(ax; position = :rt)
fig
```

The rate climbs from ≈ 7 Hz to the 18 Hz target as the recurrent weight is tuned from 0.5 to ≈ 1.3 — a
full surrogate-gradient training loop on a connected spiking network, in a few lines. For many trainable
weights, swap `ForwardDiff` for `Enzyme` reverse-mode. See [choosing a backend](backends.md) for the
backend family and its guarantees.

The change is visible in the spiking itself. Solving the same network with a real (non-surrogate) spike
at three weights along the trajectory shows the recurrent activity growing as training proceeds:

```@example tour
function netraster(w)                       # the same network, solved for its real spikes
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 0.0)
    conn = fixed_prob(CPU(), 40, 40, 0.3; weight = w, delay = steps(1), seed = UInt64(1), allow_self = false)
    p = DewdropNetwork(m, 40; input = 22.0, tspan = (0.0, 250.0),
                        projections = (Projection(DeltaSynapse(), conn),))
    return raster(solve(p, FixedStep(0.1); record = (spikes = Spikes(),)))
end

picks = [argmin(abs.(whist .- w)) for w in range(whist[1], whist[end]; length = 3)]   # steps spanning the weight's rise
fig = Figure(size = (900, 280))
for (k, s) in enumerate(picks)
    t, id = netraster(whist[s])
    ax = Axis(fig[1, k]; xlabel = "Time (ms)", ylabel = k == 1 ? "Neuron" : "",
              title = "step $(s - 1),  w = $(round(whist[s]; digits = 2))")
    scatter!(ax, t, id; color = Fathom.baikal, markersize = 5)
    k > 1 && hideydecorations!(ax; grid = false)
end
addlabels!(fig)
fig
```

Early on (a low weight) the network fires sparsely; as the weight grows the recurrent excitation lifts
the whole population, and the raster fills in to match the rate trajectory above.

## Where to go next

- [Building networks](networks.md) — the builder, named subpopulations, and spatial layouts.
- [Neuron & synapse models](models.md) — the model zoo, [`Heterogeneous`](@ref) parameters, and [`@neuron`](@ref).
- [Batching & ensembles](batching.md) — parameter sweeps, distinct-topology batches, and analysis.
- [Recording & outputs](recording.md) — monitors, on-device reducers, and labelled outputs.
- [Running on the GPU](gpu.md) — flip `arch = GPU()` and run the same code on the device.
```
