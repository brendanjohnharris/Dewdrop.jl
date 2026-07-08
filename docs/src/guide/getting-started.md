```@meta
CurrentModule = Dewdrop
```

# Getting started

Dewdrop is a fixed-step, clock-driven, struct-of-arrays spiking neural network engine. This page
takes you from installation to a running, recorded simulation: first a single unconnected
population, then a small connected E/I network.

## Installation

Dewdrop is experimental and not yet registered. Add it from the repository:

```julia
] add https://github.com/brendanjohnharris/Dewdrop.jl
```

GPU support is loaded lazily: `using CUDA` activates the CUDA extension, after which `arch = GPU()`
runs the same code on the device. Nothing else is needed for CPU runs.

## The mental model

Two orthogonal choices govern every run, and it pays to keep them separate:

- **Architecture** (`arch = CPU()` / `GPU()`, set on the [`DewdropNetwork`](@ref)): *where* the
  state lives. One source compiles to both; see [`CPU`](@ref) and [`GPU`](@ref).
- **Execution backend** (`backend = …`, passed to [`solve`](@ref)): *how* each step runs. The
  default [`Auto`](@ref) picks a good one; see [choosing a backend](backends.md).

All backends compute the same dynamics on either architecture, so you can develop on the CPU and
move to the GPU by flipping `arch`; the results match.

Models are "model as code": a small isbits parameter struct (e.g. [`LIF`](@ref)) plus pure scalar
dynamics. You build them in plain Julia, with no DSL.

## The core loop

Dewdrop implements the CommonSolve verbs: [`init`](@ref) builds an integrator, [`step!`](@ref)
advances it, and [`solve`](@ref) runs the whole window in one call. Most of the time you only need
[`solve`](@ref).

### A single population

Define a neuron model, wrap `N` units in a [`DewdropNetwork`](@ref), and solve over `tspan` with a
[`FixedStep`](@ref) of `dt`:

```julia
using Dewdrop

m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
prob = DewdropNetwork(m, 1000; input = 1.5, tspan = (0.0, 1000.0))
sol = solve(prob, FixedStep(0.1))                 # backend = Auto()
```

`input` is the external drive: a scalar constant current shared by every unit, or a per-unit array. It is
the simplest of Dewdrop's inputs; time-varying, functional, Poisson and spike-replay stimuli are covered in
the [Inputs & stimuli](@ref) guide. The result is a [`DewdropSolution`](@ref). Without any monitors it still carries the final state and
a per-unit spike count, so [`firing_rate`](@ref) and [`duration`](@ref) work immediately.
[`firing_rate`](@ref) is exported; [`duration`](@ref) is not, so reach it as `Dewdrop.duration`
(or bring it into scope with `using Dewdrop: duration`):

```julia
firing_rate(sol)          # per-unit rate (spikes per unit dt); a length-N vector
Dewdrop.duration(sol)     # nsteps · dt
sol.state                 # final SoA state (sol.state.state.V is the membrane vector)
```

The rate is in inverse units of `dt`; multiply by 1000 for Hz when `dt` is in milliseconds.

### Recording spikes and traces

To get spike times, ask for a [`Spikes`](@ref) monitor via the `record` keyword (a `NamedTuple`
mapping a name to a monitor). [`raster`](@ref) then unpacks the recorded events:

```julia
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),))
times, ids = raster(sol)              # times[k], ids[k]: the k-th spike's time and neuron
sol.record.spikes                     # the raw recorded result
```

[`Trace`](@ref) records a state variable over time, e.g. `record = (v = Trace(:V; every = 10),)`
samples `V` every tenth step. Both monitors take `of = :name` to restrict recording to a named
subpopulation; see [recording](recording.md).

## A connected E/I network

Coupling is added with a [`Projection`](@ref): a synapse model applied over a connectivity matrix.
Here we build one population of `N` units, the first `NE` excitatory and the rest inhibitory, wire
it with random connectivity from [`fixed_prob`](@ref), and feed it external spikes with a
[`PoissonDrive`](@ref).

```@setup gs
using Dewdrop, CairoMakie, TimeseriesMakie, Fathom
set_theme!(fathom())
Dewdrop.set_advice!(false)
```

```@example gs
N, NE = 1000, 800                                  # 800 E, 200 I
m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)

# delta (instantaneous voltage-jump) synapses; weight is a per-spike voltage kick.
# Sign depends on the presynaptic neuron: E excites (+), I inhibits (-).
w(pre) = pre ≤ NE ? 0.3 : -1.2
conn = fixed_prob(CPU(), N, N, 0.02;
                  weight = w, delay = steps(1), seed = 0x01, allow_self = false)
proj = Projection(DeltaSynapse(), conn)

drive = PoissonDrive(; rate = 20.0, weight = 0.5, seed = 0x07)

prob = DewdropNetwork(m, N; input = 0.0, tspan = (0.0, 1000.0),
                      projection = proj, drive = drive)
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),))

times, ids = raster(sol)
mean_rate = 1000 * sum(firing_rate(sol)) / N       # population mean, in Hz (dt in ms)
round(mean_rate; digits = 1)
```

With `CairoMakie` and `TimeseriesMakie` loaded, `spikeraster` draws the events (here excitatory neurons
in blue, inhibitory in red); see [plotting](plotting.md).

```@example gs
keep = times .≤ 150.0                              # a legible window of the 1 s run
cols = [i ≤ NE ? Fathom.baikal : Fathom.bermejo for i in ids[keep]]   # E blue, I red
fig = Figure(size = (900, 340))
spikeraster!(Axis(fig[1, 1]; xlabel = "Time (ms)", ylabel = "Neuron", title = "E/I network raster (first 150 ms)"),
             times[keep], ids[keep]; color = cols, markersize = 1.5)
fig
```

[`fixed_prob`](@ref) gives each `(pre, post)` edge probability `p` from a reproducible
counter-based RNG, so a given `seed` always yields the same connectome. `weight` and `delay` may be
scalars or functions of the presynaptic index; the function form is how excitatory and inhibitory
neurons get opposite signs. [`steps`](@ref) sets an exact integer delay in steps (a bare number is a
delay in milliseconds). [`PoissonDrive`](@ref) adds independent external spikes per neuron, each an
instantaneous kick of `weight`, drawn reproducibly from `(seed, step, neuron)`.

For current- and conductance-based synapses ([`CurrentSynapse`](@ref), [`ConductanceSynapse`](@ref))
and named subpopulations with typed `:E => :I` projections, see [synapses and
connectivity](connectivity.md) and the [fluent builder](networks.md).

## Where to go next

- [Building networks](networks.md): the fluent [`network`](@ref) / [`population!`](@ref) /
  [`project!`](@ref) / [`build`](@ref) builder, named subpopulations, and spatial layouts.
- [Inputs & stimuli](inputs.md): time-varying / functional currents, [`TimedArray`](@ref),
  [`InhomogeneousPoisson`](@ref), and [`SpikeSourceArray`](@ref) spike replay.
- [Neuron models](models.md): [`AdaptLIF`](@ref), [`AdEx`](@ref), [`FNSNeuron`](@ref),
  [`Heterogeneous`](@ref) per-neuron parameters, [`MultiModel`](@ref) mixed populations, and
  [`@neuron`](@ref) for your own.
- [Synapses and connectivity](connectivity.md): the synapse zoo, delays, and connectome builders.
- [Choosing a backend](backends.md): [`Serial`](@ref), [`Fused`](@ref), [`Turbo`](@ref), and
  when each wins.
- [Recording](recording.md): monitors, on-device reducers, and labelled outputs.
- [Plotting](plotting.md): weak-dependency Makie recipes ([`raster`](@ref)/rate/trace/phase, plus
  positions and connectivity) that specialise TimeseriesMakie for Dewdrop solutions.
- [Running on the GPU](gpu.md): `arch = GPU()`, batching with [`batch`](@ref), and the scatter
  strategies.
