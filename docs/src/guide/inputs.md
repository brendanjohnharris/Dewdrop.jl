```@meta
CurrentModule = Dewdrop
```

# Inputs & stimuli

Every external input to a network --- a constant current, a Poisson background drive, membrane noise, a
time-varying waveform, a replayed spike train --- is an `AbstractStimulus` applied at one point in the
per-neuron step. The three most common are exposed as keyword sugar on [`DewdropNetwork`](@ref) and the
builder; everything else is passed through `stimuli =`.

| Kwarg | Lowers to | Point |
|:---|:---|:---|
| `input =` | [`ConstantCurrent`](@ref) | current (`itot`) |
| `drive =` | [`PoissonDrive`](@ref) | voltage kick |
| `noise =` | [`WhiteNoise`](@ref) | membrane noise |
| `stimuli =` | any `AbstractStimulus` (or a tuple) | its own point |

Stimuli of the current kind sum into `itot`; kicks add to `V` at delivery; noise adds under the refractory
gate. All families run identically on every execution backend (Serial, Fused-CPU, GPU megakernel, batched),
and are byte-identical to the hand-written path for the three legacy inputs.

## Constant current

`input =` is a time-invariant current: a shared scalar, or a per-neuron vector.

```julia
DewdropNetwork(LIF(), 100; input = 0.3, tspan = (0.0, 1.0))          # shared
DewdropNetwork(LIF(), 100; input = range(0, 0.5; length = 100), tspan = (0.0, 1.0))  # per-neuron
```

To combine a constant current with other stimuli, pass it explicitly as [`ConstantCurrent`](@ref) in
`stimuli =`.

## Poisson drive and membrane noise

[`PoissonDrive`](@ref) gives every neuron an independent background of external spikes each step (a voltage
kick `weight · Poisson(rate · dt)`); [`WhiteNoise`](@ref) adds an exact Ornstein--Uhlenbeck membrane
increment. Both draw from the reproducible counter-based RNG.

```julia
DewdropNetwork(LIF(), 100; input = 0.0, tspan = (0.0, 1.0),
               drive = PoissonDrive(rate = 10.0, weight = 0.1, seed = 1),
               noise = WhiteNoise(2e-3; seed = 2))
```

## Time-varying inputs

A time-varying stimulus lets the network *respond* to a signal. Below, a single [`sinusoid`](@ref)
current drives a population of LIF neurons (with a little membrane noise): one neuron's membrane
follows the drive and fires near each peak (b), and the whole population entrains to the input, its
raster (c) and rate (d) tracking the stimulus (a).

```@setup inputs
using Dewdrop, CairoMakie, TimeseriesMakie, Fathom
set_theme!(fathom())
Dewdrop.set_advice!(false)
```

```@example inputs
m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 3.0)
stim = sinusoid(amplitude = 11.0, freq = 0.012, offset = 13.0)
prob = DewdropNetwork(m, 200; input = 0.0, tspan = (0.0, 400.0),
                      stimuli = stim, noise = WhiteNoise(3.0; seed = UInt64(4)))
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(), v = Trace(:V)))

fig = FourPanel()
tt = range(0, 400; length = 800)
lines!(Axis(fig[1, 1]; xlabel = "Time (ms)", ylabel = "Input current (a.u.)", title = "Stimulus"),
       tt, @. 13.0 + 11.0 * sin(2π * 0.012 * tt); color = Fathom.seohae)
axv = Axis(fig[1, 2]; xlabel = "Time (ms)", ylabel = "Membrane potential (mV)", title = "Single neuron")
lines!(axv, (1:size(sol.record.v.data, 2)) .* 0.1, sol.record.v.data[1, :]; color = Fathom.baikal)
hlines!(axv, [20.0]; color = Fathom.abyad, linestyle = :dash)                       # threshold
spikeraster!(Axis(fig[2, 1]; xlabel = "Time (ms)", ylabel = "Neuron", title = "Population raster"), sol)
psth!(Axis(fig[2, 2]; xlabel = "Time (ms)", ylabel = "Rate (per ms per neuron)", title = "Population rate"),
      sol; binwidth = 8.0, nneurons = 200)
addlabels!(fig)
fig
```

### Functional (live `f(t)`)

[`FunctionalCurrent`](@ref) evaluates a pure function at every step: `f(t)` (space-uniform) or `f(i, t)` (per
neuron `i` at simulation time `t`). It is stateless and GPU-safe whenever `f` is isbits (a bare function or a
closure over isbits data).

```julia
DewdropNetwork(LIF(), 100; input = 0.0, tspan = (0.0, 1.0),
               stimuli = FunctionalCurrent(t -> 0.3 + 0.2 * sin(2π * 5 * t)))
DewdropNetwork(LIF(), 100; input = 0.0, tspan = (0.0, 1.0),
               stimuli = FunctionalCurrent((i, t) -> 0.1 * i))          # per-neuron gradient
```

[`FunctionalKick`](@ref) applies `f` as a voltage kick (the deterministic analogue of a drive);
[`FunctionalConductance`](@ref) applies a prescribed conductance `g(t)` with a reversal potential.

Common analytic shapes build a `FunctionalCurrent` directly:

```julia
ramp(t1 = 0.5, to = 0.4)                    # linear 0 → 0.4 over [0, 0.5], flat after
step_input(amplitude = 0.3, t0 = 0.2)       # step to 0.3 at t = 0.2
sinusoid(amplitude = 0.2, freq = 5, offset = 0.3)
pulses(amplitude = 0.5, period = 0.1, width = 0.02)
```

!!! note "Functional vs. tabulated"
    Use a `FunctionalCurrent` for closed-form / analytic signals (no memory, evaluated live). For recorded or
    precomputed data, use a [`TimedArray`](@ref) --- it reads a device array by step index rather than
    calling a function.

### Tabulated (`TimedArray`)

[`TimedArray`](@ref) reads a precomputed signal by step index: a length-`nsteps` vector (shared) or an
`N × nsteps` matrix (per neuron). Apply it as a current (default) or a kick with `as =`.

```julia
signal = [0.3 + 0.2sin(0.05k) for k in 0:(nsteps - 1)]
DewdropNetwork(LIF(), 100; input = 0.0, tspan = tspan, stimuli = TimedArray(signal))
```

With `TimeseriesBase` loaded, a regularly-sampled series (whose `samplingperiod` equals the run `dt`) can be
passed directly:

```julia
using TimeseriesBase
ts = ToolsArray(signal, (𝑡(0:dt:(dt * (nsteps - 1))),))
DewdropNetwork(LIF(), 100; input = 0.0, tspan = tspan, stimuli = TimedArray(ts))
```

## Inhomogeneous Poisson

[`InhomogeneousPoisson`](@ref) generalises [`PoissonDrive`](@ref) to a rate that varies over space and/or
time: a scalar, a per-neuron vector `rate[i]`, an `N × nsteps` matrix `rate[i, n]`, or a live function
`rate(t)` / `rate(i, t)`. A per-neuron vector that is zero outside a subpopulation targets the drive to that
subpopulation.

```julia
rate = zeros(N); rate[1:100] .= 50.0                                   # drive only the first 100 neurons
DewdropNetwork(LIF(), N; input = 0.0, tspan = tspan,
               stimuli = InhomogeneousPoisson(rate; weight = 0.1, seed = 7))
DewdropNetwork(LIF(), N; input = 0.0, tspan = tspan,
               stimuli = InhomogeneousPoisson(t -> 20 + 80t; weight = 0.1, seed = 8))  # ramping rate
```

Where the `sinusoid` above drives every cycle deterministically, an `InhomogeneousPoisson` with a
rising rate drives the population *stochastically*: each neuron's spikes are random, but the population
rate (d) follows the rate profile (a).

```@example inputs
ratefn = t -> 1.5 + 0.06 * t                       # rate ramps up over the run
prob = DewdropNetwork(m, 200; input = 0.0, tspan = (0.0, 400.0),
                      stimuli = InhomogeneousPoisson(ratefn; weight = 0.25, seed = UInt64(11)))
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(), v = Trace(:V)))

fig = FourPanel()
tt = range(0, 400; length = 800)
lines!(Axis(fig[1, 1]; xlabel = "Time (ms)", ylabel = "Rate λ (a.u.)", title = "Poisson rate"),
       tt, ratefn.(tt); color = Fathom.qinghai)
axv = Axis(fig[1, 2]; xlabel = "Time (ms)", ylabel = "Membrane potential (mV)", title = "Single neuron")
lines!(axv, (1:size(sol.record.v.data, 2)) .* 0.1, sol.record.v.data[1, :]; color = Fathom.baikal)
hlines!(axv, [20.0]; color = Fathom.abyad, linestyle = :dash)
spikeraster!(Axis(fig[2, 1]; xlabel = "Time (ms)", ylabel = "Neuron", title = "Population raster"), sol)
psth!(Axis(fig[2, 2]; xlabel = "Time (ms)", ylabel = "Rate (per ms per neuron)", title = "Population rate"),
      sol; binwidth = 8.0, nneurons = 200)
addlabels!(fig)
fig
```

## Spike replay

[`SpikeSourceArray`](@ref) replays a fixed spike pattern (an `n_ext × nsteps` boolean matrix) through virtual
sources wired by an external connectome, delivering through any synapse model with per-edge delays --- the
deterministic sibling of [`PoissonSource`](@ref). It routes through the same scatter/delay pipeline as real
spikes, so the postsynaptic kinetics are exactly the wrapped synapse's.

```julia
extconn = Dewdrop.SparseCSR(arch, [(1, 5, 0.5, 1)]; npre = 1, npost = N)   # source 1 → neuron 5
spikes = fill(false, 1, nsteps); spikes[1, 100:100:end] .= true
DewdropNetwork(LIF(), N; input = 0.0, tspan = tspan,
    projections = (Projection(SpikeSourceArray(DeltaSynapse(), extconn, spikes),
                              Dewdrop._empty_csr(arch, N)),))
```

## Combining stimuli

`stimuli =` accepts one stimulus or an iterable of them; they compose with `input`/`drive`/`noise`:

```julia
DewdropNetwork(LIF(), N; input = fill(0.1, N), tspan = tspan,
               drive = PoissonDrive(rate = 20.0, weight = 0.1, seed = 3),
               stimuli = (ramp(t1 = 0.8, to = 0.2), TimedArray(signal)))
```

## Notes

- **Reproducibility.** The stochastic stimuli (drive, noise, inhomogeneous Poisson) draw from the counter-based
  RNG keyed by `(seed, step, neuron)`, so a run is bit-for-bit reproducible across threads, backends and
  devices. In a [batched](@ref "Batching & ensembles") run each column draws from its own stream (independent
  ensemble members); a shared stream reproduces the scalar run exactly.
- **GPU safety.** Array-backed stimuli (`TimedArray`, matrix/vector `InhomogeneousPoisson`, `SpikeSourceArray`)
  move to the run architecture automatically. A `FunctionalCurrent` is GPU-safe when its function is isbits;
  capture only isbits data (avoid closing over a host array --- use a `TimedArray` for tabulated data).
- **Validation.** Shapes are checked against `(N, nsteps)` at `init`, so a too-short `TimedArray` or a
  mis-sized rate matrix raises a clear error before the run.
