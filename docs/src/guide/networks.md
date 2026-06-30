```@meta
CurrentModule = Dewdrop
```

# Building networks

A Dewdrop simulation is a [`DewdropNetwork`](@ref): a flat struct-of-arrays population plus a tuple
of synaptic [`Projection`](@ref)s. You rarely construct one by hand. The fluent builder accumulates
named populations, projections and an external drive, then [`build`](@ref) assembles them into a
single concretely-typed network. This page covers the builder, the named-subpopulation addressor, the
automatic model merge, and the deferred network spec.

## The fluent builder

Begin with [`network`](@ref), add populations with [`population!`](@ref), wire them with
[`project!`](@ref), optionally attach an external [`drive!`](@ref), then [`build`](@ref):

```julia
nb = network(; arch = CPU(), tspan = (0.0, 1000.0))
adex = AdEx(; C = 281.0, gL = 30.0, EL = -70.6, VT = -50.4, ΔT = 2.0, Vr = -70.6,
            Vpeak = -40.0, a = 4.0, b = 0.0805, τw = 144.0)
population!(nb, :E, adex, 8000)
population!(nb, :I, adex, 2000)
project!(nb, :E => :I, ConductanceSynapse(τ = 5.0, Erev = 0.0); p = 0.02, weight = 0.3, delay = 1.5)
project!(nb, :I => :E, ConductanceSynapse(τ = 5.0, Erev = -80.0); p = 0.02, weight = 1.2, delay = 1.5)
prob = build(nb)
sol = solve(prob, FixedStep(0.1))
```

The builder is mutating; each verb returns the builder, so calls can also be chained. `arch` (where
the state lives, [`CPU`](@ref) / [`GPU`](@ref)) and `tspan` are fixed when the builder is created; the
execution backend is chosen later, at [`solve`](@ref) (see [choosing a backend](backends.md)).

For the common excitatory/inhibitory case, the three-argument [`network`](@ref) is sugar that adds an
`:E` population of `NE` and an `:I` population of `NI`, both of the same model:

```julia
nb = network(AdEx(; C = 281.0, gL = 30.0, EL = -70.6, VT = -50.4, ΔT = 2.0, Vr = -70.6,
                   Vpeak = -40.0, a = 4.0, b = 0.0805, τw = 144.0), 8000, 2000; tspan = (0.0, 1000.0))
```

### Populations

[`population!`](@ref) takes a `name`, a neuron `model` (see [neuron models](models.md)), and a count
`N`. `input` is a per-population constant current (a scalar, or a length-`N` vector); `positions`
(optional) are consumed by distance-kernel projections (see [spatial networks](wrcircuit.md)). The
name `:all` is reserved --- it always denotes the whole network.

```julia
lif = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
population!(nb, :E, lif, 4000; input = 0.2)
population!(nb, :I, lif, 1000; positions = random_positions(1000, (1.0, 1.0); seed = 0x01))
```

### Projections

[`project!`](@ref) adds a projection from `src` onto `dst`, given as a `src => dst` pair of
subpopulation names, over a synapse model. The single-symbol form `project!(nb, src, synapse; …)`
targets the whole network (`src => :all`). Connectivity is chosen by which keyword you pass:

| keyword | connectivity | notes |
|---|---|---|
| `p` | [`fixed_prob`](@ref) | each source--target pair connected with probability `p` |
| `kernel` | [`distance_prob`](@ref) | distance-dependent probability; needs population `positions` |
| `kernel` + `count` | [`distance_fixed_count`](@ref) | exactly `count` distance-weighted targets per source |
| `connectivity` | prebuilt | pass a [`SparseCSR`](@ref) directly |

`weight` and `delay` are scalars or per-source functions of the presynaptic index. `delay` is a
physical time in milliseconds (resolved to integer steps at the solve `dt`); use [`steps`](@ref) for
an explicit step count. Other keywords: `seed` keys the random wiring, `allow_self` permits
self-connections, `plasticity` attaches an [`STDP`](@ref) rule, and `adjust = conn -> …` runs a hook
over the materialised connectivity (e.g. [`correlate_weights`](@ref) to normalise in-degree):

```julia
project!(nb, :E => :E, ConductanceSynapse(τ = 5.0, Erev = 0.0);
    p = 0.02, weight = 0.3, delay = 1.5, seed = 0x01, adjust = correlate_weights(0.3; seed = 0x01))
```

### External drive

[`drive!`](@ref) attaches an external input. With a [`PoissonDrive`](@ref) it sets a per-neuron
background drive (every neuron receives independent external spikes each step):

```julia
drive!(nb, PoissonDrive(rate = 10.0, weight = 0.1))
```

The synapse form wires `n_ext` virtual Poisson sources (rate `rate`) onto a `target` subpopulation by
[`fixed_prob`](@ref), delivering through any synapse model (the postsynaptic kinetics). This is a
streaming drive --- no precomputed event matrix:

```julia
drive!(nb, :E, ConductanceSynapse(τ = 5.0, Erev = 0.0); rate = 8.0, n_ext = 1000, p = 0.02, weight = 0.4)
```

Give two synapse drives the same `fire_seed` (different `seed`) to make them a shared common-mode
source: the same external spikes, independent fan-out.

### Assembling

[`build`](@ref) concatenates the populations into one flat SoA (recording each subpopulation's range),
merges the models and inputs, and materialises the projections. `input` overrides every
per-population input with a single global value; `schedule` sets the within-step phase order.

## Named subpopulations and the addressor

Each population's name addresses it both in [`project!`](@ref) and on the solution. The registry
always carries an implicit `:all` spanning the whole network. On a [`DewdropSolution`](@ref):

- `sol[:E]` returns a [`SubSolution`](@ref): a lightweight view (no copy) over the subpopulation's
  range, carrying its state and spike counts.
- `firing_rate(sol, :E)` gives the per-unit firing rate of that subpopulation (see
  [`firing_rate`](@ref)).
- `raster(sol; of = :E)` restricts recorded spike events to the subpopulation and rebases the neuron
  ids into `1:|E|` (see [`raster`](@ref); needs a `Spikes()` monitor).

```julia
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),))
re = firing_rate(sol, :E)        # excitatory rates
te, ie = raster(sol; of = :E)    # excitatory spike times and (rebased) ids
```

## Automatic model merge

[`build`](@ref) merges the per-group models into one engine model, choosing the cheapest representation
that fits (see [neuron models](models.md)):

| populations | merged model | path |
|---|---|---|
| one shared model (or several identical) | the bare model | homogeneous fast path |
| same type, differing parameter values | [`Heterogeneous`](@ref) | per-neuron override arrays |
| different model types | [`MultiModel`](@ref) | per-group launches over a union SoA |

So an `:E`/`:I` pair built from one shared model stays the homogeneous fast path (byte-identical to a
hand-built [`DewdropNetwork`](@ref)); giving `:E` and `:I` different parameters collapses them to a
[`Heterogeneous`](@ref); giving them different model types ([`AdEx`](@ref) vs [`LIF`](@ref)) produces a
[`MultiModel`](@ref). The merge is automatic --- you build with whatever models you declared.

## Deferred network specs

A builder can be frozen into a reusable, run-parameter-free spec that materialises into a network only
at solve time, when `dt`/`tspan` are known. [`freeze`](@ref) snapshots a builder into an immutable
[`AbstractNetworkSpec`](@ref); [`solve`](@ref) and [`init`](@ref) accept the spec directly:

```julia
spec = freeze(nb)
sol  = solve(spec, FixedStep(0.1))                 # uses the builder's default tspan
sol2 = solve(spec, FixedStep(0.1); tspan = (0.0, 5000.0))   # override per solve
```

[`defer`](@ref) captures an arbitrary constructor `f` and its kwargs as a spec, materialised by calling
`f(; kw…, tspan, dt)` --- for constructors whose assembly genuinely needs `dt` (e.g. a streaming
spatial drive). Put `seed`/`arch` in the captured kwargs; do not put `tspan`/`dt` there (they are
injected at materialise time). [`materialize`](@ref) is the explicit `(spec, run-params) -> network`
seam if you want to inspect the built network without solving.

## The low-level constructor

For a single homogeneous population the builder is overkill; construct a [`DewdropNetwork`](@ref)
directly:

```julia
prob = DewdropNetwork(LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0),
                      1000; input = 0.3, tspan = (0.0, 1000.0))
sol  = solve(prob, FixedStep(0.1))
```

Pass a recurrent [`Projection`](@ref) via `projection = …` (or several via `projections = (…,)`). The
builder is simply the multi-population, named-subpopulation front-end to this same type.
