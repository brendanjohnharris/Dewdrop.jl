```@meta
CurrentModule = Dewdrop
```

# Case study: the WRCircuit

The WRCircuit is a spatial excitatory/inhibitory network of conductance-adaptation neurons with
dual-exponential conductance synapses on four distance-dependent recurrent paths and an external
Poisson drive --- a "working-regime" cortical circuit of the kind written in Brian or BrainPy. This
page shows how to express that style of model in Dewdrop using only the public API: the
conductance-adaptation neuron [`FNSNeuron`](@ref), the frozen-current synapse
[`FrozenDualExpSynapse`](@ref), spatial connectivity ([`random_positions`](@ref),
[`gaussian_kernel`](@ref), [`distance_fixed_count`](@ref)), and the fluent builder
([`network`](@ref) / [`population!`](@ref) / [`project!`](@ref) / [`drive!`](@ref)). No bespoke
circuit constructor is needed; the generic builder assembles it.

## Frozen-current vs exact COBA

A conductance synapse contributes a voltage-dependent current `g·(Erev − V)`. There are two ways to
integrate it over a step, and the choice lives in the *synapse*, not the neuron:

- [`DualExpSynapse`](@ref) is the exact-COBA propagator: the conductance `g` is folded into the
  membrane's effective leak `gtot` (it shunts the time constant), and the reversal drive `g·Erev`
  enters the input current. This is the more accurate scheme.
- [`FrozenDualExpSynapse`](@ref) has the *identical* dual-exponential kinetics
  `g(t) = a·(g_decay − g_rise)`, but injects `g·(Erev − V)` as an ordinary current with `V` frozen at
  its pre-update value, and adds nothing to `gtot` --- the conductance never shunts the leak. This
  reproduces the integration that Brian and BrainPy use for COBA synapses (their
  `sum_current_inputs`).

The two diverge once `g` is appreciable relative to the leak. You can see the distinction directly in
`_accumulate!` (`src/Engine.jl`): `DualExpCOBAState` writes both `gtot` and `itot`, while
`FrozenDualExpCOBAState` writes only `itot`, using the frozen `V`. For new models prefer
[`DualExpSynapse`](@ref); reach for [`FrozenDualExpSynapse`](@ref) only when you need to match a
frozen-current simulator. It is otherwise a drop-in replacement.

The neuron is unchanged either way. [`FNSNeuron`](@ref) carries its own adaptation conductance `gK`
(reversal `VK`), folded into the same exact COBA propagator; setting `ΔgK = 0` gives a plain
conductance-LIF (the inhibitory population), while `ΔgK > 0` makes the excitatory population adapt.

## A spatial FNS E/I network

The example below places E and I populations at random positions in the unit square, wires four
recurrent paths with a fixed total edge count drawn `∝ kernel(distance)`
([`distance_fixed_count`](@ref), the Gumbel-max top-k sampler), drives the whole network with a
streaming Poisson source, and records the E membrane traces and all spikes. Because `E` and `I` are
both [`FNSNeuron`](@ref) differing only in `ΔgK`, the builder merges them into a single per-neuron
[`Heterogeneous`](@ref) model automatically.

```julia
using Dewdrop

NE, NI = 1600, 400
domain = (1.0, 1.0)                                   # unit square
posE = random_positions(NE, domain; seed = 0x01 % UInt64)
posI = random_positions(NI, domain; seed = 0x02 % UInt64)

base = (; C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0, tref = 4.0, τK = 80.0)
E = FNSNeuron(; base..., ΔgK = 0.02)                  # excitatory: adapts
I = FNSNeuron(; base..., ΔgK = 0.0)                   # inhibitory: no adaptation

exc = FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)     # excitatory, frozen current
inh = FrozenDualExpSynapse(; τr = 2.0, τd = 4.5, Erev = -80.0)   # inhibitory, frozen current

nb = network(; tspan = (0.0, 1000.0))
population!(nb, :E, E, NE; positions = posE)
population!(nb, :I, I, NI; positions = posI)

# four recurrent paths; `kernel` + `count` routes to distance_fixed_count (fixed total edge count).
# `delay` is a physical time in ms, resolved to integer steps at the solve dt.
project!(nb, :E => :E, exc; kernel = gaussian_kernel(0.10), count = NE * 80, weight = 0.6, delay = 1.5, seed = 0x11 % UInt64)
project!(nb, :E => :I, exc; kernel = gaussian_kernel(0.10), count = NI * 80, weight = 0.6, delay = 1.5, seed = 0x12 % UInt64)
project!(nb, :I => :E, inh; kernel = gaussian_kernel(0.20), count = NE * 20, weight = 3.0, delay = 1.0, seed = 0x13 % UInt64)
project!(nb, :I => :I, inh; kernel = gaussian_kernel(0.20), count = NI * 20, weight = 3.0, delay = 1.0, seed = 0x14 % UInt64)

# external excitatory drive: 1000 virtual Poisson sources wired to every neuron by fixed_prob(p)
drive!(nb, :all, exc; rate = 3.0, n_ext = 1000, p = 0.02, weight = 0.6, delay = 1.0, seed = 0x21 % UInt64)

prob = build(nb)
sol = solve(prob, FixedStep(0.1); record = (V = Trace(:V; of = :E), spikes = Spikes()))

firing_rate(sol, :E)        # per-subpop observables; positions travel onto the solution as sol[:E].positions
```

See [neuron and synapse models](models.md) for the model zoo, [connectivity and spatial
networks](connectivity.md) for the kernels and connectome builders, and [building
networks](networks.md) for the builder verbs and the homogeneous/`Heterogeneous` merge.

## Matching another simulator

Choosing [`FrozenDualExpSynapse`](@ref) matches the other simulator's *integration scheme*, but a
bit-for-bit comparison needs the same *inputs* too: the connectome, per-edge weights, initial
membrane potentials, and the external spike train. These come from the reference simulator's own RNG,
so an independent run cannot regenerate them --- reproducing a specific run means exporting that
structure and ingesting it (e.g. building each [`Projection`](@ref) from a prebuilt connectivity and
passing `v0`), then verifying only the deterministic integration. Mind also the delay/onset
conventions: simulators differ in when a delivered spike first affects the postsynaptic state
relative to the nominal delay, so an alignment of one step may be required when comparing rasters.
None of this changes how you build a *new* model --- use the intended delays and let Dewdrop draw the
connectome and drive.
