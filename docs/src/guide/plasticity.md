```@meta
CurrentModule = Dewdrop
```

# Synaptic plasticity (STDP)

Plasticity in Dewdrop is *event-driven*: weights change only on spikes, and the between-spike
trace decay is folded analytically into the existing synaptic phases. There is no extra schedule
phase and no continuous integration of the learning variables; a static network pays nothing.

The one rule is [`STDP`](@ref): pair-based, additive spike-timing-dependent plasticity. It is
*orthogonal to transmission*: an STDP rule wraps any base synapse ([`CurrentSynapse`](@ref),
[`ConductanceSynapse`](@ref), [`DeltaSynapse`](@ref), ...), so the kinetics that deliver a spike and
the rule that adjusts its weight are chosen independently.

## The rule

```julia
STDP(; Aplus, Aminus, τplus, τminus, wmin = -Inf, wmax = Inf)
```

Two per-neuron eligibility traces decay exponentially between spikes: a presynaptic trace `x_pre`
(time constant `τplus`) and a postsynaptic trace `x_post` (`τminus`). On a spike pairing,

- a *post-after-pre* event potentiates the edge by `Aplus·x_pre[pre]`;
- a *pre-after-post* event depresses it by `Aminus·x_post[post]`.

Because the traces decay at `τplus`/`τminus`, the effective increments are `Aplus·exp(-Δt/τplus)`
and `Aminus·exp(-Δt/τminus)` in the spike-time difference `Δt`. Every weight is clamped to
`[wmin, wmax]` after each update. `τplus`/`τminus` accept `Unitful` times; the bounds default to
unbounded.

| parameter | meaning |
|---|---|
| `Aplus` | potentiation amplitude (post-after-pre) |
| `Aminus` | depression amplitude (pre-after-post) |
| `τplus` | presynaptic trace time constant |
| `τminus` | postsynaptic trace time constant |
| `wmin`, `wmax` | weight clamp bounds (default `-Inf`, `Inf`) |

## Plastic state

A plastic projection carries *mutable* state that a static one does not. The shared
[`SparseCSR`](@ref) connectome stays immutable (so the same wiring is still shareable, and the
batched one-CSR-across-members broadcast is never violated); the rule allocates its own per-edge
weight array, initialised from the connectivity's weights, plus the two per-neuron traces. These
live on the network's architecture (`arch`), so on a [`GPU`](@ref) the weights and traces are device
arrays.

The weight update rides the *edge scatter*: one thread owns one synapse, so the per-edge weight write
needs no atomic. This is why **a plastic projection requires `scatter = :edge`**. The compacted
scatter walks only the out-edges of neurons that actually spiked, so it never visits an edge on the
step its *postsynaptic* neuron fires; it cannot drive the potentiation branch. The requirement is
enforced at [`init`](@ref): requesting `scatter = :compacted` with any plastic projection throws. The
default `scatter = :auto` already resolves to `:edge` whenever a plastic projection is present, so you
normally need not set it.

STDP also needs a *recurrent* projection (`npre == npost`); this too is checked at `init`.

## Attaching a rule

With the fluent [builder](networks.md), pass `plasticity =` to [`project!`](@ref):

```julia
nb = network(; arch = CPU(), tspan = (0.0, 1000.0))
population!(nb, :E, LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 2.0), 800)
project!(nb, :E => :E, ConductanceSynapse(; τ = 5.0, Erev = 0.0);
         p = 0.1, weight = 0.01, delay = 1.5,
         plasticity = STDP(Aplus = 0.005, Aminus = 0.006, τplus = 20.0, τminus = 20.0,
                           wmin = 0.0, wmax = 0.05))
prob = build(nb)
```

The plastic projection here is recurrent (`:E => :E`) and resolves to `scatter = :edge` automatically;
`tspan` is fixed at [`network`](@ref) (there is no `tspan` on [`build`](@ref)).

At the lower level, the same keyword sits on [`Projection`](@ref):

```julia
conn = fixed_prob(CPU(), 800, 800, 0.1; weight = 0.01, delay = steps(15), seed = 0x1234 % UInt64)
proj = Projection(ConductanceSynapse(; τ = 5.0, Erev = 0.0), conn;
                  plasticity = STDP(Aplus = 0.005, Aminus = 0.006,
                                    τplus = 20.0, τminus = 20.0, wmin = 0.0, wmax = 0.05))
prob = DewdropNetwork(LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 2.0), 800;
                      input = 0.0, tspan = (0.0, 1000.0), projection = proj)
```

## Reading the learned weights

The [`DewdropSolution`](@ref) returned by [`solve`](@ref) holds the final state and the recorded
monitors, but not the synaptic state. To inspect the weights after learning, keep the integrator:
run [`init`](@ref) then [`solve!`](@ref), and read the projection's state from `integ.syns`. For a
plastic projection that state is the plastic wrapper, whose `weight` field is the per-edge weight
array (in the projection's CSR edge order).

```julia
integ = init(prob, FixedStep(0.1); backend = Serial())
solve!(integ)

w = integ.syns[1].weight        # learned per-edge weights (device array on a GPU; Array(w) to copy)
x_pre = integ.syns[1].x_pre     # final presynaptic eligibility traces
```

`integ.syns` is a tuple ordered as the projections were added (a streaming [`drive!`](@ref) appends a
projection too, so count accordingly). The base synapse state is reachable as `integ.syns[1].base` if
you need the transmission variables. `weight`, `x_pre`, and `x_post` are internal fields of the
plastic state struct, not part of the exported API, but they are stable and safe to read.

## Backends and reproducibility

Plasticity works on every architecture and execution [backend](backends.md). The trace decay folds
into the per-step synaptic update (the same code on the [`Serial`](@ref) per-phase path and the
[`Fused`](@ref) tight loop / GPU megakernel), and the weight update is the edge scatter, so the
decay → update → bump ordering falls out of the existing phase order. On the CPU the scatter walks
edges in a fixed order with no atomics, so the learned weights are bit-reproducible; on the device the
edge kernel gives one owner per edge, again atomic-free for the weight write (only the ring deposit
keeps its atomic). The [`Turbo`](@ref) path vectorises only the dense membrane step; its scatter and
trace decay stay scalar, so plasticity works there too (subject to `Turbo`'s usual model-support and
`scatter = :edge` requirements).
