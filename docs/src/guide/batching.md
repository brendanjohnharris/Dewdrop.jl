```@meta
CurrentModule = Dewdrop
```

# Batching and ensembles

Running many networks at once amortises launch overhead and keeps the device busy. Dewdrop has two
batch modes, chosen by what varies across members:

- **Ensemble (tensor) batch** --- one network *shape*, `B` independent members on a trailing batch
  axis. Same connectome, varied input/seed/parameters. One fused pass over an `(N, B)` state.
- **Block-diagonal batch** --- `B` possibly *different* networks stacked into one larger network with
  no cross-member edges, solved in a single scalar pass.

The ensemble batch is memory-optimal (one shared connectome, `O(edges)`) but requires identical
connectivity *structure* and delays; the block-diagonal batch handles distinct topology, models, or
weights at the cost of `O(B·edges)` memory.

## Ensemble (tensor) batch

Pass `batch = B` to [`solve`](@ref) (or [`init`](@ref)). The per-neuron state gains a trailing
column per member; the connectome is broadcast read-only across all `B`. The result is a
[`BatchedSolution`](@ref) whose fields are `(N, B)`.

```julia
conn = fixed_prob(CPU(), 1000, 1000, 0.02; weight = 0.3, delay = steps(1), seed = 0x1)

prob = DewdropNetwork(LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 5.0), 1000;
                      input = 0.0, tspan = (0.0, 1000.0),
                      projections = (Projection(CurrentSynapse(; τ = 5.0), conn),),
                      drive = PoissonDrive(; rate = 20.0, weight = 0.5, seed = 0x1))

sol = solve(prob, FixedStep(0.1); batch = 64)

firing_rate(sol)        # (N, B) per-cell rate, one column per member
sol.spike_count         # (N, B) per-cell counts
sol.record              # NamedTuple of batched monitor results (B is the middle axis)
```

Each member draws its drive and SDE noise from an independent, bit-reproducible counter-based stream
(the Philox high counter word keys per member; see [reproducibility](#Reproducibility)). The batched
path is always the fused megakernel, so `backend` does not apply; the architecture (`CPU()`/`GPU()`
on the [`DewdropNetwork`](@ref)) still selects where state lives.

### Per-member variation

| keyword | shape | varies |
|---|---|---|
| `streams` | length-`B` `Int` vector (default `0:(B-1)`) | the per-member RNG stream index |
| `input` | scalar, `(N,)` vector, or `(N, B)` matrix | the external current per member |
| `v0` | scalar, `(lo, hi)`, `(N,)` vector, or `(N, B)` matrix | the initial voltage per member |
| `model_overrides` | `NamedTuple` of length-`B` (per member) or `(N, B)` (per neuron and member) arrays | neuron-model fields |
| `syn_overrides` | `NamedTuple` keyed by projection index `→ (; field = vec)` | scalar synapse params (dual-exp COBA family) |

A parameter sweep is just an override array. To sweep an external current over members:

```julia
inputs = reduce(hcat, [fill(I, prob.n) for I in range(0.0, 2.0; length = 64)])  # (N, B)
sol = solve(prob, FixedStep(0.1); batch = 64, input = inputs)
rates = vec(sum(firing_rate(sol); dims = 1)) ./ prob.n                          # mean rate per member
```

`model_overrides` reaches any neuron-model field. An `(N, B)` override sets a field per neuron *and*
member (e.g. an adaptation strength applied to one subpopulation only), a length-`B` override sets it
per member:

```julia
sol = solve(prob, FixedStep(0.1); batch = 8,
            model_overrides = (; τ = collect(range(10.0, 30.0; length = 8))))
```

`(lo, hi)` initial voltages draw an independent uniform per `(neuron, member)`, so the columns start
from distinct conditions --- useful for noise/realisation ensembles.

### Limitations

The shared-connectome design rejects what cannot be tensorised over a single CSR:

- connectivity *structure* and per-synapse *delays* must be identical across the batch;
- [`MultiModel`](@ref) (several model groups) is not supported --- use the block-diagonal batch;
- [`STDP`](@ref) (plastic projections) is not supported --- run `B` sequential solves, or the block
  batch.

A single-group [`Heterogeneous`](@ref) model *is* supported (it resolves per neuron through the same
megakernel seam).

## Reproducibility

Member `b` draws from stream `streams[b]`. The default `streams = 0:(B-1)` means the first column uses
stream `0`, which reproduces the scalar (non-batched) path bit-for-bit: a member with stream `0` and a
given input/`v0` equals a scalar [`solve`](@ref) with that same input/`v0`. The other streams are
collision-free and bit-reproducible regardless of architecture or thread count, because the draw is a
pure function of `(seed, step, entity, stream)`. On the CPU the batch axis is a contention-free
parallelisation axis (each column writes a disjoint ring slab, no atomics), so the ensemble batch is
deterministic at any thread count.

## Block-diagonal batch

[`batch`](@ref) builds a [`NetworkBatch`](@ref) from members that may differ in model, weights,
delays, or topology. Solving stacks them into one block-diagonal [`DewdropNetwork`](@ref) and returns
a [`BatchSolution`](@ref); per-member results are addressed by index.

```julia
nb = batch([net1, net2, net3])              # explicit members (networks or specs)
nb = batch(base; gK = [4.0, 6.0, 8.0])      # parameter sweep over a base model/spec/network
nb = batch(base; gK = gs, τw = ts, cartesian = true)  # Cartesian product of the sweeps
nb = batch((b, i) -> perturb(b, i), base; n = 16)     # generator: member i is f(base, i)

nmembers(nb)                                 # B

bs = solve(nb, FixedStep(0.1); tspan = (0.0, 1000.0))
bs[2]                 # member 2's (N₂,) spike counts
firing_rate(bs)       # Vector of per-member per-cell rates
firing_rate(bs, 2)    # member 2's rates
```

`solve(::NetworkBatch, …; mode = :auto)` routes by what varies (force a mode with `mode = …`):

| mode | when | how |
|---|---|---|
| `:shared` | same model + connectome, vary input | the fused `(N, B)` ensemble above |
| `:multirun` | shared connectome, per-member model | `B` scalar solves sharing the connectome array (threaded over members) |
| `:fused` | shared connectome, uniform model *type*, per-member params | one fused `(N, B)` launch with per-member overrides |
| `:block` | distinct topology | block-diagonal stack, one scalar solve |

The realised mode is `bs.mode`; the underlying solution(s) are in `bs.raw` for full access (e.g. the
block solution exposes `memberB` subpopulations via `bs.raw[:member2]`).

## When to use which

- **Parameter sweeps** over a fixed connectome: ensemble batch (or `batch(base; param = values)`).
- **Simulation-based inference**: ensemble batch over a `(N, B)` parameter grid, one solve per call.
- **Noise / initial-condition realisations**: ensemble batch with default `streams` (independent per
  member) and a `(lo, hi)` `v0`.
- **Heterogeneous topology / plastic / multi-model members**: block-diagonal [`batch`](@ref).

See also [building networks](networks.md) and [choosing a backend](backends.md).
