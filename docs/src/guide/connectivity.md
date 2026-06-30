```@meta
CurrentModule = Dewdrop
```

# Connectivity and spatial networks

A connectome maps presynaptic spikes onto postsynaptic targets. Dewdrop stores it as a single
sparse type, [`SparseCSR`](@ref), and gives you constructors that build one from a rule: a fixed
edge probability, or a distance kernel over neuron positions. Each edge carries its own weight and
its own conduction delay, so heterogeneous wiring costs the same per-step traffic as a uniform one.

Connectivity is usually reached through `project!(...)` in the [builder](networks.md); the
constructors below are what `project!` calls, and you can build a [`SparseCSR`](@ref) directly when
you need one outside a network spec.

## The storage: `SparseCSR`

[`SparseCSR`](@ref) is compressed-sparse-row over *presynaptic* neurons: the out-edges of neuron `i`
occupy a contiguous slice, so the event-driven scatter touches only a spiking neuron's own row and
never materialises a dense `[post × pre]` matrix. Parallel arrays hold, per edge, the postsynaptic
target, the weight, and the delay. You rarely build it by hand; the rules below return one.

| field | meaning |
|---|---|
| weight | per-synapse weight (sign sets excitatory/inhibitory) |
| delay | per-synapse conduction delay; ms (Float) or integer steps (Int) until resolved |

(The other fields---`rowptr`, `post`, `src`, `maxdeg`, `npre`, `npost`---are bookkeeping for the
CSR layout.) `index_type` is not a stored field but a constructor keyword: passing `index_type =
Int32` narrows the `rowptr`/`post`/`delay` (step-delay) index arrays, halving the bandwidth of the
bandwidth-bound scatter; safe whenever `nedges < 2^31` (`Int` is the default).

## Random connectivity: `fixed_prob`

[`fixed_prob`](@ref) is Erdos--Renyi: each ordered `(pre, post)` pair is an edge with probability
`p`, sampled reproducibly from the counter-based RNG so a given `seed` yields a fixed, copyable
connectome.

```julia
conn = fixed_prob(CPU(), 1000, 1000, 0.02;
                  weight = 0.5, delay = 1.5, seed = 0x1234,
                  allow_self = false)
```

`weight` and `delay` may be scalars or functions of the presynaptic index `pre`. A per-source
function is how you give an excitatory/inhibitory split signed weights:

```julia
NE = 800
w(pre) = pre ≤ NE ? 0.4 : -1.8          # first NE neurons excitatory, rest inhibitory
conn = fixed_prob(CPU(), 1000, 1000, 0.05; weight = w, delay = 1.0, seed = 0x1)
```

`sources` and `targets` restrict the presynaptic and postsynaptic sets (contiguous global-index
ranges), which is how a named-subpopulation projection such as `:E => :I` is wired; flat `pre`/`post`
indices stay absolute. `allow_self = false` drops self-edges.

## Positions and kernels

Spatial connectivity is positions plus a kernel. A position layout returns a vector of coordinate
tuples; a kernel maps distance to a probability in `[0, 1]`.

Layouts:

| function | geometry |
|---|---|
| [`line_positions`](@ref) | `n` points evenly spaced on a line |
| [`grid_positions`](@ref) | `nx × ny` rectangular grid (`centered = true` for a periodic box) |
| [`ring_positions`](@ref) | `n` points on a circle (wraparound built into the geometry) |
| [`random_positions`](@ref) | `N` uniform-random points in a box, reproducible from `seed` |

Kernels (each returns a `distance -> probability` closure):

| function | shape |
|---|---|
| [`gaussian_kernel`](@ref) | `pmax·exp(-d²/2σ²)` |
| [`exponential_kernel`](@ref) | `pmax·exp(-d/λ)` |
| [`box_kernel`](@ref) | `p` within radius `r`, else 0 (local neighbourhood) |

## Distance-dependent connectivity

[`distance_prob`](@ref) connects each ordered pair with probability `kernel(distance(pre, post))`,
seed-reproducible. The edge count is random (per-pair Bernoulli).

```julia
pos  = grid_positions(40, 40; spacing = 1.0, centered = true)
conn = distance_prob(CPU(), pos;
                     kernel = gaussian_kernel(3.0; pmax = 0.5),
                     weight = 0.2, delay = 1.0, seed = 0x7,
                     period = (40.0, 40.0))          # periodic box
```

`period` (a tuple of per-dimension side lengths) enables periodic boundaries via the minimum-image
convention; pair it with `centered = true` from [`grid_positions`](@ref). [`ring_positions`](@ref)
already encodes its wraparound, so a ring needs no `period`.

[`distance_fixed_count`](@ref) instead samples an *exact* total `count` of edges without replacement,
each chosen with probability proportional to `kernel(distance)`, via a Gumbel-max top-k draw. Use it
for fixed-degree wiring, where a per-pair Bernoulli's random count would not do.

```julia
conn = distance_fixed_count(CPU(), pos;
                            kernel = exponential_kernel(2.0),
                            count = 8000,
                            weight = 0.2, delay = 1.0, seed = 0x7)
```

Both take the same `weight`/`delay` (scalar or per-source function), `sources`/`targets`,
`allow_self` and `index_type` keywords as [`fixed_prob`](@ref); pairs with zero kernel probability
are never selected.

## Delays: milliseconds versus steps

A delay is a physical time by default---the same units as `dt` and every other quantity---so its
meaning is independent of the solve step. A bare number is milliseconds, resolved to an integer step
count at `init` once `dt` is known (`round(ms / dt)`, clamped to at least one step). For an exact
step count, wrap it with [`steps`](@ref):

```julia
delay = 1.5          # 1.5 ms; becomes 15 steps at dt = 0.1, 3 steps at dt = 0.5
delay = steps(5)     # exactly 5 steps regardless of dt
```

[`steps`](@ref) requires `n ≥ 1`: the fixed-step engine delivers no earlier than the next step, so it
cannot represent a within-step delay. Internally the connectome holds the delay as given (Float ms or
Int steps) and the hot-path scatter always reads resolved integer steps; the delivery uses a ring
buffer, so arbitrarily distinct per-synapse delays cost the same O(1) delivery as a single global one.

## Adjusting weights after building

[`correlate_weights!`](@ref) rescales a connectome's weights in place to in-degree-normalised values,
the standard balanced-network `1/√k` scaling: each edge into a target with in-degree `k` gets mean
weight `≈ J/√k` plus a reproducible relative Gaussian `jitter`. `targets` selects the post
sub-population the in-degrees are counted over, so a sub-population projection normalises against the
right set.

```julia
correlate_weights!(conn, 0.05; seed = 0xABCD, jitter = 0.05)
```

[`correlate_weights`](@ref) is the curried form for the builder's `adjust` hook, which supplies the
projection's resolved destination range automatically:

```julia
project!(net, :E => :I; ..., adjust = correlate_weights(0.05; seed = 0xABCD))
```

`count_empty` selects the zero-in-degree convention (default `true` reproduces BrainPy's `√max(k,1)`;
`false` excludes isolated targets, the more principled form). The two agree when every target is
wired.
