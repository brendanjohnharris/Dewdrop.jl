# Differentiable simulation & training

The [`Differentiable`](@ref) backend makes a CPU forward pass **automatically differentiable**, so the
gradient of a scalar loss flows back to the model parameters through the entire time loop. This enables
gradient-based *fitting* (match a model to data) and surrogate-gradient *training* (optimise parameters
like a neural network). It is opt-in and orthogonal to every other backend, which stay bit-identical.

## How it works

A spiking simulation has one non-differentiable operation: the discontinuous spike (a hard threshold
`V ≥ Vθ`, a jump reset, and an integer count have zero gradient almost everywhere). The `Differentiable`
backend replaces it with a smooth **fast-sigmoid surrogate**

```math
s = \frac{1}{1 + \exp(-\beta\,(V - V_\theta))} \in (0,1)
```

a soft reset `V ← V - s\,(V - V_r)`, and a real-valued spike accumulation. Everything else — the exact
subthreshold propagator, the synaptic kinetics, the counter-based RNG — is already smooth and
differentiates as-is. The gradients are *approximate* gradients of the true discrete dynamics (the
standard surrogate-gradient method), so it is a distinct numerical path you opt into; `β` sets the
steepness (larger → closer to the true Heaviside, but stiffer gradients).

The engine is eltype-generic by construction: the model's `float_type` flows to every state column, so a
`Dual`/`Active`-typed model gives a `Dual`/`Active`-typed run with no further change — autodiff "just
works." Pair it with any AD tool.

## Example: fit a firing rate to a target

```julia
using Dewdrop, ForwardDiff
using Dewdrop: Differentiable, duration

# an unconnected LIF population; the trainable parameter is the membrane resistance R
build(R) = DewdropNetwork(LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = R, tref = 0.0),
    8; input = 30.0, tspan = (0.0, 500.0))

rate(R) = let s = solve(build(R), FixedStep(0.1); backend = Differentiable(β = 25))
    sum(s.spike_count) / (8 * duration(s))          # mean rate, real-valued under the surrogate
end

# the gradient of the rate through the whole 5000-step simulation, via ForwardDiff:
g = ForwardDiff.derivative(rate, 1.0)               # matches finite differences to ~1e-10

# train R so the rate hits a target (plain gradient descent):
target = 0.018
loss(R) = (rate(R) - target)^2
R = 1.0
for _ in 1:30
    R -= 6.0 * ForwardDiff.derivative(loss, R)
end
```

Optimising the *surrogate* rate also moves the *true* (hard-spike) rate toward the target — the surrogate
gradient is a faithful descent direction, not just internally consistent.

`ForwardDiff` is ideal for a handful of parameters (it costs one pass per parameter). For many parameters
(e.g. weights), use reverse-mode `Enzyme`, which costs one pass regardless of parameter count.

## Choosing the AD tool

| Tool | Mode | Best for | Notes |
|------|------|----------|-------|
| `ForwardDiff` | forward | a few parameters (neuron params, global scales, fitting) | works out of the box on the eltype-generic engine |
| `Enzyme` | reverse | many parameters (synaptic weights, deep training) | ~hundreds × faster than forward for weight training; differentiates the mutating loop |
| `Zygote` | reverse | — | does **not** support the in-place mutating engine |

## Scope and expansion roadmap

The current backend is a deliberately small, correct foundation. It is **CPU-only**, **single-population**
(unconnected), and uses the canonical schedule; it errors clearly otherwise. The design leaves clean
extension points:

- **Synaptic-weight training** — the next step. A *surrogate-weighted* scatter (`accum += weight · s`,
  the spike seam in `_diff_step!`) makes recurrent networks trainable so you can fit/learn the connectome.
- **Reverse-mode at scale** — an `Enzyme` weak-dependency extension with a custom reverse rule for the
  atomic scatter, plus truncated-BPTT / checkpointing for long runs.
- **GPU** — `Enzyme` through the KernelAbstractions megakernel + atomic scatter (the per-neuron body is
  the ideal case for AD; the atomic-scatter adjoint and device memory are the open questions).
- **Straight-through forward** — an optional mode where the *forward* pass uses the true hard spike and
  only the *backward* pass uses the surrogate derivative (real spike trains, surrogate gradients).
- **Multi-state models** — already supported: the soft adaptation bump generalises `AdaptLIF`/`AdEx`'s
  `w` increment, so adapting neurons train too.
- **A `fit`/`train` API** and simulation-based inference (which needs no gradients — it rides the
  reproducible ensemble batch axis).

See the strategic roadmap for how these sequence behind the equation front-end and the device step counter.
