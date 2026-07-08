```@meta
CurrentModule = Dewdrop
```

# Choosing a backend

Dewdrop separates two orthogonal choices:

- **Architecture** (`arch = CPU()` / `GPU()`, set on the [`DewdropNetwork`](@ref)): *where* the
  state lives.
- **Execution backend** (`backend = …`, passed to `solve`/`init`): *how* each step runs.

```julia
solve(prob, FixedStep(0.1); backend = Fused())
```

All backends compute the **same dynamics**; they differ in speed and in whether the result is
bit-reproducible. The default, `Auto`, picks a good one for you.

## The backends at a glance

| backend | what it does | bit-reproducible? | needs | use when |
|---|---|---|---|---|
| [`Auto`](@ref) | picks the best available (see below) | inherits the chosen one | none | the default; almost always |
| [`Serial`](@ref) | per-phase broadcasts, single-threaded dense phases | **yes**, allocation-free | none | small nets, exact reproducibility, debugging |
| [`Fused`](@ref) | one fused pass/step: a tight threaded CPU loop, or the GPU megakernel | **yes** (bit-identical to `Serial`) | none | large CPU nets; **required** on GPU and for `Heterogeneous`/`MultiModel` |
| [`Turbo`](@ref) | SIMD-vectorised dense step | no (spike-identical) | `using LoopVectorization` + a model with a Turbo specialization | maximum CPU throughput on supported, dense-dominated models |

## How `Auto` chooses

[`Auto`](@ref) (the default) resolves to:

- `Fused` on a **GPU** (the megakernel is the GPU path);
- `Fused` for a **`Heterogeneous`/`MultiModel`** model (per-neuron / per-group resolution needs it);
- `Fused` for a **large multithreaded CPU** network (`N ≥ 10_000`, `Threads.nthreads() > 1`, canonical
  schedule): the threaded fused step is ~2× the per-phase `Serial` baseline and bit-identical;
- `Serial` otherwise (small / single-threaded CPU nets, where the per-phase path's low overhead wins).

`Turbo` is never chosen automatically (it is not bit-identical); request it explicitly.

## Performance

Measured on the recurrent E/I AdEx benchmark (`test/brian/`), N = 16000, 5000 steps, Float32, vs
Brian2's compiled-C++ standalone (CPU×1 = 0.347 s):

| backend | N = 16000 | note |
|---|--:|---|
| `Serial` (1 thread) | ~2.0 s | the multi-pass baseline |
| `Fused` (16 threads) | **~0.38 s** | bit-identical; ≈ Brian2's compiled C++ (1.1×) |
| `Turbo` (SIMD) | ~0.55 s (1 thread) | spike-identical; wins big on dense-dominated nets |
| Dewdrop `Fused` GPU | ~0.09 s | 1.7× faster than brian2cuda |

The `Fused` backend brings Dewdrop's CPU throughput to near-parity with hand-tuned compiled C++ with
no dependency and **no change in results**; `Turbo` trades the `exp` ULP for SIMD and is fastest where
the per-neuron compute dominates the sparse scatter.

## Reproducibility

- `Serial` and `Fused` are **bit-identical** to each other at any fixed thread count (the dense update
  is per-neuron-independent, so even threaded it is order-free). The one order-dependent operation is
  the *threaded* atomic synaptic scatter: identical across backends, but not bit-reproducible across
  thread counts (run single-threaded for byte-exact reproducibility; the dynamics are statistically
  identical otherwise).
- `Turbo` is **spike-identical** but not bit-identical: SIMD `exp` (SLEEF) differs from scalar `libm`
  `exp` at the ULP level (`max|ΔV|` ~ 1e-5 to 1e-1 over a run).

## GPU spike scatter (`scatter = :auto`)

On the GPU the synaptic scatter has two strategies, selected automatically by `scatter = :auto` (the
default):

- **`:edge`**: one thread per synapse. Saturates the device at small/medium sizes, but reads a source
  index for *every* edge each step (Θ(nedges) memory traffic regardless of firing), so once the
  connectome's index arrays spill the L2 cache the step becomes bandwidth-bound and a very large network
  degrades superlinearly.
- **`:compacted`**: processes only the out-edges of neurons that actually spiked (work ∝ spikes, for the
  usual sparse firing), at the cost of one per-step device→host sync. Far faster on large networks, a
  small loss on small ones (the sync dominates when the step is launch-bound).

`:auto` switches from `:edge` to `:compacted` once the connectome's index footprint exceeds about half
the device L2 (the measured crossover), keeping the GPU step on the lower envelope at every size. On the
CPU the scatter already walks only spiking rows, so `:auto` is always `:edge` there. Force a mode with
`solve(...; scatter = :edge | :compacted)`; plastic (STDP) projections always use `:edge`.

## Guidance (the advisor)

The performance advisor emits a one-off `@info` hint when a run could use a faster backend (e.g. a
large CPU network that would benefit from `Turbo`). Silence it with `Dewdrop.set_advice!(false)` or
`solve(...; advise = false)`.

## Deprecated `step` keyword

The older `step = :auto | :serial | :fused | :turbo` keyword still works and maps onto the
corresponding backend; prefer `backend = …`.

```@docs
SimBackend
Auto
Serial
Fused
Turbo
```
