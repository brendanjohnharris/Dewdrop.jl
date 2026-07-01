```@raw html
---
layout: home

hero:
  name: "Dewdrop.jl"
  tagline: "A CPU/GPU spiking circuit simulator for Julia"
  actions:
    - theme: brand
      text: Get started
      link: /guide/getting-started
    - theme: alt
      text: Benchmarks
      link: /benchmarks
    - theme: alt
      text: View on GitHub
      link: https://github.com/brendanjohnharris/Dewdrop.jl

features:
  - title: Swap between CPU and GPU
    details: KernelAbstractions kernels run on the CPU and CUDA; the architecture chooses where, the backend chooses how.
  - title: Pluggable execution backends
    details: Serial (bit-reproducible), Fused (tight loop, near-C++ on CPU, the GPU megakernel), and Turbo (SIMD, opt-in).
  - title: Flexible models
    details: LIF, AdEx, conductance-adaptation FNS, per-neuron Heterogeneous, multi-type MultiModel, custom @neuron models.
---
```

## Highlights

Dewdrop is a fixed-step, clock-driven, struct-of-arrays spiking neural network engine. It is built
around two orthogonal choices:

- **Architecture** (`arch = CPU()` / `GPU()`) --- *where* the state lives.
- **[Backend](guide/backends.md)** (`backend = Auto()` / `Serial()` / `Fused()` / `Turbo()`) --- *how*
  each step executes.

```julia
using Dewdrop

m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
prob = DewdropNetwork(m, 10_000; input = 1.5, tspan = (0.0, 1000.0))
sol = solve(prob, FixedStep(0.1))            # backend = Auto() picks the best execution path
```

`solve` shows a **live progress bar** by default (`progress = :auto`): it appears once a run has
gone on long enough to matter (calibrating its update rate from the first ~0.3 s) and renders
natively in VSCode, Pluto and any `TerminalLogger` REPL, while staying silent in plain scripts. Pass
`progress = false` to silence it, `progress = true` to force it on from the start, a `String` to name
the bar, or an `Int` to set a fixed update stride. It is a standard
[ProgressLogging](https://github.com/JuliaLogging/ProgressLogging.jl) log record --- so it needs no
extra dependency, and it never perturbs the simulation (a `progress` run is bit-identical to none).

See **[Choosing a backend](guide/backends.md)** for when to use each, and **[Turbo & model
specialization](guide/turbo.md)** for the SIMD backend and how to add a Turbo specialization to a
model.

## Performance

On an identical recurrent E/I [`AdEx`](@ref) network, Dewdrop's [`Fused`](@ref) backend is the fastest
CPU simulator measured at scale (3.4× Brian2's compiled C++, 15.6× NEST at N=512k) and the fastest on
the GPU (2.2× brian2cuda, 8× GeNN), while staying competitive with both at small sizes. See the full
cross-simulator comparison in **[Benchmarks](benchmarks.md)**.

![Simulation time and peak memory vs network size, across simulators.](assets/comparison.png)
