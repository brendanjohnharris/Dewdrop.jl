```@raw html
---
layout: home

hero:
  name: "Dewdrop.jl"
  text: "A GPU-aware spiking neural network simulator"
  tagline: "Fixed-step, struct-of-arrays, CPU & GPU from one source."
  actions:
    - theme: brand
      text: Choosing a backend
      link: /guide/backends
    - theme: alt
      text: View on GitHub
      link: https://github.com/brendanjohnharris/Dewdrop.jl

features:
  - icon: 🌱
    title: One source, CPU & GPU
    details: KernelAbstractions kernels run on the CPU and CUDA; the architecture chooses where, the backend chooses how.
  - icon: 🚀
    title: Pluggable execution backends
    details: Serial (bit-reproducible), Fused (tight loop, near-C++ on CPU, the GPU megakernel), and Turbo (SIMD, opt-in).
  - icon: 🧠
    title: Flexible models
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

See **[Choosing a backend](guide/backends.md)** for when to use each, and **[Turbo & model
specialization](guide/turbo.md)** for the SIMD backend and how to add a Turbo specialization to a
model.
