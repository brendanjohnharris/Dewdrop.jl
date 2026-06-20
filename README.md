# Dewdrop

[![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)
[![](https://img.shields.io/badge/%F0%9F%9B%A9%EF%B8%8F_tested_with-JET.jl-233f9a)](https://github.com/aviatesk/JET.jl)

A generic, intuitive, GPU-aware spiking neural network simulator for Julia.

Dewdrop consolidates ideas from the gold-standard simulators (Brian2, NEST, BrainPy) into a
fast, fixed-step, struct-of-arrays engine that is **CPU-first but GPU-ready by construction**:
every hot path is written as a portable broadcast or `KernelAbstractions` kernel and runs
unmodified on `Array` and on device arrays. It follows SciML conventions by implementing the
`CommonSolve` interface over its own SoA types rather than the flat-vector container convention.

## Quick start

```julia
using Dewdrop

# a leaky integrate-and-fire model (parameters are plain code)
m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)

# 64 units driven by a constant current
prob = DewdropNetwork(m, 64; input = 0.5, tspan = (0.0, 1000.0))
sol  = solve(prob, FixedStep(0.1))
firing_rate(sol)
```

A connected network adds a `Projection` (synapse model + connectivity) and, optionally, an
external `PoissonDrive`:

```julia
conn  = fixed_prob(CPU(), N, N, 0.1; weight = pre -> pre ≤ NE ? J : -g*J, delay = 15, seed = UInt64(1))
prob  = DewdropNetwork(m, N; input = 0.0, tspan = (0.0, 600.0),
                       projection = Projection(DeltaSynapse(), conn),
                       drive = PoissonDrive(; rate = 6.0, weight = J, seed = UInt64(2)))
sol   = solve(prob, FixedStep(0.1); record_spikes = true)
times, ids = raster(sol)
```

## Status

Early development. Implemented: LIF neurons (exact subthreshold propagator), current-based
(CUBA) and delta synapses, per-synapse heterogeneous conduction delays, event-driven sparse
scatter, fixed-probability connectivity, counter-based RNG and Poisson drive, opt-in recording,
and the `CommonSolve` verb interface. Validated against the analytic LIF f–I curve and the
Brunel (2000) asynchronous-irregular regime. GPU backend, conductance-based (COBA) synapses,
an `@neuron` macro, and labeled outputs are in progress.

The design is CPU-first with GPU-readiness enforced in CI (via `JLArrays` +
`allowscalar(false)`); the test suite is Aqua- and JET-clean.
