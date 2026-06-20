# Changelog

All notable changes to Dewdrop.jl are documented here.

## [Unreleased]

### Added

- Initial spiking neural network engine.
  - **Neurons & synapses:** LIF neurons with the exact subthreshold (linear-propagator)
    integrator; current-based (CUBA) and instantaneous (delta) synapses.
  - **Connectivity & delays:** CSR connectivity with per-synapse heterogeneous conduction
    delays (NEST-style ring buffer); fixed-probability random connectivity with signed
    excitatory/inhibitory weights.
  - **Propagation:** event-driven sparse scatter written once as a `KernelAbstractions`
    kernel (CPU and device), with `Atomix` atomic accumulation.
  - **Stochasticity:** counter-based (Philox) RNG, Poisson sampling, and external Poisson
    drive --- reproducible and identical across threads and devices.
  - **Interface:** the `CommonSolve` verbs (`init`/`step!`/`solve!`/`solve`) over the
    engine's own struct-of-arrays state; a compile-time, `Val`-dispatched within-step
    schedule; opt-in spike-raster and voltage-trace recording.
- CPU-first with GPU-readiness enforced in CI (`JLArrays` + `allowscalar(false)`); the test
  suite is Aqua- and JET-clean.
- Validated against the analytic LIF f--I curve and the Brunel (2000) asynchronous-irregular
  regime (reproducing its classical raster / population-rate figure).
