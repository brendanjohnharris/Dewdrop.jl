# Reproducing the WRCircuit (BrainPy)

The WRCircuit is a spatial excitatory/inhibitory network of conductance-adaptation FNS neurons with
dual-exponential COBA synapses on four distance-dependent recurrent paths and an external Poisson drive --
the "working-regime" circuit from `WRCircuit.jl`, originally simulated in [BrainPy](https://brainpy.tech).
Dewdrop re-expresses it in native syntax and reproduces the reference dynamics **bit-for-bit up to numerical
error**: on a small seeded network the firing rates are identical, the mean membrane error is ~0.002 mV, and
the great majority of spikes match to the exact time step (all within one step). The residual is chaotic
amplification of floating-point ordering differences between JAX and Julia, not a scheme difference.

The full cross-simulator pipeline (export, ingest, compare, benchmark) lives in
`test/simulator_comparisons/wrcircuit/`; this page documents the two reusable building blocks and the
`wrcircuit` builder.

## Why exact reproduction needs an export

The connectome (a JAX Gumbel-top-k sample), the per-edge weights, the initial membrane potentials, and the
external Poisson spike train are all JAX-PRNG outputs. No independent RNG can regenerate them, so the
*structure* is exported from a seeded BrainPy run and ingested by Dewdrop; only the deterministic
integration is then verified. The integration scheme, by contrast, **is** reproduced from first principles.

## Frozen-current COBA

BrainPy (like Brian) evaluates a conductance synapse's current `g·(Erev − V)` at the **pre-step** `V` and
holds it constant over the step (`sum_current_inputs`), rather than folding the synaptic conductance into
the membrane leak (Dewdrop's default exact-COBA propagator). [`FrozenDualExpSynapse`](@ref) reproduces that
scheme -- it has the same dual-exponential conductance kinetics as [`DualExpSynapse`](@ref), but contributes
the frozen current to the input current `itot` and **nothing** to the effective leak `gtot`:

```julia
exc = FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)    # excitatory
inh = FrozenDualExpSynapse(; τr = 2.0, τd = 4.5, Erev = -80.0)  # inhibitory
```

Because the synaptic conductance never reaches the leak, the unmodified [`FNSNeuron`](@ref) step
(`denom = 1 + R·gK`, carrying only its own adaptation conductance) reproduces BrainPy's membrane coefficient
`A = −(gL + gK)/C` exactly. The explicit-vs-implicit choice lives in the *synapse*, not the neuron. For new
models prefer `DualExpSynapse` (the exact propagator); use `FrozenDualExpSynapse` only to match a BrainPy or
Brian COBA network.

## Replayed external drive

The external Poisson population is replayed as a [`PrescribedCOBA`](@ref): a per-target conductance
trajectory `g[:, n]` (precomputed from the exported external spike raster by running the same dual-exp
filter) injected each step as the frozen current `g[:, n]·(Erev − V)`. This avoids instantiating a
spike-source population for a drive whose spikes are already known.

## The `wrcircuit` builder

[`wrcircuit`](@ref) assembles the spatial FNS E/I network. `E` and `I` are `FNSNeuron` models (merged into a
per-neuron [`Heterogeneous`](@ref) model when they differ -- e.g. E adapts and I does not); `projections` is
a vector of recurrent [`Projection`](@ref)s over the flat `1:NE+NI` index space; `gext` is the optional
external conductance matrix. The `:E`/`:I` subpopulations are registered for `sol[:E]`, `firing_rate(sol, :I)`:

```julia
prob = wrcircuit(; NE, NI, E, I, projections, gext, positions, tspan)
sol  = solve(prob, FixedStep(0.1); v0 = v0, record = (v = Trace(:V), spikes = Spikes()))
```

## Delay convention

The one calibration when matching BrainPy is a `−1`-step delay adjustment: BrainPy's delay `D` onsets the
postsynaptic conductance at `spike + D`, whereas Dewdrop's ring buffer plus the dual-exponential's
deliver-then-decay onsets it at `spike + D + 1`. The reproduction subtracts one step from every synaptic
delay; a native Dewdrop model would simply use the intended delay.

## Performance

On the identical small network the Dewdrop solve is about **2× faster than BrainPy on CPU** (e.g. 0.35 s vs
0.69 s at N = 180, 0.54 s vs 1.07 s at N = 1280), and reproduces the same dynamics.

## API

```@docs
wrcircuit
FrozenDualExpSynapse
PrescribedCOBA
```
