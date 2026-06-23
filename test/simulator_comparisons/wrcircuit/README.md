# WRCircuit: Dewdrop ⇄ BrainPy reproduction

End-to-end reproduction of the spatial FNS E/I "working-regime" circuit (the BrainPy `Spatial` model in
[`WRCircuit.jl`](../../../../DDC/WorkingRegime.jl/WRCircuit.jl)) in native Dewdrop, validated to reproduce
the reference dynamics **bit-for-bit up to numerical error** on a small, seeded network (CPU, < 30 s).

## What is reproduced

The model: conductance-adaptation FNS neurons (`C dV/dt = -gL(V-VL) - gK(V-VK) + I`, `τK dgK/dt = -gK`,
spike-triggered `gK += ΔgK`; E adapts, I does not), dual-exponential COBA synapses on four distance-
dependent recurrent paths (E→E, E→I, I→E, I→I) plus an external Poisson population driving E and I.

Because the connectome (JAX Gumbel-top-k), per-edge weights (`correlate_weights`), initial V (`Uniform`)
and the external Poisson spikes are all JAX-PRNG outputs, no independent RNG can regenerate them. So the
**structure is exported from a seeded BrainPy run and ingested by Dewdrop**; only the deterministic
integration is then verified. Two choices make an exact comparison possible (both documented in `run.py`):
`bm.enable_x64()` (float64 both sides) and a constant synaptic rise time `τr` (BrainPy randomises it per
source; pinned constant via a localised monkeypatch, the WRCircuit source is **not** edited).

The native re-expression (`Dewdrop.wrcircuit`, `src/WRCircuit.jl`) reproduces BrainPy's integration scheme
with two pieces:
- **`FrozenDualExpSynapse`** — dual-exp COBA whose current `g·(Erev−V)` is frozen at the pre-step `V`
  (BrainPy's `sum_current_inputs`), contributing to the input current and not the membrane leak. This lets
  the unmodified `FNSNeuron` reproduce BrainPy's `A = −(gL+gK)/C` exactly.
- **`PrescribedCOBA`** — replays the external Poisson population's dual-exp conductance from the exported
  spike raster, so no spike-source population is needed.

The one calibration is a `−1`-step delay convention (`DELAY_ADJ`): BrainPy's delay `D` onsets the
postsynaptic conductance at `spike+D`; Dewdrop's ring buffer + dual-exp onsets at `spike+D+1`.

## Result (N = 180, 400 ms, dt = 0.1)

| pop | spikes bp/dd | rate bp/dd (Hz) | mean \|ΔV\| | V cor | spike match (exact / ±1) |
|-----|--------------|-----------------|------------|-------|--------------------------|
| E   | 128 / 128    | 2.22 / 2.22     | 0.0019 mV  | 0.99980 | 121/128 (128/128 within ±1) |
| I   | 37 / 37      | 2.57 / 2.57     | 0.0012 mV  | 0.99989 | 36/37 (37/37 within ±1) |

The residual (a handful of spikes jittered by one step) is chaotic amplification of floating-point ordering
differences between JAX and Julia, not a scheme mismatch. See `wrcircuit_comparison.png`.

## CPU timing (BrainPy `Spatial` vs Dewdrop, same network)

| N    | nedges  | BrainPy (s) | Dewdrop (s) | speedup |
|------|---------|-------------|-------------|---------|
| 180  | 14628   | 0.69        | 0.35        | 2.0×    |
| 720  | 58816   | 0.95        | 0.42        | 2.3×    |
| 1280 | 104664  | 1.07        | 0.54        | 2.0×    |

## Files / running it

- `brainpy/run.py` — build the seeded `Spatial` net, run it, export structure + traces to `brainpy/out/`.
  Needs the BrainPy env: `JAX_PLATFORMS=cpu python brainpy/run.py` (env vars `WRC_RHO`/`WRC_DX`/`WRC_NU`/…
  tune the size/drive; `WRC_BENCH=1` adds a timing line).
- `dewdrop/run.jl` — ingest the export, build `wrcircuit`, solve, write matching traces/spikes to
  `dewdrop/out/`. `julia dewdrop/run.jl` (optional arg overrides `DELAY_ADJ`).
- `compare.jl` — metrics table + the Fathom comparison figure. `julia compare.jl` (`metrics-only` to skip
  the figure). Defines `compare_wrcircuit(; plot)`.
- `validate.jl` — guarded `@testset` asserting the reproduction (skips if the export is absent).
- `bench.sh` — the CPU timing sweep above.

The self-contained unit tests for the new synapse models + the `wrcircuit` builder (no BrainPy needed) live
in `test/wrcircuit.jl` and run in the main suite.
