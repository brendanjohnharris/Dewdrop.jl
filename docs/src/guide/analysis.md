```@meta
CurrentModule = Dewdrop
DocTestSetup = quote
    using Dewdrop
end
```

# Analysis and observables

Dewdrop separates two ways of computing statistics from a run:

- **Post-hoc host observables** ([`src/Stats.jl`](https://github.com/brendanjohnharris/Dewdrop.jl/blob/main/src/Stats.jl))
  --- pure functions of a [`DewdropSolution`](@ref) (or a raw raster matrix) that run on the host
  *after* the integration finishes, from the recorded [`Trace`](@ref)/[`Spikes`](@ref) data and the
  per-neuron positions.
- **Streaming device reducers** ([`MADev`](@ref), [`Welch`](@ref), [`SpikeRate`](@ref),
  [`Fano`](@ref)) --- monitors that fold each per-step sample into a fixed-size accumulator *during*
  the run, on whatever [architecture](backends.md) the state lives on, so the full trace is never
  materialised.

The first group is convenient and exact over an already-recorded trace; the second avoids storing a
trace at all, at the cost of fixing the statistic before the run.

## Host observables

These are the spatial-network measures ported from the reference `stats.py` and cross-validated
against it. Each has a matrix-core method over a Neuron×Time raster `S` (Dewdrop's recording
orientation) and a `sol`-based wrapper that pulls the recorded data, the time step `sol.dt`, the
named-subpopulation registry, and (for spatial measures) the positions off the solution.

The spike-raster measures need a full `Spikes()` recording; the trace measures need a `Trace(var)`.
Pass them through `record` (see [recording](recording.md)):

```julia
sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(), V = Trace(:V)))
```

| observable | computes | inputs → output |
|---|---|---|
| [`coarsegrain`](@ref) | sums a matrix into non-overlapping bins of width `binsize` along `dims` (default the time axis); trailing remainder discarded | `(S, binsize; dims=2)` → `Matrix` |
| [`susceptibility`](@ref) | population susceptibility `χ = ⟨ρ²⟩_t − ⟨ρ⟩_t²` of the active fraction `ρ(t)` (large when synchronous, small when asynchronous) | `(sol; bin, of)` → scalar |
| [`mua`](@ref) | multi-unit activity: population spike count per time bin (sum over neurons) | `(sol; bin, of)` → `Vector` |
| [`temporal_average`](@ref) | per-unit time average of a trace (e.g. mean membrane potential) | `(sol, var=:V; of)` → `Vector` |
| [`grand_distribution`](@ref) | histogram of all values in `X` into `nbins` equal-width bins | `(X, nbins)` → `(counts, centers)` |
| [`cv_isi`](@ref) | coefficient of variation of inter-spike intervals, averaged over neurons | `(sol; of)` → `Float64` |
| [`power_spectrum`](@ref) | Bartlett-averaged PSD of the raster (periodogram per segment, averaged over segments and neurons) | `(sol; n_segments, of)` → `(psd, freqs)` |
| [`efficiency`](@ref) | spatial coding efficiency per time bin `η = n·H/C` (entropy of the spatial spike distribution over energy cost) | `(S, bin_indices, tau; dt)` → `Vector` |
| [`radial_autocorrelation`](@ref) | radially-averaged 2-D spatial autocorrelation of each frame on a rectangular lattice | `(sol; dr, of)` → `(g_r, r_bins)` |

Most spike-raster wrappers take `bin` (in time units) to coarse-grain time first, and `of` to
restrict to a named subpopulation (e.g. `of = :E`); subpop slicing needs a full recording
(`idx = :all`).

```julia
chi = susceptibility(sol; bin = 1.0, of = :E)   # bin in time units, excitatory subpop
m   = mua(sol; bin = 1.0)                        # population count per 1.0-unit bin
psd, f = power_spectrum(sol; n_segments = 8)     # Bartlett over 8 segments
```

The spectral measures ([`power_spectrum`](@ref), [`radial_autocorrelation`](@ref)) use Dewdrop's
self-contained FFT (an internal module, not part of the public API), so they carry no FFTW
dependency.

Several observables are plain functions of arrays, callable directly:

```jldoctest
julia> counts, centers = grand_distribution([1.0, 1.0, 2.0, 3.0, 3.0, 3.0], 3);

julia> counts
3-element Vector{Int64}:
 2
 1
 3

julia> cv_isi([1.0, 2.0, 3.0, 4.0, 5.0])   # perfectly regular ISIs -> CV 0
0.0
```

### Positions travel onto the solution

The spatial measures need geometry. Positions set on the [`DewdropNetwork`](@ref) (via `positions =`,
or laid out with [`grid_positions`](@ref) / [`random_positions`](@ref); see
[connectivity & space](connectivity.md)) are host-side metadata that travel onto the solution as
`sol.positions`, kept host-resident even on a GPU run. [`radial_autocorrelation`](@ref) reads them
directly:

```julia
net = network(; arch = CPU(), tspan = (0.0, 1000.0))
population!(net, :pop, FNSNeuron(), 64 * 64; positions = grid_positions(64, 64))
# ...
sol = solve(build(net), FixedStep(0.1); record = (spikes = Spikes(),))
g_r, r = radial_autocorrelation(sol; dr = 0.05)   # uses sol.positions; needs a full rectangular grid
```

[`radial_autocorrelation`](@ref) errors if the solution carries no positions, or if the sites do not
form a full, evenly-spaced rectangular grid. [`efficiency`](@ref) takes its spatial grouping
explicitly as a `bin_indices` matrix (each entry the neuron indices in one spatial bin) rather than
reading `sol.positions`.

## Streaming device reducers

The reducers ([`MADev`](@ref), [`Welch`](@ref), [`SpikeRate`](@ref), [`Fano`](@ref)) are temporal
statistics computed *during* the run. Each consumes one sample per recorded step and accumulates a
reduced result on-device; the only memory that scales with run length is a fixed ring/segment buffer
(`O(maxlag)` or `O(nfft)`), never `O(nsteps)`. They are the on-device analogues of the corresponding
`TimeseriesTools` estimators, and run on CPU or GPU through the same code.

Use them when the trace itself is too large to store --- a long run over many (neuron, member) cells
--- and you only want the temporal statistic. They are available on the batched solve path (see
[batching](batching.md)); the result is shaped `(n_out, B)` over the selected units and batch
members, with a trailing axis for the per-lag / per-frequency / per-timescale parameter.

| reducer | computes | result |
|---|---|---|
| [`MADev`](@ref) | `p=1` mean-absolute-displacement of `var` at integer step `lags` | `(n_out, B, nlags)` |
| [`Welch`](@ref) | Hann-windowed, 50%-overlap Welch power spectrum of `var` (mean removed), minimum frequency `f_min` | `(n_out, B, nfreq)` |
| [`SpikeRate`](@ref) | per-(neuron, member) mean firing rate (total spikes / observation time) | `(n_out, B)` |
| [`Fano`](@ref) | per-(neuron, member) Fano factor curve (variance/mean of spike counts in width-`τ` windows) at `taus` | `(n_out, B, ntau)` |

```julia
record = (
    md   = MADev(:V; lags = 1:50, transient = 100),
    psd  = Welch(:itot; f_min = 1.0),
    rate = SpikeRate(),
    ff   = Fano(taus = [1.0, 2.0, 4.0]),
)
```

Common keywords: `of` restricts to a subpopulation; `transient` skips the first N recorded steps;
`every` records every Nth step (so the effective rate is `fs = 1/(every·dt)`, which sets `Welch`'s
`nfft = ceil(fs/f_min)`). `MADev` lags are in recorded-step counts; `Fano` `taus` are in the same
time units as `dt`.

The two groups answer different questions. Reach for the host observables when you have the trace and
want a spatial or population summary after the fact; reach for the reducers when storing the trace is
the bottleneck and a fixed temporal statistic is all you need.
