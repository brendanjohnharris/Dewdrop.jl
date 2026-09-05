# Changelog

All notable changes to Dewdrop.jl are documented here.

## [Unreleased]

### Added

- Short-term synaptic depression (`STD`, Tsodyks-Markram): a presynaptic spike transmits `w·x` and then
  depletes the resource `x ← (1-U)·x`, which recovers towards 1 with `τD`. Weights are never mutated, so
  unlike `STDP` the rule is also supported on the ensemble-batched path (which keeps one shared
  connectome and so cannot carry per-column weights).
- Time-axis reduction on every monitor spec: `bin = k` reduces each window of `k` steps (spikes summed
  into `UInt16` counts, signals averaged) as the alternative to the `every = k` stride. The two are
  mutually exclusive, and `every > 1` over a spike record is refused outright, since striding an event
  record discards spikes rather than coarse-graining them. `raster` and the statistics built on it
  require an unbinned recording.
- Per-postsynaptic-neuron synapse coefficients: a coefficient may be a length-`N` vector, resolved per
  cell inside the fused kernel. It composes with the ensemble axis, so a batched sweep can vary a
  coefficient per member and per neuron at once.
- Block-diagonal batching of members whose connectome already lives on a device: the edge arrays are
  pulled to the host in one bulk copy instead of being read a scalar at a time.
- Batch members are checked for the fields no execution mode can express per member (`drive`, `noise`,
  `stimuli`, `schedule`); a member that differs is refused instead of being silently replaced by
  member 1's.
- `PoissonSource` refuses a rate whose per-step spike probability would exceed 1 (every source would
  fire every step, with no Poisson statistics left).
- The performance advisor emits each suggestion once per session and is safe to call from several
  threads at once.
- Initial spiking neural network engine.
  - **Neurons & synapses:** LIF neurons with the exact subthreshold (linear-propagator)
    integrator; current-based (CUBA) and instantaneous (delta) synapses.
  - **Connectivity & delays:** CSR connectivity with per-synapse heterogeneous conduction
    delays (NEST-style ring buffer); fixed-probability random connectivity with signed
    excitatory/inhibitory weights.
  - **Propagation:** event-driven sparse scatter written once as a `KernelAbstractions`
    kernel (CPU and device), with `Atomix` atomic accumulation.
  - **Stochasticity:** counter-based (Philox) RNG, Poisson sampling, and external Poisson
    drive, all reproducible and identical across threads and devices.
  - **Interface:** the `CommonSolve` verbs (`init`/`step!`/`solve!`/`solve`) over the
    engine's own struct-of-arrays state; a compile-time, `Val`-dispatched within-step
    schedule; opt-in spike-raster and voltage-trace recording.
- CPU-first with GPU-readiness enforced in CI (`JLArrays` + `allowscalar(false)`); the test
  suite is Aqua- and JET-clean.
- Validated against the analytic LIF f-I curve and the Brunel (2000) asynchronous-irregular
  regime (reproducing its classical raster / population-rate figure).

### Fixed

Almost all of these were silent: the run completed and returned a number computed without the feature
that had been configured. Each now has a regression test.

- `solve(::NetworkBatch, …)` threw a `MethodError` for every `Float32` model, and for any GPU run, after
  the whole simulation had already finished: `BatchSolution` declared `duration::Float64` and
  `spike_counts::Vector{Vector{T}}`, which no member float type other than `Float64` (and no device
  array) could satisfy. Every mode now collects its counts to the host, and `duration` keeps the
  member's precision.
- `stimulate!`-attached `stimuli` were dropped by the `:shared` and `:multirun` batch modes, and by the
  `input`-sweep constructor, so those members ran with no stimulus at all. The three rebuild sites that
  each carried their own copy of the field list are now one helper.
- The `:fused` batch mode built its per-member overrides from the model's own fields but matched them
  against the underlying scalar model's, so a sweep over `Heterogeneous` members silently ran member 1's
  parameters for every member. Such a sweep now routes to `:multirun`; forcing `:fused` errors.
- `backend = Differentiable()` omitted the once-per-step hook that generates a streaming drive's events,
  so a network driven by `PoissonSource` or `SpikeSourceArray` trained against no input at all.
- A `Heterogeneous` group inside a `MultiModel` was resolved at the flat neuron index by the step kernel
  but at the group-local index when initialising `V`. Override arrays are now documented and validated
  as spanning the whole population; a group-length array had been read past its end under `@inbounds`.
- `StreamingWelch` accumulated its moments in the state float type. For a membrane trace (a mean that
  dwarfs its fluctuation) the `Float32` variance came out negative and the analytically removed mean
  reappeared as a DC peak larger than the signal. The accumulators are now `Float64`; the segment
  buffer, much the largest array, still follows the state type.
- Batched runs lost the named-subpopulation registry: `Trace(:V; of = :E)` worked on the scalar path and
  failed with "unknown selector" under `batch = B`, and the solution carried no registry. The registry
  now reaches the batched monitors, and `firing_rate(sol, :E)` works on a `BatchedSolution`.
- Block-diagonal stacking applied member 1's synapse to every member's edges, and dropped `plasticity`
  from the stacked projection entirely. Members are merged only when they genuinely share both.
- `WhiteNoise` and `PoissonDrive` could not be passed through `stimuli =` or `stimulate!` at all
  (a `MethodError` at construction), although both are documented as ordinary stimuli. Membership in the
  seam is now the `stim_point` trait rather than the `AbstractStimulus` subtype.
- `backend = Turbo()` tested the `noise` field rather than the `:noise` application point, so a noise
  stimulus arriving through `stimuli =` ran silently noise-free.
- `TimedArray(data; as = …)` accepted any symbol; a misspelling became a type parameter matching no
  application point, making the stimulus a silent no-op.
- `correlate_weights!` indexed its in-degree array from edge data under `@inbounds`, so a `targets`
  narrower than the connectome's post set wrote out of bounds and returned without error.
- `SparseCSR` validated nothing: an out-of-range `post` became an out-of-bounds write into the delay
  ring, and a 0-step delay deposited into the slot the deliver phase had just cleared, arriving a whole
  ring length later (silently the longest delay rather than the shortest).
- `temporal_average(sol, var)` returned the first recorded trace regardless of `var`. `RecordResult` now
  carries the variable it recorded.
- `@neuron` with all-integer parameter defaults produced an integer model, hence integer state columns
  and an `InexactError` on the first membrane store; and every generated model required a field named
  literally `EL`, since that is what the default initial `V` read. The macro now floats its parameters
  and derives the resting potential from `@asymptote` at zero input.
- `Aggregate` accepted any reducer symbol: the CPU path raised a `MethodError` mid-solve while the GPU
  kernel silently computed a sum. Validated where it is written.
- The Welch reducer admitted `nfft = 2` (at `f_min = fs/2`), where the symmetric Hann window is exactly
  zero and every normalisation divides by it, returning `NaN`.
- The serial `:current` broadcast passed the current accumulator where the fused kernel passes the
  frozen start-of-step `V`, so a voltage-dependent `:current` stimulus would have diverged between them.
- A batched run with a swept `EL` started every column at the unswept base model's resting potential.
- `STDP` and `STD` checked `npre == npost` but index their traces by the flat neuron index, so a
  projection narrower than the population read past the end of them.
- `distance` computed the periodic minimum image as `min(δ, period - δ)`, which is negative (and squares
  into a spurious distance) for a separation wider than the box.
- `connectivity(network)` silently plotted only the first projection when the projections binned to
  different shapes.
- `show` treated a host-resident view or range as device-resident and skipped its extrema summary.

### Changed

- Keywords are now refused on the path that does not read them, rather than accepted and ignored:
  `input` / `streams` / `syn_overrides` / `model_overrides` are per-member data and need `batch = B`,
  while `backend` / `step` choose a dense-step implementation and do not apply to a batched solve.
- `project!` rejects unknown keywords instead of dropping them, so a misspelling no longer silently
  leaves that setting at its default.
- `fixed_prob` validates `0 ≤ p ≤ 1` rather than failing inside the gap sampler. Note that its
  `allow_self` still defaults to `true`, where `distance_prob` and `distance_fixed_count` default to
  `false`; `project!` passes `false` to all three.
- `InhomogeneousPoisson`'s documented rate unit was Hz while the code used the canonical per-`dt` unit,
  a silent factor of 1000. The docstring is corrected; `PoissonSource`, which genuinely does take Hz, now
  documents that as the deliberate exception it is.
- `SpikeSourceArray` checks its replay pattern against the run length at `init`, as `TimedArray` already
  did, instead of failing with a `BoundsError` part-way through the loop.

### Performance

- `cv_isi(sol)` grouped spikes with a full rescan of the raster per neuron (O(neurons × spikes)); it now
  makes one grouping pass, over the same visiting order, so the result is unchanged.
- `coarsegrain` traverses its column-major input neuron-innermost. Each output cell still sums its bin's
  steps in ascending order, so results are unchanged.
- A monitor's windowed device-to-host flush reuses one host landing buffer instead of allocating a
  window-sized array on every flush.
- The batched Poisson drive draws once per source rather than once per (source, member); the draw is
  keyed on the source alone, so the repeated work was identical.
