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

- The guard that refuses to record `:gtot`/`:itot` on a backend that never materialises them matched a
  bare `Trace` only, so `Aggregate(Trace(:itot), :sum)` slipped past it and returned a column of zeros
  with no warning. It now looks through an `Aggregate` to its inner spec. A `Probe` still cannot be
  checked (its `f` is opaque, and refusing every probe would reject the many that read neither).
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
- A `SpikeSourceArray` drive is now offset per member in a block-diagonal batch. It carries its own
  member-local `extconn`, like `PoissonSource`, but was treated as shareable, so members 2..B replayed
  onto member 1's neurons and were themselves left undriven, with no error.
- The connectivity builders and the per-step stimuli no longer share one counter-RNG key space.
  `random_positions`, `distance_prob`, `distance_fixed_count`, `fixed_prob`, `correlate_weights`,
  `PoissonSource` firing, the random initial voltage, `WhiteNoise`, `PoissonDrive` and
  `InhomogeneousPoisson` each mix a distinct tag into the seed. Reusing one seed across the builders
  gave a neuron's coordinate along dimension `d` the same draw as its connection probe against target
  `d`, correlating geometry with connectivity. The stimuli collided harder still: `WhiteNoise` and
  `PoissonDrive` both key on `(step, neuron)` and both default to `seed = 0`, and `draw_normal` builds
  its Box-Muller radius from the very word `draw_uniform` returns, so at the defaults the noise
  magnitude was a deterministic decreasing function of the drive's draw (correlation -0.67) rather than
  an independent process, and two `InhomogeneousPoisson` stimuli at equal rates emitted identical
  trains. Networks and noise realisations from a given seed differ from before; reproducibility from a
  seed is unchanged.
- `distance_fixed_count` throws when the kernel and the source/target sets admit fewer than `count`
  pairs, instead of returning a quietly smaller connectome, which contradicted its exact-count promise.
  Read as a per-target in-degree, `count = K * length(targets)` needs `K ≤ length(sources)`; a request
  beyond that was silently delivering a fully connected projection. The reference model this mirrors
  (WRCircuit's `Spatial.py`) already validated the same condition.
- The batched random initial voltage is drawn on the column's own `streams` entry rather than its raw
  column index, so an all-zero `streams` reproduces the scalar reference exactly, as documented, and the
  default `streams = 0:(B-1)` still gives each column an independent initial condition.
- `StreamingWelch` returns a zero spectrum for a channel with no variance rather than propagating a
  `NaN` out of the `0/0` variance-matching factor.
- `Welch`, `MADev`, `SpikeRate` and `Fano` report that they need a batched run when passed to a scalar
  `solve`, instead of surfacing a `MethodError` on an internal function.
- `@neuron` no longer requires the name `Dewdrop` to be bound in the calling module, so
  `import Dewdrop: @neuron` works; and a parameter named `V` or `I` is refused rather than silently
  shadowing the membrane or the input current in the hook expressions.
- `show` on a solution reports 0 Hz for an empty named subpopulation instead of `NaN Hz`, and skips the
  rate summary for a device-resident solution rather than launching a reduction per subpopulation. The
  residency check covers all four renderings: the solution, a `SubSolution` (`sol[:E]`), a
  `BatchedSolution`, and a `BatchSolution`, the last of which reduced once per member.
- The connectivity builders accept an empty `sources`/`targets`. Each probes `first(sources)` to type
  its weight and delay arguments, so an empty set (a subpopulation a sweep has driven to zero) escaped
  as a `BoundsError` from inside the builder rather than the empty connectome it describes. `fixed_prob`,
  `distance_prob` and `distance_fixed_count` now return that connectome without invoking a `weight`/`delay`
  callback, which has no valid index to take; `correlate_weights!` still reports any edge outside an empty
  `targets` through its existing range check.
- `Aggregate` refuses an inner spec that is not a `Trace` or `Spikes`. It reduces over the units of a
  per-unit source, but validated only that the inner `every`/`bin` were unset, and `Probe` carries those
  fields too: `Aggregate(Probe(f; n), :mean)` constructed and then died mid-solve on `inner.of`, a field
  `Probe` does not have. This is the condition `_check_accum_record` already documents itself as relying on.

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
- `Auto` and `coarsegrain` are no longer exported; both are unchanged and reachable as `Dewdrop.Auto`
  and `Dewdrop.coarsegrain`. Each collided with a package the documentation tells you to load alongside
  Dewdrop, and a name exported by two loaded modules resolves to neither: `Auto` with Makie's layout
  type (so `using Dewdrop, CairoMakie` broke `Auto()`), `coarsegrain` with TimeseriesBase's, which is a
  different operation (take every second element and stack them into a new dimension) rather than this
  one (sum into bins of a given width). They were the only two collisions against Statistics, Random,
  Unitful, TimeseriesBase, Makie, StructArrays, Adapt, CUDA and ForwardDiff.
- The performance advisor is off by default. Enable it with `Dewdrop.set_advice!(true)`; a single call
  is still suppressed with `solve(...; advise = false)`. It printed unsolicited `@info` hints on every
  run before, to the point that five documentation guides opened by silencing it.
- What makes a synapse a source model is declared once, as `is_source_synapse`; `_plastic_wrappable`
  and `_block_mergeable` derive from it rather than each restating it per type in a different file.
  That duplication is what let `SpikeSourceArray` be declared on one path and missed on the other.
- `STDP` documents where its pairing interval is measured. It is timed at the two somata, with the
  transmission delay playing no part, which is the midpoint of the two conventions in wide use: NEST
  assigns the whole delay to the dendrite, Brian2 and BrainPy assign it all to the axon, and those two
  differ from each other by twice the delay. NEST GPU uses the same convention as this.
- `Differentiable` documents that the surrogate is substituted in the forward pass as well as the
  backward one, so it simulates a smoothed, rate-like network rather than the spiking one the other
  backends run.

### Removed

- `src/FFT.jl`, the hand-written radix-2 / Bluestein transform behind `power_spectrum` and
  `radial_autocorrelation`. Both now use FFTW, which the package already depended on and already used
  for the streaming Welch spectrum. The two agree to a few ulp at every length tested (worst case
  1.5e-15 relative, over lengths 2 to 4096, powers of two, primes and awkward composites), the
  observables agree to under 1e-15, and FFTW is about 20× faster. The frequency-axis helper `_fftfreq`
  stays: `AbstractFFTs.fftfreq` takes a sampling rate where this takes a sample spacing.
- Thirteen hand-written copies of the same tuple recursion (`_deliver_all!`, `_accum_all!`,
  `_bpropagate_all!`, `_rec_tuple!` and the rest), replaced by one `_foreach_tuple!`. The per-step phase
  walks still unroll to straight-line code and the step stays allocation-free.

### Performance

- `cv_isi(sol)` grouped spikes with a full rescan of the raster per neuron (O(neurons × spikes)); it now
  makes one grouping pass, over the same visiting order, so the result is unchanged.
- `coarsegrain` traverses its column-major input neuron-innermost. Each output cell still sums its bin's
  steps in ascending order, so results are unchanged.
- A monitor's windowed device-to-host flush reuses one host landing buffer instead of allocating a
  window-sized array on every flush.
- The batched Poisson drive draws once per source rather than once per (source, member); the draw is
  keyed on the source alone, so the repeated work was identical.
- The CPU scatter threads only once a step deposits enough edges to pay for it. Threading costs a fixed
  `@threads` dispatch (~13 µs on 16 threads) however little work follows, and the scatter skips every
  non-spiking row, so at low firing each thread got almost nothing while every step paid the full
  dispatch; it was unconditional whenever more than one thread was available. Whole-solve on 16 threads:
  5.6× faster at N=400, 2.7× at N=1000, 1.35× at N=4000, unchanged at N=16000 where threading already
  paid. The step is allocation-free again below the threshold (it was allocating 6848 bytes per step).
  The same gate covers the two batched scatters, which thread over the member axis. Both branches leave
  identical ring contents (fixed-point counts, so integer addition is associative), so results are
  bit-identical either side of it.
- The device `Aggregate` monitor uses a parallel reduction instead of a single-threaded in-kernel loop:
  23× faster at 50k units and 98× at 200k on an L40S, and now independent of population size. Float
  `:mean` aggregates may differ in the last ulp, since the summation order changed; integer `:sum`
  aggregates are unchanged.
- The batched `Aggregate` monitor gets that same parallel reduction; it had kept the in-kernel walk,
  one thread per batch member. On an L40S at `batch = 32` its whole-run cost falls from 1.5 s to
  0.11 s at 10k units, making the solve itself 1.95× faster, and from 7.9 s to 0.10 s at 50k. The CPU
  backend gains 5-16×, a plain reduction being cheaper than a launch. The same last-ulp caveat applies
  to float `:mean` aggregates.
- The fused step materialises `itot`/`gtot` only when a monitor records them, keeping them in registers
  otherwise. That removes two per-neuron global stores per step, measured at 24% of a 10⁶-neuron GPU
  step. `Serial` and `Turbo` read the arrays and are unaffected.
- `StreamingWelch` transforms each completed segment in place. It allocated a windowed segment and its
  transform every segment, on the device as well as the host; both are preallocated scratch now, which
  adds `LinearAlgebra` (a standard library) as a dependency for `mul!`.
