# Monitor (Recording) Framework — Design Plan (for review)

Status: **proposed, not built**. This supersedes the current hardcoded `record_spikes`/`record_voltage` recording.

## 1. Motivation & the gap today

Recording today is a two-case MVP:
- `record_spikes=true` → an `N×nsteps` Bool/BitMatrix of the spike mask → `raster`.
- `record_voltage=true|idx` → `V` (the membrane potential), all-N or a neuron index subset.

That is the *entire* surface. `_record_voltage!` reads `integ.state.state.V` and nothing else. There is no way to record other state variables (`refrac`, future `w` for AdEx), synaptic state (`g`/`Isyn` per projection), the per-neuron accumulators (`gtot`/`itot`), aggregate functions (population rate, mean V), or to group by (sub)population. The tell: `Timeseries(sol; var=:V)` in the TimeseriesBase ext has a `var` kwarg that is **cosmetic** — it only renames the array; the data is always `V`.

**Goal:** record arbitrary variables, from arbitrary unit subsets (and later named subpopulations), per-unit *or* aggregated, optionally strided — all type-stable, allocation-free, GPU-safe, and generically labeled by TimeseriesBase.

## 2. Design principles

1. **Reuse the projection-tuple machinery.** The engine already runs type-stable tuple-recursion over `integ.syns` (a tuple of heterogeneous synapse states), dispatch-free (`opt:0`). Monitors are the same shape: a (named)tuple of heterogeneous monitor objects, unrolled in the `:record` phase.
2. **The `:record` schedule slot already exists** (`Schedule.jl`: "record — snapshot monitored variables"); generalize its body.
3. **GPU-safety (M0 contracts 3/8):** preallocated arch-owned buffers, broadcast/view writes, no host scalar indexing, reductions via `mapreduce`. Device-resident.
4. **Pluggable + labeled:** each monitor's buffer + metadata → a labeled `ToolsArray` via the TimeseriesBase ext, with `Var`/`Neuron`/(`Population`/`Area`)/`Time` dims.
5. **Backward-compatible:** `record_spikes`/`record_voltage` become sugar; `sol.spikes`/`sol.voltages`/`raster`/`firing_rate` keep working.

## 3. The Monitor abstraction — four orthogonal axes

A monitor is `(source, selector, reduction, timing)`:

| axis | options |
|---|---|
| **source** (what) | `StateVar(:V)` (any `statevars` column) · `SynVar(proj, :g\|:Isyn)` · `Accumulator(:gtot\|:itot)` · `SpikeMask` · `Probe(f)` (arbitrary `f(integ)→array`) |
| **selector** (where) | `:all` · an index vector/range · *(Phase 3)* a named `(sub)population` |
| **reduction** (how) | `nothing` (per-unit) · a reducer `f` (`sum`/`mean`/`std`/custom) → scalar (or small vector) per step |
| **timing** (when) | `every=1` · `every=k` (stride) |

The buffer shape follows: per-unit → `(n_selected, ⌈nsteps/every⌉)`; aggregate → `(1, ⌈nsteps/every⌉)` (or `(n_out, …)` for multi-output reducers).

## 4. User-facing spec API

```julia
sol = solve(prob, alg; record = (
    spikes = Spikes(),                     # the raster (Bool)
    V      = Trace(:V),                     # all neurons' V
    Vsub   = Trace(:V; of = 1:10),          # a subset
    Isyn   = Trace(:Isyn; projection = 1),  # synaptic current of projection 1
    gtot   = Trace(:gtot),                  # the conductance accumulator
    rate   = Reduce(Spikes(), sum),         # population spike count / step  → a Time series
    meanV  = Reduce(Trace(:V), mean),       # population mean V / step
    cv     = Probe(integ -> ...; n = 1),    # arbitrary derived quantity
))

sol.record.V       # labeled Time × Neuron  (TimeseriesBase loaded)
sol.record.rate    # labeled Time
```

Spec constructors (lightweight, immutable, isbits where possible):
- `Trace(var; of=:all, projection=nothing, every=1)` — per-unit trace of a state/syn/accumulator variable.
- `Spikes(; of=:all, every=1)` — the spike mask.
- `Reduce(inner, f; every=1)` — aggregate an inner `Trace`/`Spikes` with reducer `f`.
- `Probe(f; n=…, every=1)` — arbitrary function of the integrator (kernel-safe if on GPU).

`record` is a `NamedTuple`, so the names become `sol.record.<name>`.

## 5. Internal types + record mechanics

- `init` materialises each spec into a concrete `Monitor{Source, Sel, Red}` carrying its preallocated buffer + a column counter, and assembles them into a **`NamedTuple` of monitors** (`integ.monitors`), replacing the current `spike_rec`/`voltage_rec` fields.
- `:record` phase: `_record_all!(integ.monitors, integ)` — tuple-recursion over `values(monitors)`, dispatch-free.
- `record!(m, integ)`:
  1. stride gate: `integ.n % m.every == 0` (cheap; `every` in the type or a field).
  2. read the source as an array/view (type-stable; see §8).
  3. select (a `@view src[idx]` or the whole array).
  4. reduce (if a reducer) or copy.
  5. write the column: `m.buffer[:, col] .= selected` (per-unit) or `view(m.buffer, :, col) .= reduced` (aggregate).

## 6. Solution changes

- `DewdropSolution` gains `record::NamedTuple` mapping name → `(buffer, meta)` where `meta` carries the selector indices, stride, and kind (per-unit / aggregate / spikes) needed for labeling.
- `spike_count` stays. `raster(sol)`/`firing_rate(sol)` read from `sol.record.spikes` (or unchanged behaviour).
- Backward-compat: `record_spikes=true` injects a `spikes=Spikes()` monitor; `record_voltage=true|idx` injects `V=Trace(:V; of=…)`. `sol.spikes`/`sol.voltages`/`sol.voltage_idx` become thin accessors over `sol.record` (or kept as populated fields). **Decision (a) below.**

## 7. TimeseriesBase labeling — generalized

The ext stops special-casing V and labels *every* monitor from its `meta`:
- per-unit → `ToolsArray(buffer', (𝑡(times), Neuron(meta.idx)); name)` — Time × Neuron.
- aggregate → `ToolsArray(vec(buffer), 𝑡(times); name)` — a Time series, optionally tagged `Var(name)`.
- probe → shape from `meta`.

`times = (1:n_cols) .* (dt*every)`. This **fixes the cosmetic `var`**: the data now genuinely corresponds to the variable. The `Var` dim (currently unused) carries the variable name; `Population`/`Area` light up in Phase 3.

## 8. GPU-safety & type-stability notes

- `integ.monitors` is a `NamedTuple` of concrete monitor types → fully static dispatch in `:record` (target: `step!` stays `opt:0`, `@allocated==0`).
- **Source field access must be compile-time.** `StateVar(:V)` reads `state.state.V` via `getproperty(state.state, Val(:V))` (or the symbol in the monitor's type parameter), so the column is resolved statically — not a runtime symbol lookup. `SynVar(proj, …)` indexes `syns[proj]` with `proj` a type parameter.
- **Reductions** carry `f` in the type parameter (`Reduce{Inner,F}`), so dispatch is static; the device reduction is `mapreduce(f, op, @view src[idx])` → a host scalar.
- **Aggregate writes are the one wrinkle on GPU:** writing one element per step (`buffer[col]=s`) is scalar indexing. Use a 1-element broadcast `view(buffer, :, col) .= s` (a kernel filling one slot — correct under `allowscalar(false)`, a small per-step launch). Cheap data; acceptable. Per-unit traces are already a column-broadcast (GPU-safe, as today). Flag for the GPU backend (M6) if the per-step 1-element launch matters.
- **`Probe(f)`** must be a kernel-safe function when running on GPU (document the constraint, or restrict `Probe` to CPU). **Decision (e).**

## 9. Phasing

- **Phase 1 — engine, single population.** Monitor types (`Trace`/`Spikes`/`Reduce`/`Probe`), the `:record` generalization, `sol.record`, backward-compat sugar. Index selectors only. CPU-complete; GPU per-unit works; GPU aggregates per §8.
- **Phase 2 — ext.** Generalize `TimeseriesBaseExt` to label all monitors (the `Var` dim). Drop the cosmetic `var`.
- **Phase 3 — gated on multi-population compose.** Named `(sub)population` selectors (`of=:E`) resolved against the network's addressing metadata; `Population`/`Area` dims. **The monitor framework is the natural forcing function for the hierarchical-addressing layer.**

## 10. Open decisions (expanded — please review)

### (a) Backward-compat depth
**Q.** When we move to `record=(…)`/`sol.record`, what becomes of the existing `record_spikes`/`record_voltage` kwargs, the `sol.spikes`/`sol.voltages`/`sol.voltage_idx` fields, and `raster`/`firing_rate`?
**Options.** (i) *Clean break* — only `record=(…)`; `sol.record.spikes`/`.V`; remove the old kwargs+fields. (ii) *Sugar + accessors* — `record_spikes=true`/`record_voltage=…` inject `Spikes()`/`Trace(:V)` monitors; `sol.spikes`/`.voltages` become thin accessors over `sol.record`.
**Trade-offs.** A clean break churns essentially the whole suite — `recording.jl`, `brunel.jl`, `vogels_abbott.jl`, `drive.jl`, `plots.jl` all use `record_spikes=true` + `raster`. The sugar is ~5 lines and keeps every existing test/script green. `firing_rate` is independent (it reads the always-on `spike_count`, not a monitor); `raster` just reads the spikes monitor. "Pre-1.0 so a break is allowed" is true, but the test suite is the de-facto user.
**Recommendation. Sugar + accessors.** `sol.record` is canonical; the old kwargs are sugar; the old fields are accessors; `spike_count` stays a separate cheap always-on accumulator (not a monitor). Lowest churn, and `record_spikes=true`/`raster(sol)` remains the natural one-liner for the common case.

### (b) Spec names
**Q.** The user-facing constructor names. **Clash check (vs Base / Makie / TimeseriesBase+DimensionalData / Unitful):** `Trace`, `Spikes`, `Reduce`, `Aggregate`, `Probe`, `Monitor`, `Bin`, `Snapshot` are all **clear**; `Record` and `Observable` **clash with Makie** (out).
**Recommendation. `Trace` / `Spikes` / `Aggregate` / `Probe`** — i.e. `Reduce`→`Aggregate` (reads as a noun, "an `Aggregate` of `Spikes` by `sum`", and sidesteps the Base-`reduce` verb feel; both are clash-free). This is your API surface, so the final call is yours; `Reduce` is equally available if you prefer the verb.

### (c) GPU aggregates
**Q.** An aggregate produces one scalar per step; `buffer[col]=s` is forbidden scalar-indexing on a device array.
**The real cost** isn't the write — it's that the per-step reduction (`mapreduce` over a device view) returns a *host* scalar, forcing a **device→host sync every recorded step**. Per-unit traces have no sync (a pure device column copy).
**Options.** (i) *Ship simple now:* `mapreduce`→host `s`; `view(buffer,:,col) .= s` (1-element broadcast, `allowscalar`-safe) — incurs the per-step sync. (ii) *In-kernel device aggregate:* a kernel reduces and writes `buffer[col]` on-device, no host round-trip (the M6 fused form).
**Trade-offs.** The engine is CPU-only until M6, so the sync cost is hypothetical today; aggregates are recorded sparingly and `every>1` cuts the sync frequency.
**Recommendation. Ship the simple path; the in-kernel device aggregate is an M6 optimization.** Note the per-step sync, mitigate with stride, and keep the write localized so swapping in an on-device reduce later is a one-spot change.

### (d) `sol.record` as a `NamedTuple`
**Q.** `NamedTuple` (named access `sol.record.V`) vs a `Dict`.
**Trade-offs.** `NamedTuple` → type-stable access, concrete buffer types flow to the labeling ext, symmetric with the `record=(…)` input; the only cost is `DewdropSolution` gains a type parameter (already the case for the projections tuple). `Dict{Symbol,Any}` → type-unstable `Any` values that break the ext and downstream analysis.
**Recommendation. `NamedTuple`, confirmed.**

### (e) `Probe` on GPU
**Q.** `Probe(f)` runs an arbitrary `f(integ)`; on GPU `f` must be kernel-safe (broadcast/reduction over device arrays, no scalar indexing).
**Options.** (i) *Document the constraint* — let any `f` through, fail at runtime if unsafe. (ii) *Hard-restrict* `Probe` to CPU.
**Trade-offs.** Most probes ARE expressible safely; a naive scalar-indexing `f` would fail cryptically under `allowscalar(false)`. But there is no GPU backend yet to restrict against, and `Trace`/`Aggregate` cover the common cases so `Probe` is a power-user escape hatch.
**Recommendation. CPU-unrestricted now; document the kernel-safety constraint for the M6 GPU path (don't hard-restrict).** Add good "use broadcast/reduce, not scalar indexing" error messages when the GPU backend lands.

### (f) Multi-output reducers
**Q.** Aggregate reducers that return a *vector* per step (quantiles, histograms, per-bin counts) in v1, or scalar reducers only?
**Key insight.** A multi-output reducer is just a per-unit trace whose "units" are output bins — the buffer is already `(n_out, nsteps)`, with scalar being `n_out=1`. And `Probe(f; n=m)` already records arbitrary vector outputs in v1.
**Recommendation. Scalar `Aggregate` in v1**, with the buffer shaped `(n_out, nsteps)` so multi-output is a trivial later generalization; `Probe(f; n=m)` is the v1 escape hatch for vector-valued derived quantities. Named convenience reducers (`quantiles`, `histogram`) are a fast-follow, not a redesign.

### (g) Windowed / streaming flush
**Q.** M0 contract 8 wants windowed device→host flush (O(1) host transfers per window); v1 uses full preallocated buffers (`n × nsteps`, like today). Windowed now, or later?
**Trade-offs.** Full buffers are the memory cliff — but the framework's *own* knobs already cut it hard: **subset recording** (10 traces not 10⁴), **aggregates** (a scalar/step), **stride** (every-k). Windowed flush is real machinery (window buffer, host accumulation/streaming, window sizing, boundary handling) and pays off mainly for the GPU backend + very long runs.
**Recommendation. Full buffers in v1; subset/aggregate/stride ARE the v1 memory mitigation; windowed flush folds into M6.** Keep the column-write window-ready (a window-relative `col` + a flush hook is a localized later change).

### (h) Snapshot point
**Q.** `:record` runs after `:reset`, so `Trace(:V)` records V **post-reset** — a neuron that spiked this step shows `Vr`, not the threshold-crossing value. Intended?
**Context.** This is the current behaviour and the `recording.jl` assertion (V ∈ [EL, Vθ)). It's the standard "subthreshold trace + separate spike train" convention: record V (subthreshold) and spikes separately; the raster gives timing. For LIF there's no action-potential shape, so post-reset V *is* the meaningful subthreshold state.
**Options.** (i) *Keep* end-of-step/post-reset. (ii) *Pre-reset/peak* — records `≈Vθ` for spiked units, conflating spikes into the V trace (cosmetic for a reset model). (iii) *Per-monitor `at=`* — over-engineered now.
**Recommendation. Keep end-of-step (post-reset); document it explicitly; spike timing is the spike monitor's job (don't conflate).** If a future spike-shaped model (HH/AdEx) needs a pre-reset snapshot, add a per-monitor `at=` option then.

## 11. Test plan

- **Parity:** `Trace(:V)` reproduces the current voltage recording exactly; `Spikes()` reproduces the raster.
- **Variables:** `Trace(:refrac)`, `Trace(:gtot)`, `Trace(:Isyn; projection=i)` record the correct arrays (vs manual extraction).
- **Aggregates:** `Reduce(Spikes(), sum)` == per-step population spike count; `Reduce(Trace(:V), mean)` == per-step mean V.
- **Selectors:** `of=idx` records the right subset; **Phase 3:** `of=:E`.
- **Stride:** `every=k` → buffer length `⌈nsteps/k⌉`, correct time axis.
- **Probe:** an arbitrary function records correctly.
- **Backward-compat:** `record_spikes`/`record_voltage` sugar; `sol.spikes`/`sol.voltages`; `raster`/`firing_rate`.
- **GPU-readiness:** per-unit traces step under `JLArray` + `allowscalar(false)`; aggregates per §8.
- **Quality:** `step!` with monitors stays `opt:0` (JET) and `@allocated==0`.
- **Labeling:** each monitor → correctly-dimensioned `ToolsArray` (`Var`/`Neuron`/`Time`), and the cosmetic `var` is gone.
