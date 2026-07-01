# Simulator comparisons

A modular benchmark comparing Dewdrop against other spiking-network simulators (Brian2/brian2cuda,
BrainPy, NEST, …) on **one shared, well-defined test problem**, measuring how **wall time and memory
scale with network size** across CPU and GPU backends, and verifying that every simulator produces the
same simulation.

## Layout

```
simulator_comparisons/
  spec.toml                 # THE shared test-problem spec — every simulator reads this
  run_all.sh                # run every simulator sequentially, then compare
  compare_simulators.jl     # the Julia driver: discovers <sim>/out, verifies, plots scaling
  dewdrop/  run.jl          # Dewdrop  (Julia):  Serial / Fused / Turbo × cpu/gpu
  brian/    run.py          # Brian2   (Python): cpp_standalone + cuda_standalone
  brainpy/  run.py          # BrainPy  (Python): jax cpu/gpu
  nest/     run.py          # NEST     (Python): aeif_psc_exp, native multithreading (CPU gold standard)
  genn/     run.py          # GeNN     (Python): GPU code-gen, custom AdEx (CUDA)
  nestgpu/  run.py          # NEST GPU (Python): NEST's CUDA engine → plotted as nest(gpu)
  snn/      run.jl          # SpikingNeuralNetworks.jl (Julia): AdEx + CurrentSynapse, CPU-only
  stats_validation/         # (separate) Dewdrop Stats.jl vs BrainPy stats.py cross-validation
  <add e.g. neuron/ run.py> …
```

Each simulator lives in its OWN directory with its OWN language and environment: no Julia in the
Python folders, no Python in the Julia ones. The directories are independent; `compare_simulators.jl`
only reads their output.

## The test problem

A recurrent **E/I AdEx** network with fixed in-degree and CUBA exponential synapses (it stresses the
sparse synaptic scatter, the real SNN bottleneck), defined entirely in `spec.toml`. The connectome is
a **language-agnostic splitmix64 fixed-in-degree graph** (algorithm in `spec.toml`), so every
simulator builds the *byte-identical* connectome (verified across Julia and Python). The spec sweeps
a **doubling range of sizes** (`Ns = [1000 … 32000]`) so the comparison reveals scaling behaviour.

## Data contract

Each `<sim>/run.*` writes two CSVs into `<sim>/out/`:

- **`values.csv`** (`N,rate_hz,cv_isi,nedges`): the simulation result, used to verify all
  simulators ran the same problem (identical `nedges`; statistically-matching rate / CV-ISI: spike
  trains diverge by chaos, but the statistics agree, e.g. ~38 Hz / CV ~0.13).
- **`performance.csv`** (`simulator,backend,device,N,wall_s,mem_mb`): the throughput scan: pure
  simulation wall time and peak network memory per `(backend, device, N)`. `wall_s` excludes
  compilation / JIT (warmup + min-of-reps, or the standalone binary's run loop); `mem_mb` is the peak
  RSS above the post-import baseline (the network footprint).

## Running

One command runs every discovered simulator sequentially, then verifies + plots:

```bash
./run_all.sh                                       # all simulators → compare_simulators.jl
./run_all.sh dewdrop brian                         # only the named ones, then compare
```

`run_all.sh` continues past a simulator that fails (missing GPU, missing venv, …) and still compares
whatever produced output. Or run each simulator independently (each in its own environment), then compare:

```bash
./dewdrop/run.jl                                   # → dewdrop/out/{values,performance}.csv
brian/.venv/bin/python   brian/run.py              # → brian/out/...   (needs brian2 + brian2cuda)
brainpy/.venv/bin/python brainpy/run.py            # → brainpy/out/... (needs brainpy + jax[cuda12])
nest/.venv/bin/python    nest/run.py               # → nest/out/...    (needs nest-simulator)
genn/.venv/bin/python    genn/run.py               # → genn/out/...    (needs pygenn + CUDA)
nestgpu/.venv/bin/python nestgpu/run.py            # → nestgpu/out/... (writes simulator=nest, device=gpu)
./snn/run.jl                                       # → snn/out/...     (needs its own instantiated project)
./compare_simulators.jl                            # verify + plot → out/comparison.pdf
```

GPU runs of the Python simulators need a CUDA toolkit on PATH (`export CUDA_PATH=/usr/local/cuda-12.9`);
brian2cuda also needs the explicit-GPU prefs already set in `brian/run.py` (this node exposes the GPU
by UUID, which its auto-detect can't parse).

**NEST** is the established C++ CPU gold standard. It maps the spec onto `aeif_psc_exp` (AdEx with
current-based exponential synapses) and runs CPU-only (standard NEST has no GPU backend). It uses
busy-wait barriers synchronised every `min_delay` window (here 0.2 ms), so **running one thread per
core is catastrophic**: the main/OS thread has no free core and a single descheduled straggler stalls
every barrier (~40× slowdown). `run.py` therefore auto-selects the faster of ½ / ¾ cores per network
size; force a fixed count with `NEST_THREADS=N`. It also loosens the adaptive GSL solver tolerance
(`gsl_error_tol=1e-3`, vs the 1e-6 default); this gives byte-identical statistics (38.0 Hz / CV 0.127,
verified unchanged even at 1e-1) so it is NEST at its best, not a thumb on the scale. **Measure NEST with
no other load**: its busy-wait barriers make it acutely sensitive to CPU contention (a concurrent job can
inflate its times ~2×). Set up its env with
`uv venv nest/.venv --python 3.11 && uv pip install --python nest/.venv/bin/python nest-simulator numpy`.

**GeNN** is a GPU code-GENERATOR: it emits + compiles bespoke CUDA per model (so `build()`/`load()` carry a
one-time codegen+compile cost, excluded from `wall_s` like Brian's standalone compile). It has no built-in
AdEx, so `run.py` defines a custom AdEx neuron (forward-Euler) + `ExpCurr` (CUBA) synapses split into one E
population (weight GE) and one I (−GI), with the splitmix64 edge list pushed via `set_sparse_connections`.
Not on PyPI; build from source into its venv:
`uv venv genn/.venv --python 3.11 && uv pip install --python genn/.venv/bin/python numpy`, then
`CUDA_PATH=/usr/local/cuda-12.9 uv pip install --python genn/.venv/bin/python git+https://github.com/genn-team/genn`.

**NEST GPU** is NEST's own CUDA engine (PyNEST-like API, same `aeif_psc_exp`), so it reports
`simulator=nest, device=gpu` and plots as the GPU variant of the NEST line. It is **source-only** (CMake +
CUDA) and ships a build bug: `src/connect.h`'s `getMPIComm()` uses `MPI_Comm` without the `#ifdef HAVE_MPI`
guard the rest of the file has, so a no-MPI build fails. Build:
```bash
git clone --depth 1 https://github.com/nest/nest-gpu.git && cd nest-gpu
# guard the unguarded getMPIComm() (wrap it in #ifdef HAVE_MPI ... #endif), then:
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=<repo>/nestgpu/install \
      -Dwith-gpu-arch=89 -Dwith-mpi=OFF -Dwith-python=ON   # 89 = L40S/Ada; set your GPU's arch
cmake --build build -j && cmake --install build
```
`run.py` wires up `NESTGPU_LIB` + the module path itself (it re-execs once to put CUDA on the loader path);
the venv just needs `numpy`. Standard NEST GPU has no `uniform` init, so v0 uses its GPU-RNG `normal` (the
steady-state rate is v0-invariant). Both GPU sims need a CUDA toolkit; set `CUDA_PATH` to match your GPU.

**SpikingNeuralNetworks.jl** (JuliaSNN) is a pure-Julia simulator, mapped onto built-in types only (no
custom model code). The spec's `C, gL, …` feed an `AdExParameter` (which derives `τm = C/gL`, `R = 1/gL`);
the synapse is the current-based `CurrentSynapse` (CUBA: its membrane injection is voltage-independent),
and a single recurrent `SpikingSynapse` carries the signed splitmix64 weight matrix. All edges route into
one `:glu` channel, so its `ge` accumulator becomes the spec's single signed current accumulator (E: +GE,
I: −GI); JuliaSNN's membrane adds `−R·syn_curr = +R·ge`, matching the spec's `+R·I_syn`. The constant drive
is the population's `I` field, and the 2-step conduction delay is a fixed `delay_dist`. Two implementation
details differ from the spec without changing the model: JuliaSNN fires at a hardwired 0 mV cutoff (the spec
uses `Vpeak = −40 mV`, but the AdEx exponential diverges super-exponentially, so spike times shift by ≪ dt),
and its adaptive-threshold machinery reduces to a constant when `PostSpike.At = 0`. Cross-validated at
N=8000: 37.8 Hz / CV-ISI 0.13, matching NEST and Dewdrop. It is **CPU-only** (JuliaSNN has no GPU engine)
and single-threaded; the sweep is capped at N=128000 because the delayed synapse's per-neuron event lists
make it the slowest CPU entry. `mem_mb` is the resident network footprint (RSS delta across the build), not
`Sys.maxrss`, since loading JuliaSNN pulls Makie/CUDA and spikes the process peak to ~1 GB. It has its OWN
Julia project (`snn/Project.toml`, independent of Dewdrop's); set it up with
`julia +1.12 --project=snn -e 'import Pkg; Pkg.instantiate()'`.

## Adding a simulator (e.g. NEURON, GeNN)

1. `mkdir mysim/` with a `run.<lang>` that reads `../spec.toml`, builds the network with the splitmix64
   connectome (copy the ~10-line algorithm), runs the size sweep, and writes
   `mysim/out/values.csv` + `mysim/out/performance.csv` in the schema above.
2. `./compare_simulators.jl` auto-discovers `mysim/out/` and includes it: no change to the driver.

Output artifacts (`out/`, `.venv/`, build dirs) are gitignored; only the scripts + `spec.toml` are tracked.
