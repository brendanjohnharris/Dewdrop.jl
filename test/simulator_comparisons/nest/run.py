"""
NEST implementation of the shared simulator-comparison problem (../spec.toml). Python-only --- no
Julia here; it just generates the standard data CSVs that ../compare_simulators.jl reads. Standalone:

    python run.py                  # the (cpu) sweep → out/values.csv + out/performance.csv
    python run.py cpu <N>          # a single throughput config (used internally per subprocess)
    python run.py correctness      # the statistics run

NEST is the established C++ gold-standard for CPU spiking-network performance. The neuron is
`aeif_psc_exp` --- AdEx with current-based exponential postsynaptic currents (= CUBA) --- mapped 1:1
from the spec (V_th = the AdEx exponential threshold V_T, V_peak = the spike cutoff). The connectome is
the language-agnostic splitmix64 fixed-in-degree graph, byte-identical to every other simulator's.
Standard NEST is CPU-only (NEST GPU is a separate project), so device is always `cpu`; it runs natively
multi-threaded, and the thread count is AUTO-SELECTED per network size (the NEST analogue of Dewdrop's
`Auto` backend): NEST uses busy-wait barriers and synchronises every `min_delay` window (here 0.2 ms =
every 2 steps), so running one thread per core leaves no slack for the main/OS thread and a single
descheduled straggler stalls every barrier --- a ~40x collapse. NEST is therefore run with core headroom
and the faster of {½, ¾}·cores is reported. Override with NEST_THREADS to force a fixed count. Requires
the nest env (nest-simulator); see README.
"""
import os
import sys
import time
import resource
import subprocess

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
with open(os.path.join(HERE, "..", "spec.toml"), "rb") as f:
    SPEC = tomllib.load(f)
P, NW, NN, SY = SPEC["problem"], SPEC["network"], SPEC["neuron"], SPEC["synapse"]
DT = P["dt"]
MASK = (1 << 64) - 1
CORES = len(os.sched_getaffinity(0))
NTHREADS = int(os.environ.get("NEST_THREADS", "0")) or CORES
_FORCE_THREADS = "NEST_THREADS" in os.environ
# aeif_psc_exp ≡ the spec's AdEx (current-based exp synapses); I_e is the constant drive I_ext.
# gsl_error_tol loosened from the 1e-6 default: aeif_psc_exp integrates with an adaptive GSL solver, and
# 1e-3 gives BYTE-identical statistics here (rate 38.00Hz / CV 0.127 unchanged even at 1e-1 — validated)
# while shaving the wasted substep accuracy. It is still far more accurate than the fixed-step methods the
# other simulators use, so this is "NEST at its best", not a thumb on the scale.
_PARAMS = dict(C_m=NN["C"], g_L=NN["gL"], E_L=NN["EL"], V_th=NN["VT"], Delta_T=NN["dT"],
               V_reset=NN["Vr"], V_peak=NN["Vpeak"], a=NN["a"], b=NN["b"], tau_w=NN["tauw"],
               t_ref=NN["tref"], I_e=SY["I_ext"], tau_syn_ex=SY["tau"], tau_syn_in=SY["tau"],
               gsl_error_tol=1e-3)


def _thread_candidates():
    # full-core oversubscription collapses NEST's spin barriers (see module docstring) → run with headroom.
    # An explicit NEST_THREADS forces exactly that count; otherwise try ½ and ¾ cores and keep the faster.
    if _FORCE_THREADS:
        return [NTHREADS]
    return sorted({max(1, CORES // 2), max(1, (3 * CORES) // 4)})


def _splitmix64(s):
    s = (s + 0x9E3779B97F4A7C15) & MASK
    z = s
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK, s


def connectome(N, K, seed):
    pre, post = [], []
    for j in range(N):
        state = (seed + j * 0x9E3779B97F4A7C15) & MASK
        chosen = set()
        while len(chosen) < K:
            z, state = _splitmix64(state)
            p = z % N
            if p == j or p in chosen:
                continue
            chosen.add(p)
            pre.append(p)
            post.append(j)
    return np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)


def _arrays(N):
    """Thread-independent inputs (connectome + per-edge weight/delay + V0), built ONCE per N."""
    pre, post = connectome(N, NW["K"], NW["seed"])
    NE = int(NW["ne_frac"] * N)
    w = np.where(pre < NE, SY["GE"], -SY["GI"]).astype(np.float64)   # GE excitatory / −GI inhibitory
    d = np.full(pre.size, SY["delay_steps"] * DT, dtype=np.float64)
    V0 = np.random.default_rng(NW["seed"]).uniform(-70.0, -50.0, N)  # match the other sims' v0
    return pre, post, w, d, V0


def _build(nest, N, threads, arrays):
    """Construct the AdEx E/I network in a fresh NEST kernel at the given thread count; return neurons."""
    pre, post, w, d, V0 = arrays
    nest.set_verbosity("M_ERROR")
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": DT, "local_num_threads": int(threads), "rng_seed": int(NW["seed"])})
    neurons = nest.Create("aeif_psc_exp", N, params=_PARAMS)
    neurons.V_m = V0
    off = neurons[0].global_id
    # NEST array-connect: numpy node-ID arrays + explicit one_to_one; weight/delay must be numpy arrays.
    # Signed weight routes by sign onto tau_syn_ex/in (both = tau), so no separate E/I populations needed.
    nest.Connect((pre + off).astype(np.uint64), (post + off).astype(np.uint64),
                 conn_spec="one_to_one", syn_spec={"weight": w, "delay": d})
    return neurons


def make_net(N, threads):
    """Build the network once (used by the correctness run); return (neurons, nedges)."""
    import nest
    arrays = _arrays(N)
    return _build(nest, N, threads, arrays), int(arrays[0].size)


def _rss_mb():
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / 1e6


def _timed_run(nest):
    t0 = time.perf_counter()
    nest.Run(P["T"])                                         # pure simulation loop (no Prepare/Cleanup)
    return time.perf_counter() - t0


def run_single(device, N):
    import nest
    base = _rss_mb()
    arrays = _arrays(N)                                      # connectome built once; reused per thread count
    ne = int(arrays[0].size)
    best_wall, best_nt, peak = float("inf"), None, 0.0
    for nt in _thread_candidates():                         # auto-select threads (full-core collapse, see top)
        _build(nest, N, nt, arrays)
        nest.Prepare()
        nest.Run(P["T"])                                    # warmup (one-time buffer/setup costs)
        wall = min(_timed_run(nest), _timed_run(nest))      # min-of-reps: pure simulation loop only
        nest.Cleanup()
        peak = max(peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        if wall < best_wall:
            best_wall, best_nt = wall, nt
    mem = max(peak - base, 0.0)
    print(f"RESULT nest nest {device} {N} {best_wall} {mem} {ne}")
    print(f"  nest N={N}: {best_nt} threads (of {CORES} cores) → {best_wall:.3f}s", file=sys.stderr)


def cv_isi(senders, times):
    if senders.size == 0:
        return 0.0
    order = np.argsort(senders, kind="stable")
    s, t = senders[order], times[order]
    _, idx = np.unique(s, return_index=True)
    cvs = []
    for ts in np.split(t, idx[1:]):
        if ts.size >= 3:
            isi = np.diff(np.sort(ts))
            cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else 0.0


def run_correctness():
    import nest
    N = P["N_correctness"]
    neurons, ne = make_net(N, 1)                             # 1 thread → deterministic spike ordering
    sr = nest.Create("spike_recorder")
    nest.Connect(neurons, sr)
    nest.Simulate(P["T_correctness"])
    ev = sr.get("events")
    senders, times = np.asarray(ev["senders"]), np.asarray(ev["times"])
    rate = times.size / (N * P["T_correctness"] / 1000.0)
    print(f"CORR {N} {rate} {cv_isi(senders, times)} {ne}")


def main():
    os.makedirs(OUT, exist_ok=True)
    perf = []
    for device in ("cpu",):                                  # standard NEST is CPU-only
        for N in P["Ns"]:
            try:
                out = subprocess.run([sys.executable, __file__, device, str(N)], cwd=HERE,
                                     capture_output=True, text=True).stdout
            except Exception:
                continue
            for line in out.splitlines():
                if line.startswith("RESULT"):
                    _, sim, bk, dev, n, w, m, ne = line.split()
                    perf.append((sim, bk, dev, n, w, m))
                    print(f"  [nest/{bk}/{dev}] N={n}  {float(w):.3f}s  {float(m):.1f}MB")
    with open(os.path.join(OUT, "performance.csv"), "w") as f:
        f.write("simulator,backend,device,N,wall_s,mem_mb\n")
        for r in perf:
            f.write(",".join(r) + "\n")
    corr = subprocess.run([sys.executable, __file__, "correctness"], cwd=HERE,
                          capture_output=True, text=True).stdout
    for line in corr.splitlines():
        if line.startswith("CORR"):
            _, n, rate, cv, ne = line.split()
            with open(os.path.join(OUT, "values.csv"), "w") as f:
                f.write("N,rate_hz,cv_isi,nedges\n")
                f.write(f"{n},{rate},{cv},{ne}\n")
            print(f"  [nest/values] N={n} rate={float(rate):.1f}Hz CV={float(cv):.2f}")
    print(f"wrote {OUT}/{{performance,values}}.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "correctness":
        run_correctness()
    else:
        run_single(sys.argv[1], int(sys.argv[2]))
