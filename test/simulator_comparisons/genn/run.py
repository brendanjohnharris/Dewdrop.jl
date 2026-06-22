"""
GeNN implementation of the shared simulator-comparison problem (../spec.toml). Python-only --- no Julia
here; it generates the standard data CSVs that ../compare_simulators.jl reads. Standalone:

    python run.py                  # the (gpu) sweep → out/values.csv + out/performance.csv
    python run.py gpu <N>          # a single throughput config (used internally per subprocess)
    python run.py correctness      # the statistics run

GeNN (https://github.com/genn-team/genn) is a GPU code-GENERATOR for spiking networks: it emits + compiles
bespoke CUDA per model, and is among the fastest GPU SNN simulators --- the key GPU rival to Dewdrop's own
GPU backend. The neuron is a CUSTOM AdEx model (GeNN has no built-in AdEx) integrated forward-Euler at the
spec's dt; synapses are `ExpCurr` (current-based exponential = CUBA) with `StaticPulseConstantWeight`, split
into one E population (weight GE) and one I population (weight −GI). The connectome is the language-agnostic
splitmix64 fixed-in-degree graph, byte-identical to every other simulator's, supplied as an explicit edge
list via `set_sparse_connections`. Requires the genn env (pygenn) + a CUDA toolkit (CUDA_PATH); see README.
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

os.environ.setdefault("CUDA_PATH", "/usr/local/cuda-12.9")        # GeNN needs nvcc to compile generated code

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
with open(os.path.join(HERE, "..", "spec.toml"), "rb") as f:
    SPEC = tomllib.load(f)
P, NW, NN, SY = SPEC["problem"], SPEC["network"], SPEC["neuron"], SPEC["synapse"]
DT = P["dt"]
MASK = (1 << 64) - 1


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


# --- the AdEx neuron as a GeNN custom model (forward-Euler; Iexp capped to stay finite pre-cutoff) ---
def _adex_model():
    import pygenn
    return pygenn.create_neuron_model(
        "AdEx",
        params=["C", "gL", "EL", "VT", "dT", "Vr", "Vpeak", "a", "b", "tauw", "Iext"],
        vars=[("V", "scalar"), ("w", "scalar")],
        sim_code="""
        const scalar Iexp = gL * dT * exp(fmin((V - VT) / dT, (scalar)30.0));
        V += dt * (-gL * (V - EL) + Iexp - w + Isyn + Iext) / C;   // V: old w, synaptic + constant drive
        w += dt * (a * (V - EL) - w) / tauw;                       // w: new V (symplectic order)
        """,
        threshold_condition_code="V >= Vpeak",
        reset_code="V = Vr; w += b;",
    )


def build_model(N, record):
    import pygenn
    model = pygenn.GeNNModel("float", f"adex_{N}_{'rec' if record else 'perf'}")
    model.dt = DT
    params = {"C": NN["C"], "gL": NN["gL"], "EL": NN["EL"], "VT": NN["VT"], "dT": NN["dT"],
              "Vr": NN["Vr"], "Vpeak": NN["Vpeak"], "a": NN["a"], "b": NN["b"], "tauw": NN["tauw"],
              "Iext": SY["I_ext"]}
    V0 = np.random.default_rng(NW["seed"]).uniform(-70.0, -50.0, N).astype(np.float32)
    pop = model.add_neuron_population("pop", N, _adex_model(), params, {"V": 0.0, "w": 0.0})
    pop.spike_recording_enabled = record
    pre, post = connectome(N, NW["K"], NW["seed"])
    NE = int(NW["ne_frac"] * N)
    is_e = pre < NE
    for name, mask, g in (("E", is_e, SY["GE"]), ("I", ~is_e, -SY["GI"])):
        sp = model.add_synapse_population(
            name, "SPARSE", pop, pop,
            pygenn.init_weight_update("StaticPulseConstantWeight", {"g": float(g)}),
            pygenn.init_postsynaptic("ExpCurr", {"tau": SY["tau"]}))
        sp.set_sparse_connections(pre[mask].astype(np.uint32), post[mask].astype(np.uint32))
        sp.axonal_delay_steps = int(SY["delay_steps"])
    return model, pop, int(pre.size), V0


def _set_v0(pop, V0):
    pop.vars["V"].view[:] = V0                                    # match the other sims' uniform v0
    pop.vars["V"].push_to_device()


def _gpu_proc_mem_mb():
    """This process's GPU *device* memory (MB) via nvidia-smi (the network's GPU footprint, not host RSS)."""
    pid = os.getpid()
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                return float(parts[1])
    except Exception:
        pass
    return 0.0


def run_single(device, N):
    nsteps = round(P["T"] / DT)
    model, pop, ne, V0 = build_model(N, record=False)
    model.build()
    model.load()
    _set_v0(pop, V0)
    for _ in range(nsteps):                                       # warmup (first-touch / page-in)
        model.step_time()
    best = float("inf")
    for _ in range(2):                                            # min-of-reps: pure step loop
        t0 = time.perf_counter()
        for _ in range(nsteps):
            model.step_time()
        best = min(best, time.perf_counter() - t0)
    mem = _gpu_proc_mem_mb()                                      # GPU device memory (network resident)
    print(f"RESULT genn cuda {device} {N} {best} {mem} {ne}")


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
    N = P["N_correctness"]
    nsteps = round(P["T_correctness"] / DT)
    model, pop, ne, V0 = build_model(N, record=True)
    model.build()
    model.load(num_recording_timesteps=nsteps)
    _set_v0(pop, V0)
    for _ in range(nsteps):
        model.step_time()
    model.pull_recording_buffers_from_device()
    times, ids = pop.spike_recording_data[0]
    times, ids = np.asarray(times), np.asarray(ids)
    rate = times.size / (N * P["T_correctness"] / 1000.0)
    print(f"CORR {N} {rate} {cv_isi(ids, times)} {ne}")


def main():
    os.makedirs(OUT, exist_ok=True)
    perf = []
    for device in ("gpu",):                                       # GeNN here targets the CUDA backend
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
                    print(f"  [genn/{bk}/{dev}] N={n}  {float(w):.3f}s  {float(m):.1f}MB")
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
            print(f"  [genn/values] N={n} rate={float(rate):.1f}Hz CV={float(cv):.2f}")
    print(f"wrote {OUT}/{{performance,values}}.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "correctness":
        run_correctness()
    else:
        run_single(sys.argv[1], int(sys.argv[2]))
