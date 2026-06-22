"""
Brian2 / brian2cuda implementation of the shared simulator-comparison problem (../spec.toml).
Python-only --- no Julia here; this just generates the standard data CSVs that
../compare_simulators.jl reads. Standalone:

    python run.py                  # every device, write out/values.csv + out/performance.csv
    python run.py <cpu|gpu> <N>    # a single throughput config (used internally per subprocess)
    python run.py correctness      # the statistics run

Each config runs in a fresh subprocess (clean standalone build + clean peak memory). The connectome
is the language-agnostic splitmix64 fixed-in-degree graph from the spec, so it is byte-identical to
every other simulator's. Requires the brian env: `uv venv .venv && uv pip install brian2 brian2cuda`.
"""
import os
import sys
import time
import threading
import resource
import subprocess

import numpy as np
import brian2 as b2
from brian2 import ms, mV, pF, nS, pA

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


# --- language-agnostic connectome (splitmix64 fixed in-degree; identical to dewdrop/run.jl) ---
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


def setup_device(device, tag):
    directory = os.path.join(OUT, f"build_{tag}")
    import shutil
    shutil.rmtree(directory, ignore_errors=True)
    if device == "gpu":
        cb = "devices.cuda_standalone.cuda_backend."
        import brian2cuda  # noqa: F401
        b2.prefs[cb + "detect_gpus"] = False                # this node sets CUDA_VISIBLE_DEVICES to a UUID
        b2.prefs[cb + "gpu_id"] = 0
        b2.prefs[cb + "compute_capability"] = 8.9            # L40S (Ada); override for another GPU
        b2.prefs[cb + "detect_cuda"] = False
        b2.prefs[cb + "cuda_path"] = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.9")
        b2.prefs.devices.cpp_standalone.openmp_threads = 0
        b2.set_device("cuda_standalone", directory=directory, build_on_run=False)
    else:
        b2.prefs.codegen.cpp.extra_compile_args_gcc = ["-O3", "-march=native", "-ffast-math"]
        b2.prefs.devices.cpp_standalone.openmp_threads = 0   # single-thread C++ is Brian2's fast SNN path
        b2.set_device("cpp_standalone", directory=directory, build_on_run=False)
    return directory


def build_net(N):
    NE = int(NW["ne_frac"] * N)
    pre, post = connectome(N, NW["K"], NW["seed"])
    eqs = """
    dv/dt = (gL*(EL - v) + gL*dT*exp((v - VT)/dT) + Iext + ge - w) / C : volt
    dge/dt = -ge / tau_syn : amp
    dw/dt = (a*(v - EL) - w) / tauw : amp
    """
    ns = dict(gL=NN["gL"]*nS, EL=NN["EL"]*mV, dT=NN["dT"]*mV, VT=NN["VT"]*mV, C=NN["C"]*pF,
              Iext=SY["I_ext"]*pA, tau_syn=SY["tau"]*ms, a=NN["a"]*nS, tauw=NN["tauw"]*ms)
    G = b2.NeuronGroup(N, eqs, threshold=f"v > {NN['Vpeak']}*mV",
                       reset=f"v = {NN['Vr']}*mV; w += {NN['b']}*pA", method="euler", namespace=ns)
    G.v = "(-70.0 + 20.0*rand()) * mV"
    S = b2.Synapses(G, G, model="w_syn : amp", on_pre="ge_post += w_syn", delay=SY["delay_steps"]*DT*ms)
    S.connect(i=pre, j=post)
    S.w_syn = np.where(pre < NE, SY["GE"], -SY["GI"]) * pA
    return G, S, len(pre)


def _rss_mb():  # current resident set (MB)
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / 1e6


def _gpu_total_used_mb():  # total GPU 0 device memory in use (MB)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        return float(out.strip().split("\n")[0])
    except Exception:
        return 0.0


def run_single(device, N):
    b2.prefs.core.default_float_dtype = np.float32
    base = _rss_mb()
    directory = setup_device(device, f"{device}_N{N}")
    G, S, ne = build_net(N)
    b2.run(P["T"] * ms)
    if device == "gpu":
        # the cuda_standalone binary is a SEPARATE process, so host RSS / per-pid don't see its GPU use.
        # Poll total device memory during the build+run and report the peak above the pre-run baseline.
        gpu_base = _gpu_total_used_mb(); peak = [gpu_base]; stop = threading.Event()
        def _poll():
            while not stop.is_set():
                peak[0] = max(peak[0], _gpu_total_used_mb()); time.sleep(0.02)
        th = threading.Thread(target=_poll, daemon=True); th.start()
        b2.device.build(directory=directory, compile=True, run=True, debug=False, clean=True)
        stop.set(); th.join(timeout=2)
        mem = max(peak[0] - gpu_base, 0.0)
    else:
        b2.device.build(directory=directory, compile=True, run=True, debug=False, clean=True)
        mem = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 - base, 0.0)   # host RSS for CPU
    wall = float(b2.device._last_run_time)                   # pure simulation time (excludes compile)
    backend = "cpp_standalone" if device == "cpu" else "cuda_standalone"
    print(f"RESULT brian {backend} {device} {N} {wall} {mem} {ne}")


def cv_isi(spike_trains, dt):
    cvs = []
    for ts in spike_trains.values():
        if len(ts) >= 3:
            isi = np.diff(np.sort(np.asarray(ts)))
            cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else 0.0


def run_correctness():
    N = P["N_correctness"]
    setup_device("cpu", "corr")
    G, S, ne = build_net(N)
    mon = b2.SpikeMonitor(G)
    b2.run(P["T_correctness"] * ms)
    b2.device.build(directory=os.path.join(OUT, "build_corr"), compile=True, run=True, debug=False, clean=True)
    rate = mon.num_spikes / (N * P["T_correctness"] / 1000.0)
    cv = cv_isi({i: np.asarray(t / ms) for i, t in mon.spike_trains().items()}, DT)
    print(f"CORR {N} {rate} {cv} {ne}")


def main():
    os.makedirs(OUT, exist_ok=True)
    perf = []
    for device in ("cpu", "gpu"):
        for N in P["Ns"]:
            try:
                out = subprocess.run([sys.executable, __file__, device, str(N)], cwd=HERE,
                                     capture_output=True, text=True, env=os.environ).stdout
            except Exception:
                continue
            for line in out.splitlines():
                if line.startswith("RESULT"):
                    _, sim, bk, dev, n, w, m, ne = line.split()
                    perf.append((sim, bk, dev, n, w, m))
                    print(f"  [brian/{bk}/{dev}] N={n}  {float(w):.3f}s  {float(m):.1f}MB")
    with open(os.path.join(OUT, "performance.csv"), "w") as f:
        f.write("simulator,backend,device,N,wall_s,mem_mb\n")
        for r in perf:
            f.write(",".join(r) + "\n")
    corr = subprocess.run([sys.executable, __file__, "correctness"], cwd=HERE,
                          capture_output=True, text=True, env=os.environ).stdout
    for line in corr.splitlines():
        if line.startswith("CORR"):
            _, n, rate, cv, ne = line.split()
            with open(os.path.join(OUT, "values.csv"), "w") as f:
                f.write("N,rate_hz,cv_isi,nedges\n")
                f.write(f"{n},{rate},{cv},{ne}\n")
            print(f"  [brian/values] N={n} rate={float(rate):.1f}Hz CV={float(cv):.2f}")
    print(f"wrote {OUT}/{{performance,values}}.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "correctness":
        run_correctness()
    else:
        run_single(sys.argv[1], int(sys.argv[2]))
