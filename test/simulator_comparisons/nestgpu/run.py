"""
NEST GPU implementation of the shared simulator-comparison problem (../spec.toml). This is "NEST with a
GPU": NEST GPU (https://github.com/nest/nest-gpu) is NEST's own CUDA engine, with a PyNEST-like API and the
SAME `aeif_psc_exp` AdEx model as NEST CPU. It therefore reports `simulator=nest, device=gpu` so the
comparison plots it as the GPU variant of the NEST line. Python-only; writes the standard data CSVs.

    python run.py                  # the (gpu) sweep → out/values.csv + out/performance.csv
    python run.py gpu <N>          # a single throughput config (used internally per subprocess)
    python run.py correctness      # the statistics run

Built from source under ./install (CMake + CUDA, sm_89); this script wires up its NESTGPU_LIB + module path
itself. The connectome is the language-agnostic splitmix64 graph, byte-identical to every other simulator's.
"""
import os
import sys

# --- self-contained env: locate the source-built NEST GPU + put CUDA on the loader path (re-exec once) ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_INSTALL = os.path.join(_HERE, "install")
os.environ.setdefault("NESTGPU_LIB", os.path.join(_INSTALL, "lib64", "nestgpu", "libnestgpukernel.so"))
sys.path.insert(0, os.path.join(_INSTALL, "lib64", "python3.11", "site-packages"))
_CUDA = os.environ.get("CUDA_PATH", "/usr/local/cuda-12.9") + "/lib64"
if _CUDA not in os.environ.get("LD_LIBRARY_PATH", "") and not os.environ.get("_NESTGPU_REEXEC"):
    os.environ["LD_LIBRARY_PATH"] = _CUDA + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["_NESTGPU_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)             # loader reads LD_LIBRARY_PATH at start

import time
import resource
import subprocess

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

OUT = os.path.join(_HERE, "out")
with open(os.path.join(_HERE, "..", "spec.toml"), "rb") as f:
    SPEC = tomllib.load(f)
P, NW, NN, SY = SPEC["problem"], SPEC["network"], SPEC["neuron"], SPEC["synapse"]
DT = P["dt"]
MASK = (1 << 64) - 1
# aeif_psc_exp ≡ the spec's AdEx (identical mapping to NEST CPU); I_e is the constant drive I_ext
_PARAMS = dict(C_m=NN["C"], g_L=NN["gL"], E_L=NN["EL"], V_th=NN["VT"], Delta_T=NN["dT"],
               V_reset=NN["Vr"], V_peak=NN["Vpeak"], a=NN["a"], b=NN["b"], tau_w=NN["tauw"],
               t_ref=NN["tref"], I_e=SY["I_ext"], tau_syn_ex=SY["tau"], tau_syn_in=SY["tau"])


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


def make_net(N):
    import nestgpu as ngpu
    ngpu.SetKernelStatus({"time_resolution": DT, "rnd_seed": int(NW["seed"])})
    neurons = ngpu.Create("aeif_psc_exp", N)
    ngpu.SetStatus(neurons, _PARAMS)
    # heterogeneous v0 (breaks initial synchrony); NEST GPU sets a scalar var per-node via its GPU RNG
    # "normal" distribution (it has no "uniform") --- centred in the spec's [-70,-50] band. The steady-state
    # rate is an attractor (validated invariant to v0 across all simulators), so this only shapes the transient.
    ngpu.SetStatus(neurons, {"V_m": {"distribution": "normal", "mu": -60.0, "sigma": 5.0}})
    pre, post = connectome(N, NW["K"], NW["seed"])
    NE = int(NW["ne_frac"] * N)
    i0 = neurons.i0
    delay = SY["delay_steps"] * DT
    # signed weight routes by sign onto tau_syn_ex/in (both = tau); split E/I → constant weight per Connect
    for mask, g in ((pre < NE, SY["GE"]), (pre >= NE, -SY["GI"])):
        src = (pre[mask] + i0).tolist()
        tgt = (post[mask] + i0).tolist()
        ngpu.Connect(src, tgt, {"rule": "one_to_one"}, {"weight": float(g), "delay": float(delay)})
    return neurons, int(pre.size)


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
    import nestgpu as ngpu
    make_net(N)
    ngpu.Simulate(P["T"])                                          # warmup (one-time GPU build/calibrate)
    best = float("inf")
    for _ in range(2):                                            # min-of-reps: pure simulation
        t0 = time.perf_counter()
        ngpu.Simulate(P["T"])
        best = min(best, time.perf_counter() - t0)
    mem = _gpu_proc_mem_mb()                                       # GPU device memory (network resident)
    ne = N * int(NW["K"])
    print(f"RESULT nest nest {device} {N} {best} {mem} {ne}")


def cv_isi_from_rec(spike_lists):
    cvs = []
    for ts in spike_lists:
        ts = np.asarray(ts)
        if ts.size >= 3:
            isi = np.diff(np.sort(ts))
            cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else 0.0


def run_correctness():
    import nestgpu as ngpu
    N = P["N_correctness"]
    neurons, ne = make_net(N)
    ngpu.ActivateRecSpikeTimes(neurons, 2000)                     # per-neuron spike-time buffer
    ngpu.Simulate(P["T_correctness"])
    spikes = ngpu.GetRecSpikeTimes(neurons)                       # list (per neuron) of spike-time arrays
    total = sum(len(np.asarray(s)) for s in spikes)
    rate = total / (N * P["T_correctness"] / 1000.0)
    print(f"CORR {N} {rate} {cv_isi_from_rec(spikes)} {ne}")


def main():
    os.makedirs(OUT, exist_ok=True)
    perf = []
    for device in ("gpu",):
        for N in P["Ns"]:
            try:
                out = subprocess.run([sys.executable, __file__, device, str(N)], cwd=_HERE,
                                     capture_output=True, text=True).stdout
            except Exception:
                continue
            for line in out.splitlines():
                if line.startswith("RESULT"):
                    _, sim, bk, dev, n, w, m, ne = line.split()
                    perf.append((sim, bk, dev, n, w, m))
                    print(f"  [nestgpu/{dev}] N={n}  {float(w):.3f}s  {float(m):.1f}MB")
    with open(os.path.join(OUT, "performance.csv"), "w") as f:
        f.write("simulator,backend,device,N,wall_s,mem_mb\n")
        for r in perf:
            f.write(",".join(r) + "\n")
    corr = subprocess.run([sys.executable, __file__, "correctness"], cwd=_HERE,
                          capture_output=True, text=True).stdout
    for line in corr.splitlines():
        if line.startswith("CORR"):
            _, n, rate, cv, ne = line.split()
            with open(os.path.join(OUT, "values.csv"), "w") as f:
                f.write("N,rate_hz,cv_isi,nedges\n")
                f.write(f"{n},{rate},{cv},{ne}\n")
            print(f"  [nestgpu/values] N={n} rate={float(rate):.1f}Hz CV={float(cv):.2f}")
    print(f"wrote {OUT}/{{performance,values}}.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "correctness":
        run_correctness()
    else:
        run_single(sys.argv[1], int(sys.argv[2]))
