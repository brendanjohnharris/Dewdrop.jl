"""
BrainPy implementation of the shared simulator-comparison problem (../spec.toml). Python-only: no
Julia here; it just generates the standard data CSVs that ../compare_simulators.jl reads. Standalone:

    python run.py                  # every device, write out/values.csv + out/performance.csv
    python run.py <cpu|gpu> <N>    # a single throughput config (subprocess; sets JAX_PLATFORMS)
    python run.py correctness      # the statistics run

The connectome is the language-agnostic splitmix64 fixed-in-degree graph from the spec, byte-identical
to every other simulator's. Requires the brainpy env (brainpy + jax[cuda12]); GPU also needs nvcc via
BRAINEVENT_NVCC_PATH (see README).
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
    import brainpy as bp
    NE = int(NW["ne_frac"] * N)
    pre, post = connectome(N, NW["K"], NW["seed"])
    is_e = pre < NE
    # AdEx (BrainPy AdExIF): R = 1/gL, tau = C/gL
    adex = dict(V_rest=NN["EL"], V_reset=NN["Vr"], V_th=NN["Vpeak"], V_T=NN["VT"], delta_T=NN["dT"],
                a=NN["a"], b=NN["b"], tau=NN["C"] / NN["gL"], tau_w=NN["tauw"], R=1.0 / NN["gL"])

    class EINet(bp.DynSysGroup):
        def __init__(self):
            super().__init__()
            self.neu = bp.neurons.AdExIF(N, V_initializer=bp.init.Uniform(-70.0, -50.0), **adex)
            self.synE = bp.synapses.Exponential(self.neu, self.neu, bp.conn.IJConn(i=pre[is_e], j=post[is_e]),
                                                output=bp.synouts.CUBA(), g_max=SY["GE"], tau=SY["tau"],
                                                delay_step=SY["delay_steps"])
            self.synI = bp.synapses.Exponential(self.neu, self.neu, bp.conn.IJConn(i=pre[~is_e], j=post[~is_e]),
                                                output=bp.synouts.CUBA(), g_max=-SY["GI"], tau=SY["tau"],
                                                delay_step=SY["delay_steps"])

        def update(self):
            self.synE(); self.synI()
            self.neu.input += SY["I_ext"]
            self.neu()
            return self.neu.spike.value

    return EINet(), len(pre)


def _rss_mb():
    return int(open("/proc/self/statm").read().split()[1]) * 4096 / 1e6


def _gpu_proc_mem_mb():
    """This process's GPU *device* memory (MB) via nvidia-smi (needs XLA_PYTHON_CLIENT_PREALLOCATE=false)."""
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
    import brainpy as bp
    import brainpy.math as bm
    bm.set_dt(DT)
    base = _rss_mb()
    net, ne = make_net(N)
    runner = bp.DSRunner(net, monitors=[], dt=DT, progress_bar=False)
    runner.run(P["T"])                                       # warmup (JIT this scan length)
    _ = np.asarray(net.neu.V.value)
    t0 = time.perf_counter()
    runner.run(P["T"])
    _ = np.asarray(net.neu.V.value)
    wall = time.perf_counter() - t0
    mem = _gpu_proc_mem_mb() if device == "gpu" else max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 - base, 0.0)
    print(f"RESULT brainpy jax {device} {N} {wall} {mem} {ne}")


def cv_isi(spk, dt):
    cvs = []
    for j in range(spk.shape[1]):
        ts = np.flatnonzero(spk[:, j]) * dt
        if len(ts) >= 3:
            isi = np.diff(ts)
            cvs.append(isi.std() / isi.mean())
    return float(np.mean(cvs)) if cvs else 0.0


def run_correctness():
    import brainpy as bp
    import brainpy.math as bm
    bm.set_dt(DT)
    N = P["N_correctness"]
    net, ne = make_net(N)
    runner = bp.DSRunner(net, monitors={"spike": net.neu.spike}, dt=DT, progress_bar=False)
    runner.run(P["T_correctness"])
    spk = np.asarray(runner.mon["spike"])
    rate = spk.sum() / (N * P["T_correctness"] / 1000.0)
    cv = cv_isi(spk, DT)
    print(f"CORR {N} {rate} {cv} {ne}")


def main():
    os.makedirs(OUT, exist_ok=True)
    perf = []
    for device in ("cpu", "gpu"):
        platform = "cuda" if device == "gpu" else "cpu"      # jax names the GPU backend 'cuda', not 'gpu'
        env = dict(os.environ, JAX_PLATFORMS=platform)       # force the jax platform in the subprocess
        if device == "gpu":
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"   # so nvidia-smi sees the network, not jax's 75% grab
        for N in P["Ns"]:
            try:
                out = subprocess.run([sys.executable, __file__, device, str(N)], cwd=HERE,
                                     capture_output=True, text=True, env=env).stdout
            except Exception:
                continue
            for line in out.splitlines():
                if line.startswith("RESULT"):
                    _, sim, bk, dev, n, w, m, ne = line.split()
                    perf.append((sim, bk, dev, n, w, m))
                    print(f"  [brainpy/{bk}/{dev}] N={n}  {float(w):.3f}s  {float(m):.1f}MB")
    with open(os.path.join(OUT, "performance.csv"), "w") as f:
        f.write("simulator,backend,device,N,wall_s,mem_mb\n")
        for r in perf:
            f.write(",".join(r) + "\n")
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    corr = subprocess.run([sys.executable, __file__, "correctness"], cwd=HERE,
                          capture_output=True, text=True, env=env).stdout
    for line in corr.splitlines():
        if line.startswith("CORR"):
            _, n, rate, cv, ne = line.split()
            with open(os.path.join(OUT, "values.csv"), "w") as f:
                f.write("N,rate_hz,cv_isi,nedges\n")
                f.write(f"{n},{rate},{cv},{ne}\n")
            print(f"  [brainpy/values] N={n} rate={float(rate):.1f}Hz CV={float(cv):.2f}")
    print(f"wrote {OUT}/{{performance,values}}.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "correctness":
        run_correctness()
    else:
        run_single(sys.argv[1], int(sys.argv[2]))
