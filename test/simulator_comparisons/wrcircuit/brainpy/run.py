#!/usr/bin/env python
"""
WRCircuit (BrainPy) reference export for the Dewdrop reproduction.

Builds the *real* `WRCircuit.jl` spatial FNS E/I network (Spatial model) on a small, seeded
network, runs it, and writes everything Dewdrop needs to reproduce the run BIT-FOR-BIT given
identical structure: the six projection connectomes (+ per-edge weights + delays), the neuron
parameters, the initial membrane potentials, the external Poisson spike raster, and the recorded
traces (V, synaptic input current, spikes). Cross-language IO is plain CSV (read with Julia's
DelimitedFiles), matching the other simulator_comparisons dirs.

Two deliberate, documented choices make an EXACT cross-simulator comparison possible:
  * float64 everywhere (`bm.enable_x64`)  -- removes float32 RNG/precision as a confound.
  * constant synaptic rise time `tau_r`   -- Spatial randomises tau_r per *source* (align-pre);
    a per-source rise time is the one feature that breaks Dewdrop's (equivalent, for a linear
    filter) per-target dual-exponential. We pin it constant via a localised monkeypatch of
    `bp.init.Normal` (its ONLY use in Spatial construction is the tau_r initialiser), WITHOUT
    editing the WRCircuit source.

Run:  python run.py            # build + run + export to ./out
"""
import os
import sys
import json

import numpy as np

# --- locate the (unmodified) WRCircuit package -------------------------------------------------
WRC_ROOT = os.environ.get("WRC_ROOT", "/import/taiji1/bhar9988/code/DDC/WorkingRegime.jl/WRCircuit.jl")
sys.path.insert(0, WRC_ROOT)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# --- small, seeded network parameters (overridable via env for sweeps) -------------------------
SEED = int(os.environ.get("WRC_SEED", "20"))
DT = float(os.environ.get("WRC_DT", "0.1"))            # ms
T = float(os.environ.get("WRC_T", "400.0"))           # ms (<30 s on CPU at this size)
RHO = float(os.environ.get("WRC_RHO", "400.0"))       # → ne_grid = round(sqrt(rho*dx^2))
DX = float(os.environ.get("WRC_DX", "0.6"))           # rho=400, dx=0.6 → ne=12 → NE=144, NI=36
NU = float(os.environ.get("WRC_NU", "15.0"))          # external population rate (Hz)
N_EXT = int(os.environ.get("WRC_NEXT", "40"))
# reduced in-degrees so num_connections = K*N stays < N_pre*N_post for the small grid
K_EE = int(os.environ.get("WRC_KEE", "20"))
K_EI = int(os.environ.get("WRC_KEI", "25"))
K_IE = int(os.environ.get("WRC_KIE", "20"))
K_II = int(os.environ.get("WRC_KII", "25"))
DELTA_GK = float(os.environ.get("WRC_DGK", "0.005"))  # E adaptation strength (bumped a touch for visibility)


def main():
    import brainpy as bp
    import brainpy.math as bm

    bm.enable_x64()                  # float64 both sides → exact comparison
    bm.set_dt(DT)

    # --- monkeypatch: constant tau_r (Normal → its mean). Only used by Spatial for tau_r. --------
    _OrigNormal = bp.init.Normal

    def _ConstMean(mean, *args, **kwargs):
        return bp.init.Constant(mean)
    bp.init.Normal = _ConstMean
    try:
        from src.models.Spatial import Spatial
        model = Spatial(
            rho=RHO, dx=DX, gamma=4,
            K_ee=K_EE, K_ei=K_EI, K_ie=K_IE, K_ii=K_II,
            nu=NU, n_ext=N_EXT, Delta_g_K=DELTA_GK,
            key=SEED,
        )
    finally:
        bp.init.Normal = _OrigNormal

    NE = int(np.prod(model.E.size))
    NI = int(np.prod(model.I.size))
    N_ext = int(model.ext.num)
    nsteps = int(round(T / DT))
    print(f"NE={NE} NI={NI} N_ext={N_ext} nsteps={nsteps}")

    # --- positions (E grid, I random) -----------------------------------------------------------
    posE = np.asarray(model.E.positions, dtype=np.float64)   # (NE, 2)
    posI = np.asarray(model.I.positions, dtype=np.float64)   # (NI, 2)
    np.savetxt(os.path.join(OUT, "posE.csv"), posE, delimiter=",")
    np.savetxt(os.path.join(OUT, "posI.csv"), posI, delimiter=",")

    # --- initial membrane potentials (read after a clean reset; deterministic given SEED) -------
    bp.reset_state(model)
    v0E = np.asarray(model.E.V.value, dtype=np.float64).reshape(-1)
    v0I = np.asarray(model.I.V.value, dtype=np.float64).reshape(-1)
    np.savetxt(os.path.join(OUT, "v0E.csv"), v0E, delimiter=",")
    np.savetxt(os.path.join(OUT, "v0I.csv"), v0I, delimiter=",")

    # --- connectomes (CSR per projection → (pre_local, post_local, weight) edge lists) ----------
    def export_edges(name, proj):
        comm = proj.proj.comm
        indptr = np.asarray(comm.indptr).astype(np.int64)    # length pre_num+1
        indices = np.asarray(comm.indices).astype(np.int64)  # post ids, length nedges
        weight = np.asarray(comm.weight).astype(np.float64)
        if weight.ndim == 0 or weight.size == 1:             # scalar g_max broadcast
            weight = np.full(indices.shape, float(weight))
        pre = np.empty(indices.shape[0], dtype=np.int64)
        for i in range(indptr.shape[0] - 1):
            pre[indptr[i]:indptr[i + 1]] = i
        arr = np.column_stack([pre, indices, weight])        # (nedges, 3): pre, post, weight
        np.savetxt(os.path.join(OUT, f"edges_{name}.csv"), arr, delimiter=",")
        return arr.shape[0]

    projs = {"E2E": model.E2E, "E2I": model.E2I, "I2E": model.I2E, "I2I": model.I2I,
             "ext2E": model.ext2E, "ext2I": model.ext2I}
    nedges = {nm: export_edges(nm, p) for nm, p in projs.items()}
    print("nedges:", nedges)

    # --- scalar parameters / per-projection kinetics --------------------------------------------
    def delay_steps(d):
        return int(round(d / DT))

    meta = dict(
        seed=SEED, dt=DT, T=T, nsteps=nsteps, NE=NE, NI=NI, N_ext=N_ext,
        rho=RHO, dx=DX, nu=NU,
        neuron_E=dict(C=float(model.E.C), g_L=float(model.E.g_L), V_L=float(model.E.V_L),
                      V_K=float(model.E.V_K), V_th=float(model.E.V_th), V_rt=float(model.E.V_rt),
                      tau_ref=float(model.E.tau_ref), tau_K=float(model.E.tau_K),
                      Delta_g_K=float(model.E.Delta_g_K)),
        neuron_I=dict(C=float(model.I.C), g_L=float(model.I.g_L), V_L=float(model.I.V_L),
                      V_K=float(model.I.V_K), V_th=float(model.I.V_th), V_rt=float(model.I.V_rt),
                      tau_ref=float(model.I.tau_ref), tau_K=float(model.I.tau_K),
                      Delta_g_K=float(model.I.Delta_g_K)),
        # per projection: (pre_pop, post_pop, tau_r, tau_d, Erev, delay_steps)
        proj=dict(
            E2E=dict(pre="E", post="E", tau_r=float(model.tau_r_e), tau_d=float(model.tau_d_e),
                     Erev=float(model.V_rev_e), delay=delay_steps(model.e_delay)),
            E2I=dict(pre="E", post="I", tau_r=float(model.tau_r_e), tau_d=float(model.tau_d_e),
                     Erev=float(model.V_rev_e), delay=delay_steps(model.e_delay)),
            I2E=dict(pre="I", post="E", tau_r=float(model.tau_r_i), tau_d=float(model.tau_d_i),
                     Erev=float(model.V_rev_i), delay=delay_steps(model.i_delay)),
            I2I=dict(pre="I", post="I", tau_r=float(model.tau_r_i), tau_d=float(model.tau_d_i),
                     Erev=float(model.V_rev_i), delay=delay_steps(model.i_delay)),
            ext2E=dict(pre="ext", post="E", tau_r=float(model.tau_r_e), tau_d=float(model.tau_d_e),
                       Erev=float(model.V_rev_e), delay=delay_steps(model.e_delay)),
            ext2I=dict(pre="ext", post="I", tau_r=float(model.tau_r_e), tau_d=float(model.tau_d_e),
                       Erev=float(model.V_rev_e), delay=delay_steps(model.e_delay)),
        ),
        nedges=nedges,
    )
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # flat CSVs for the Julia side (DelimitedFiles, no JSON dep) --------------------------------
    def write_kv(name, d):
        with open(os.path.join(OUT, name), "w") as f:
            for k, v in d.items():
                f.write(f"{k},{v}\n")
    write_kv("scalars.csv", dict(dt=DT, T=T, nsteps=nsteps, NE=NE, NI=NI, N_ext=N_ext))
    write_kv("neuronE.csv", meta["neuron_E"])
    write_kv("neuronI.csv", meta["neuron_I"])
    with open(os.path.join(OUT, "projmeta.csv"), "w") as f:
        f.write("name,pre,post,tau_r,tau_d,Erev,delay\n")
        for nm, p in meta["proj"].items():
            f.write(f"{nm},{p['pre']},{p['post']},{p['tau_r']},{p['tau_d']},{p['Erev']},{p['delay']}\n")

    # --- run, recording V / input / spikes for E and I, and the external Poisson raster ---------
    runner = bp.DSRunner(
        model,
        monitors={"E.V": model.E.V, "I.V": model.I.V,
                  "E.input": model.E.input, "I.input": model.I.input,
                  "E.spike": model.E.spike, "I.spike": model.I.spike,
                  "E.g_K": model.E.g_K, "ext.spike": model.ext.spike},
        dt=DT, progress_bar=False, numpy_mon_after_run=True,
    )
    runner.run(T)
    mon = runner.mon

    vE = np.asarray(mon["E.V"], dtype=np.float64)            # (nsteps, NE)
    vI = np.asarray(mon["I.V"], dtype=np.float64)
    inE = np.asarray(mon["E.input"], dtype=np.float64)
    inI = np.asarray(mon["I.input"], dtype=np.float64)
    gKE = np.asarray(mon["E.g_K"], dtype=np.float64)
    sE = np.asarray(mon["E.spike"]).astype(bool)            # (nsteps, NE)
    sI = np.asarray(mon["I.spike"]).astype(bool)
    sExt = np.asarray(mon["ext.spike"]).astype(bool)        # (nsteps, N_ext)

    np.savetxt(os.path.join(OUT, "vE.csv"), vE, delimiter=",")
    np.savetxt(os.path.join(OUT, "vI.csv"), vI, delimiter=",")
    np.savetxt(os.path.join(OUT, "inputE.csv"), inE, delimiter=",")
    np.savetxt(os.path.join(OUT, "inputI.csv"), inI, delimiter=",")
    np.savetxt(os.path.join(OUT, "gKE.csv"), gKE, delimiter=",")

    def save_raster(name, spk):
        steps, ids = np.nonzero(spk)                         # 0-based (step, neuron)
        np.savetxt(os.path.join(OUT, name), np.column_stack([steps, ids]).astype(np.int64),
                   delimiter=",", fmt="%d")

    save_raster("spikesE.csv", sE)
    save_raster("spikesI.csv", sI)
    save_raster("ext_spikes.csv", sExt)

    rE = sE.sum() / (NE * T / 1000.0)
    rI = sI.sum() / (NI * T / 1000.0)
    print(f"firing rates: E={rE:.2f} Hz, I={rI:.2f} Hz   (E spikes={sE.sum()}, I spikes={sI.sum()})")
    print(f"wrote reference export to {OUT}")

    # --- optional timing (WRC_BENCH=1): pure run wall time, no monitors, after a warmup --------
    if os.environ.get("WRC_BENCH") == "1":
        import time
        r2 = bp.DSRunner(model, monitors=[], dt=DT, progress_bar=False)
        r2.run(T)                                            # warmup (already JIT-ed, but be safe)
        t0 = time.perf_counter()
        r2.run(T)
        wall = time.perf_counter() - t0
        nedges = sum(meta["nedges"].values())
        print(f"BENCH brainpy N={NE+NI} nedges={nedges} wall={wall:.4f}")


if __name__ == "__main__":
    main()
