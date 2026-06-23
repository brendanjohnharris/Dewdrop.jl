#!/bin/bash
# CPU timing: the existing WRCircuit (BrainPy `Spatial`) vs the Dewdrop reproduction, on the same
# small spatial FNS E/I network at a few sizes. Each simulator builds the network, JIT/compiles on a
# warmup run, then reports the pure run/solve wall time (no recording) via its WRC_BENCH path. The
# BrainPy export is regenerated at each size first (its connectome is what Dewdrop ingests), so both
# time the identical network. Run: bash bench.sh
set -e
WRC=${WRC_ROOT:-/import/taiji1/bhar9988/code/DDC/WorkingRegime.jl/WRCircuit.jl}
PY=$WRC/.CondaPkg/.pixi/envs/default/bin/python
DD=/import/taiji1/bhar9988/code/Dewdrop.jl
RUNPY=$DD/test/simulator_comparisons/wrcircuit/brainpy/run.py
RUNJL=$DD/test/simulator_comparisons/wrcircuit/dewdrop/run.jl

# (rho dx) pairs → NE = round(sqrt(rho*dx^2))^2:  400→144, 1600→576, 2844→1024
SIZES=("400 0.6" "1600 0.6" "2844 0.6")

printf "\n%-8s %-10s %-14s %-14s %-10s\n" "N" "nedges" "BrainPy (s)" "Dewdrop (s)" "speedup"
printf "%s\n" "--------------------------------------------------------------"
for s in "${SIZES[@]}"; do
    read -r rho dx <<<"$s"
    bp=$(cd "$WRC" && WRC_BENCH=1 WRC_RHO="$rho" WRC_DX="$dx" JAX_PLATFORMS=cpu "$PY" "$RUNPY" 2>/dev/null | grep '^BENCH brainpy')
    jl=$(cd "$DD" && WRC_BENCH=1 julia +1.12 -t auto --project=. "$RUNJL" 2>/dev/null | grep '^BENCH dewdrop')
    N=$(echo "$bp" | sed -n 's/.*N=\([0-9]*\).*/\1/p')
    NE=$(echo "$bp" | sed -n 's/.*nedges=\([0-9]*\).*/\1/p')
    bpw=$(echo "$bp" | sed -n 's/.*wall=\([0-9.]*\).*/\1/p')
    jlw=$(echo "$jl" | sed -n 's/.*wall=\([0-9.]*\).*/\1/p')
    sp=$(awk "BEGIN{if($jlw>0)printf \"%.2fx\", $bpw/$jlw; else print \"-\"}")
    printf "%-8s %-10s %-14s %-14s %-10s\n" "$N" "$NE" "$bpw" "$jlw" "$sp"
done
