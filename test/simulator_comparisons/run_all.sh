#!/usr/bin/env bash
# Run every simulator's benchmark sequentially, then compare. The one-command entry point:
#
#     ./run_all.sh                 # run all discovered simulators, then compare_simulators.jl
#     ./run_all.sh dewdrop brian   # run only the named simulators, then compare
#
# MODULAR: each simulator directory holds a run.jl (Julia) or run.py (Python); this discovers them,
# runs each in its own language/environment, and finally runs compare_simulators.jl (verify + plot).
# A failure in one simulator (missing GPU, missing venv, …) does NOT stop the others or the compare.
# Drop in a new nest/ or neuron/ with a run.* and it is picked up automatically. (stats_validation/
# has no run.*, so it is skipped.) No simulator-specific logic lives here --- everything is in spec.toml.
set -u
cd "$(dirname "$(readlink -f "$0")")"

# --- which simulators? explicit args win; otherwise discover, dewdrop first (reference, always built) ---
if [[ $# -gt 0 ]]; then
    sims=("$@")
else
    sims=()
    [[ -f dewdrop/run.jl ]] && sims+=(dewdrop)
    for d in */; do
        d=${d%/}
        [[ "$d" == dewdrop ]] && continue
        [[ -f "$d/run.jl" || -f "$d/run.py" ]] && sims+=("$d")
    done
fi
[[ ${#sims[@]} -gt 0 ]] || { echo "no simulators (run.jl/run.py) found in $PWD"; exit 1; }

echo "Simulators to run: ${sims[*]}"
ok=(); failed=()
for sim in "${sims[@]}"; do
    echo; echo "==================== $sim ===================="
    rc=0
    if [[ -f "$sim/run.jl" ]]; then
        bash "$sim/run.jl" || rc=$?                          # dual-shebang: execs the pinned julia
    elif [[ -f "$sim/run.py" ]]; then
        py="$sim/.venv/bin/python"
        [[ -x "$py" ]] || py="$(command -v python3 || command -v python || true)"
        if [[ -z "$py" ]]; then
            echo "  no python (and no $sim/.venv) --- skipping $sim"; failed+=("$sim"); continue
        fi
        "$py" "$sim/run.py" || rc=$?
    else
        echo "  no run.jl/run.py in $sim/ --- skipping"; continue
    fi
    if [[ $rc -eq 0 ]]; then ok+=("$sim"); else failed+=("$sim"); echo "  !! $sim exited $rc (continuing)"; fi
done

echo; echo "==================== compare ===================="
echo "ran: ${ok[*]:-none}   failed: ${failed[*]:-none}"
bash compare_simulators.jl                                   # verify all ran the same problem + plot scaling
