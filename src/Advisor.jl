# * Performance advisor
# Heuristic `@info` guidance toward the specialised GPU scatter/precision paths when a run's
# regime suggests one would help. It inspects the problem (architecture, float type, connectivity
# index width, mean degree) and the solution (firing fraction per step), and emits at most one
# message per distinct suggestion per session (so a parameter sweep is not spammed). Silenceable
# globally (`Dewdrop.set_advice!(false)`) or per call (`solve(...; advise = false)`). The advice is
# hedged --- these are rules of thumb, not guarantees.

const _ADVISE = Ref(true)
const _ADVISED = Set{Symbol}()
const _RUNTIME_DONE = Ref(false)   # runtime advice needs a device reduction; do it once per session

"""
    set_advice!(on::Bool)

Enable or disable the performance advisor globally (default `true`). Disabling also skips the
device reduction the runtime advice needs, so it is free on a hot path. Per-call override:
`solve(prob, alg; advise = false)`.
"""
set_advice!(on::Bool) = (_ADVISE[] = on; on)
export set_advice!

# Re-arm the once-per-session dedup (used by the advisor's own tests).
reset_advice!() = (empty!(_ADVISED); _RUNTIME_DONE[] = false; nothing)

@inline function _emit(key::Symbol, msg::AbstractString)
    (_ADVISE[] && key ∉ _ADVISED) || return false
    push!(_ADVISED, key)
    @info "Dewdrop performance advisor: " * msg * "\n(silence with `Dewdrop.set_advice!(false)` or `solve(...; advise = false)`)"
    return true
end

# --- problem-side facts (no device reads) ---
_is_gpu(prob::DewdropNetwork) = prob.arch isa GPU
_conns(prob::DewdropNetwork) = (p.conn for p in prob.projections)
_total_edges(prob::DewdropNetwork) = sum(nedges, _conns(prob); init = 0)
function _mean_degree(prob::DewdropNetwork)
    ne = _total_edges(prob)
    return ne == 0 ? 0.0 : ne / prob.n
end

# --- solution-side firing fraction per step (one device reduction; gated behind `advise`) ---
# Duck-typed over DewdropSolution / BatchedSolution (both carry `spike_count` + `nsteps`); for the
# batched case `length` counts all N×B cells, giving the per-cell-per-step fraction.
_firing_fraction(sol) =
    sol.nsteps == 0 ? 0.0 : sum(sol.spike_count) / (length(sol.spike_count) * sol.nsteps)

# Static suggestions (precision + index width): immediately actionable, no runtime info needed.
function _advise_static(prob::DewdropNetwork)
    _is_gpu(prob) || return nothing
    if float_type(prob.model) === Float64
        _emit(:float32,
            "running Float64 on the GPU. Float32 state (e.g. `LIF(; τ = 20.0f0, …)` with " *
            "`FixedStep(0.1f0)` and Float32 weights) runs ~1.7-2.4× faster combined with Int32 " *
            "indices, and matches Float64 dynamics to within ~5%.")
    end
    for conn in _conns(prob)
        if eltype(conn.post) === Int && 0 < nedges(conn) < typemax(Int32)
            _emit(:int32,
                "connectivity uses 64-bit indices (nedges = $(nedges(conn))). Pass " *
                "`index_type = Int32` to `fixed_prob`/`SparseCSR` to halve the scatter's index " *
                "bandwidth (~1.5-2×, bit-identical).")
            break
        end
    end
    return nothing
end

# Runtime suggestions (regime-dependent): need the measured firing fraction.
function _advise_runtime(prob::DewdropNetwork, frac::Real)
    _is_gpu(prob) || return nothing
    ne = _total_edges(prob)
    md = _mean_degree(prob)
    pct = round(frac * 100; digits = 2)
    # only suggest compaction if the default `scatter = :auto` did NOT already select it for this network
    # (past the L2-spill crossover it is automatic, so the hint would be redundant); below that crossover
    # the connectome is L2-resident and the edge scatter's no-sync launch is the right pick.
    if ne > 1_000_000 && frac < 0.02 && _resolve_scatter(:auto, prob.arch, prob.projections) === :edge
        _emit(:compaction,
            "sparse firing ($(pct)% of neurons per step) over a large network " *
            "(nedges = $ne): the scatter wastes most of its threads on silent synapses. The " *
            "compacted scatter (`solve(...; scatter = :compacted)`) processes only active " *
            "synapses --- measured up to ~30× faster in this regime.")
    elseif md > 500 && frac > 0.05
        _emit(:gather,
            "dense connectivity (mean degree ≈ $(round(Int, md))) at high firing ($(pct)%/step): " *
            "the atomic edge-parallel scatter is contention-bound here. A gather/SpMV backend " *
            "(atomic-free, target-parallel) is the structural fit --- not yet implemented.")
    elseif prob.n < 5000 && frac < 0.01
        _emit(:graphs,
            "a small, quiet network: the step is launch-bound, not compute-bound, so scatter " *
            "tuning will not help. CUDA-graph capture of the per-step launch sequence is the " *
            "lever here (not yet implemented); for parameter sweeps, batching (`solve(...; " *
            "batch = B)`) amortises launches across instances.")
    end
    return nothing
end

# CPU suggestion (static, free): at large N the dense per-neuron phases dominate the step. The default
# `Auto` backend now routes EVERY canonical CPU network through the `Fused` step (the single-pass tight
# loop --- bit-identical, ~2× the per-phase `Serial` baseline; it gates its own threading on
# work-per-thread, so small nets run single-threaded and large ones thread), so nothing is needed there.
# The remaining lever is SIMD: `backend = Turbo()` (LoopVectorization) vectorises the membrane `exp` for
# ~compiled-C++ throughput, at the cost of bit-identicality (spike-identical). Point users to it when
# they could benefit (large N, multicore, canonical schedule).
function _advise_cpu(prob::DewdropNetwork)
    if Threads.nthreads() > 1 && prob.n ≥ 10_000 && prob.schedule == default_schedule()
        _emit(:turbo_cpu,
            "a large CPU network (N = $(prob.n)): the default `Auto` backend already runs the threaded " *
            "`Fused` step here (a single fused pass, ~2× the per-phase `Serial` baseline, bit-identical). " *
            "For close to compiled-C++ throughput, `using LoopVectorization` unlocks `backend = Turbo()` " *
            "--- a SIMD-vectorised step for models with a Turbo specialization (AdEx, LIF, …; see the " *
            "backend docs). Turbo is spike-identical but not bit-identical (SIMD `exp`).")
    end
    return nothing
end

# Entry point: called by `solve` with the problem and its solution.
function _run_advisor(prob::DewdropNetwork, sol)
    _ADVISE[] || return nothing
    if _is_gpu(prob)
        _advise_static(prob)                             # GPU precision/index advice (free)
        if !_RUNTIME_DONE[]                              # GPU firing-rate advice: one device reduction per session
            _RUNTIME_DONE[] = true
            _advise_runtime(prob, _firing_fraction(sol))
        end
    else
        _advise_cpu(prob)                                # CPU step-strategy advice (free)
    end
    return nothing
end

# `solve` over a Dewdrop problem replicates the CommonSolve default (`solve! ∘ init`) and then runs
# the advisor on the (problem, solution). `advise = false` disables it for this call (and skips the
# runtime device reduction); `kwargs` (record/v0/batch/…) flow to `init`.
function CommonSolve.solve(prob::DewdropNetwork, alg::FixedStep; advise::Bool = true, kwargs...)
    sol = solve!(init(prob, alg; kwargs...))
    advise && _run_advisor(prob, sol)
    return sol
end
