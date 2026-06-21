using Dewdrop
using Test

Dewdrop.set_advice!(false)   # silence the perf advisor during the suite; test/advisor.jl re-enables it

@testset verbose = true "Dewdrop.jl" begin
    include("architecture.jl")
    include("population.jl")
    include("rng.jl")
    include("connectivity.jl")
    include("fixedprob.jl")
    include("spatial.jl")
    include("delays.jl")
    include("scatter.jl")
    include("schedule.jl")
    include("neurons.jl")
    include("synapses.jl")
    include("engine.jl")
    include("fi_curve.jl")
    include("behavior.jl")
    include("connected.jl")
    include("multiproj.jl")
    include("builder.jl")
    include("initial_conditions.jl")
    include("edge_cases.jl")
    include("macro.jl")
    include("unitful.jl")
    include("timeseriesbase.jl")
    include("recording.jl")
    include("drive.jl")
    include("gpu_readiness.jl")
    include("fused.jl")             # M6 fused megakernel ≡ broadcast (JLArrays, no GPU needed)
    include("batch.jl")             # ensemble (tensor) batching ≡ scalar oracle (CPU + JLArrays)
    include("compaction.jl")        # compacted scatter ≡ edge/CPU reference (JLArrays, no GPU needed)
    include("cuda.jl")              # arch-seam check always; device sims guarded by CUDA.functional()
    include("advisor.jl")           # performance advisor heuristics (metadata-only; no GPU needed)
    # quality checks last: a slow/version-sensitive gate never blocks earlier feedback
    include("aqua.jl")
    include("jet.jl")
    # scenario / classical-figure reproductions (heaviest load: CairoMakie) last
    include("plots.jl")
    include("brunel.jl")
    include("vogels_abbott.jl")
end
