using Dewdrop
using Test

@testset verbose = true "Dewdrop.jl" begin
    include("architecture.jl")
    include("population.jl")
    include("rng.jl")
    include("connectivity.jl")
    include("fixedprob.jl")
    include("delays.jl")
    include("scatter.jl")
    include("schedule.jl")
    include("neurons.jl")
    include("synapses.jl")
    include("engine.jl")
    include("fi_curve.jl")
    include("behavior.jl")
    include("connected.jl")
    include("recording.jl")
    include("drive.jl")
    include("gpu_readiness.jl")
    # quality checks last: a slow/version-sensitive gate never blocks earlier feedback
    include("aqua.jl")
    include("jet.jl")
    # scenario / classical-figure reproductions (heaviest load: CairoMakie) last
    include("plots.jl")
    include("brunel.jl")
end
