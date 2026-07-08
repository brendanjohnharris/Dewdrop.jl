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
    include("adaptation.jl")        # AdaptLIF + AdEx (multi-state (V,w) seam)
    include("heterogeneous.jl")     # per-neuron heterogeneous parameters
    include("multimodel.jl")        # multi-type populations (MultiModel: AdEx-E + LIF-I, union SoA)
    include("fns.jl")               # FNSNeuron (conductance-adaptation LIF)
    include("synapses.jl")
    include("dualexp.jl")           # dual-exponential COBA synapse (DualExpSynapse)
    include("frozendualexp.jl")     # frozen-current variant (FrozenDualExpSynapse): no membrane shunt
    include("engine.jl")
    include("fi_curve.jl")
    include("behavior.jl")
    include("connected.jl")
    include("multiproj.jl")
    include("plasticity.jl")        # event-driven STDP (mutable weights + traces, scatter-path)
    include("builder.jl")
    include("networkspec.jl")       # deferred network spec: freeze(builder)/defer(constructor) → materialise at solve
    include("blockbatch.jl")        # batching: block-diagonal general path + batch(...) input forms
    include("addressing.jl")        # named-subpopulation registry + symbol reference API
    include("stats.jl")             # statistical observables + internal FFT
    include("initial_conditions.jl")
    include("edge_cases.jl")
    include("macro.jl")
    include("unitful.jl")
    include("timeseriesbase.jl")
    include("recording.jl")
    include("progress.jl")          # host-side progress reporting (ProgressLogging convention) for solve!
    include("show.jl")              # hierarchical Base.show rendering (models, synapses, network, builder, solutions)
    include("drive.jl")
    include("stimuli.jl")           # unified AbstractStimulus seam: Functional/TimedArray/InhomogeneousPoisson/SpikeSourceArray
    include("poissonsource.jl")     # generic streaming Poisson drive: PoissonSource{Synapse} + drive! verb
    include("noise.jl")             # SDE noise: draw_normal, exact-OU variance, 3-path ≡
    include("gpu_readiness.jl")
    include("backends.jl")          # execution backends: Auto/Serial/Fused/Turbo selection + dispatch
    include("fused.jl")             # fused megakernel ≡ broadcast (JLArrays, no GPU needed)
    include("turbo.jl")             # Turbo backend (LoopVectorization ext) ≡ Serial spike-identical; loads the ext
    include("differentiable.jl")    # Differentiable backend (surrogate-gradient): ForwardDiff gradient ≡ finite-diff, trains
    include("batch.jl")             # ensemble (tensor) batching ≡ scalar reference (CPU + JLArrays)
    include("batched_params.jl")    # batched per-member parameters (neuron + synapse; generic physical-param sweeps)
    include("compaction.jl")        # compacted scatter ≡ edge/CPU reference (JLArrays, no GPU needed)
    include("cuda.jl")              # arch-seam check always; device sims guarded by CUDA.functional()
    include("advisor.jl")           # performance advisor heuristics (metadata-only; no GPU needed)
    # quality checks last: a slow/version-sensitive gate never blocks earlier feedback
    include("aqua.jl")
    include("jet.jl")
    # scenario / classical-figure reproductions (heaviest load: CairoMakie) last
    include("makie.jl")             # TimeseriesMakie ext: spikeraster/psth/ratemap/traceplot/phaseplane/positionplot/connectivity
    include("plots.jl")
    include("brunel.jl")
    include("vogels_abbott.jl")
end
