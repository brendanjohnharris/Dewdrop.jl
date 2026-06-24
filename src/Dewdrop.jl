module Dewdrop

# Dewdrop.jl --- a generic, intuitive, GPU-aware spiking neural network simulator.
#
# Built CPU-first on a GPU-aware architecture (see the design plan): a custom
# fixed-step, clock-driven, struct-of-arrays engine. Optional dependencies
# (CUDA, DimensionalData, Makie, ...) live behind package extensions in `ext/`.
#
# Components are added milestone by milestone, each test-driven:
#   M0 --- scaffold + GPU-readiness contracts (architecture seam, SoA state,
#          counter-based RNG, connectivity/operator interfaces, schedule)
#   M1 --- LIF + current synapse, exponential-Euler, partitioned CSR scatter,
#          per-synapse-delay ring buffer.

using Adapt: Adapt, adapt
using StructArrays: StructArray
using CommonSolve: CommonSolve

include("Units.jl")
include("Architecture.jl")
include("Population.jl")
include("RNG.jl")
include("Connectivity.jl")
include("Spatial.jl")
include("Delays.jl")
include("Scatter.jl")
include("Schedule.jl")
include("Neurons.jl")
include("Adaptation.jl")
include("Heterogeneous.jl")
include("MultiModel.jl")
include("Synapses.jl")
include("Monitors.jl")
include("Noise.jl")
include("Backends.jl")          # execution backends (Auto/Serial/Fused/Turbo) — types before the integrator
include("Progress.jl")          # host-side progress reporting (ProgressLogging convention) for solve!
include("Engine.jl")
include("Fused.jl")
include("Compaction.jl")
include("Differentiable.jl")    # surrogate-gradient backend (autodiff-able CPU step); after Fused/Compaction
include("Batch.jl")
include("Plasticity.jl")
include("Advisor.jl")
include("Macro.jl")
include("Builder.jl")
include("PoissonSource.jl")     # generic streaming Poisson drive: PoissonSource{Synapse} over any synapse
include("NetworkSpec.jl")       # deferred network spec: specify a network without building the connectome
include("BlockBatch.jl")        # batching: run B members together (block-diagonal general path; batch(...) forms)
include("FFT.jl")               # self-contained DFT/FFT for the spectral observables
include("Stats.jl")             # statistical observables (WRCircuit stats.py ports; host-side analysis)
include("WRCircuit.jl")         # WRCircuit reproduction: frozen-COBA + prescribed synapses + native builder
include("SpatialFNS.jl")        # native spatial FNS "working-regime" constructor (BrainPy `Spatial`, no Python)
include("Show.jl")              # beautiful hierarchical REPL rendering (Base.show); last — renders every type above

end # module Dewdrop
