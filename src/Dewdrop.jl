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

include("Architecture.jl")
include("Population.jl")
include("RNG.jl")
include("Connectivity.jl")
include("Delays.jl")
include("Scatter.jl")
include("Schedule.jl")
include("Neurons.jl")
include("Synapses.jl")
include("Engine.jl")

end # module Dewdrop
