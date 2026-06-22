using Dewdrop
using Test

# WRCircuit end-to-end --- the spatial E/I "working-regime" circuit assembled in pure Dewdrop from
# the pieces built across the WRCircuit phases: FNS conductance-adaptation neurons (E adapts, I does
# not), dual-exponential COBA synapses (excitatory Erev=0 / inhibitory Erev<0), distance-dependent
# fixed-count connectivity with exponential kernels + periodic boundaries (broad inhibition), E on a
# grid + I at random positions, all wired through the fluent named-subpopulation builder. This is a
# small-scale run check (it executes, both populations are active, adaptation engages, no NaN) --- a
# statistical match to the BrainPy backend on an identical connectome is the separate validation step.

@testset "WRCircuit-style spatial E/I network (end-to-end)" begin
    Lx = Ly = 16.0
    NE, NI = 256, 64                                  # 16×16 E grid + 64 random I (4:1)
    posE = grid_positions(16, 16)
    posI = random_positions(NI, (Lx, Ly); seed = UInt64(11))

    # one FNS model type, E adapts (ΔgK>0) and I does not (ΔgK=0) → the builder merges them into a
    # single Heterogeneous FNS (block per-neuron ΔgK).
    fnsE = FNSNeuron(; ΔgK = 0.01)
    fnsI = FNSNeuron(; ΔgK = 0.0)

    exc() = DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    inh() = DualExpSynapse(; τr = 1.0, τd = 8.0, Erev = -85.0)

    nb = network(; tspan = (0.0, 300.0))
    population!(nb, :E, fnsE, NE; input = 0.5, positions = posE)     # input > FNS rheobase (≈0.334)
    population!(nb, :I, fnsI, NI; input = 0.5, positions = posI)
    # four distance-dependent fixed-count projections; broad inhibition (larger σ), periodic box
    project!(nb, :E => :E, exc(); kernel = exponential_kernel(2.0), count = 6 * NE, weight = 0.002,
        delay = 2, seed = UInt64(1), period = (Lx, Ly))
    project!(nb, :E => :I, exc(); kernel = exponential_kernel(2.5), count = 6 * NI, weight = 0.002,
        delay = 2, seed = UInt64(2), period = (Lx, Ly))
    project!(nb, :I => :E, inh(); kernel = exponential_kernel(4.5), count = 6 * NE, weight = 0.004,
        delay = 2, seed = UInt64(3), period = (Lx, Ly))
    project!(nb, :I => :I, inh(); kernel = exponential_kernel(4.5), count = 6 * NI, weight = 0.004,
        delay = 2, seed = UInt64(4), period = (Lx, Ly))
    prob = build(nb)

    @test prob.model isa Heterogeneous                # same FNS type, per-neuron ΔgK
    @test prob.n == NE + NI
    @test length(prob.projections) == 4
    @test prob.subpops.E == 1:NE && prob.subpops.I == (NE + 1):(NE + NI)
    @test Dewdrop.nedges(prob.projections[1].conn) == 6 * NE   # exact fixed count

    # randomised initial V breaks synchrony; run and inspect via the named-subpop reference API
    sol = solve(prob, FixedStep(0.1); v0 = (-70.0, -50.0), record = (spikes = Spikes(),))
    @test all(isfinite, sol.state.state.V)            # no NaN from the conductance dynamics
    @test sum(sol.spike_count[1:NE]) > 0              # excitatory population active
    @test sum(firing_rate(sol, :I)) > 0              # inhibitory population active (via the addressor)
    @test maximum(sol.state.state.w[1:NE]) > 0        # adaptation conductance gK accumulated in E
    t_E, _ = raster(sol; of = :E)                    # per-subpop raster works on the spatial net
    @test !isempty(t_E)

    # spatial metadata flows through to the solution (for radial-AC / spatial measures)
    @test length(sol.positions) == NE + NI
    @test sol[:E].positions == posE                  # grid coordinates of the E subpopulation
    @test sol[:I].positions == sol.positions[(NE + 1):end]
end
