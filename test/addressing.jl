using Dewdrop
using Test

# Named-subpopulation addressing: a subpop is a contiguous range into the one flat
# concatenated SoA, recorded in a registry on the problem and carried onto the solution. The
# reference API is symbol-indexed: `sol[:E]`, `firing_rate(sol, :E)`, `raster(sol; of = :E)`. This
# is pure addressing metadata over the flat engine (zero hot-loop cost); the simulation is
# byte-identical to the same network with no registry.

@testset "subpop registry + symbol reference API" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    N = 100
    prob = DewdropNetwork(m, N; subpops = (E = 1:50, I = 51:100), input = 30.0, tspan = (0.0, 200.0))
    sol = solve(prob, FixedStep(0.1))

    @testset "registry travels onto the solution" begin
        @test sol.subpops.E == 1:50
        @test sol.subpops.I == 51:100
        @test haskey(sol.subpops, :all)          # an implicit :all spanning the whole population
        @test sol.subpops.all == 1:100
    end

    @testset "sol[:name] is a sub-solution view" begin
        sub = sol[:E]
        @test sub.spike_count == sol.spike_count[1:50]
        @test sub.state.state.V == sol.state.state.V[1:50]
        @test sol[:all].spike_count == sol.spike_count
        @test_throws Exception sol[:nope]         # unknown subpop errors clearly
    end

    @testset "firing_rate by subpop" begin
        @test firing_rate(sol, :E) == firing_rate(sol)[1:50]
        @test firing_rate(sol, :I) == firing_rate(sol)[51:100]
        @test firing_rate(sol[:E]) == firing_rate(sol)[1:50]   # also works on the sub-solution
    end

    @testset "addressing is metadata-only (byte-identical sim)" begin
        bare = solve(DewdropNetwork(m, N; input = 30.0, tspan = (0.0, 200.0)), FixedStep(0.1))
        @test sol.spike_count == bare.spike_count
        @test sol.state.state.V == bare.state.state.V
    end
end

@testset "neuron positions carried onto the solution" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    pos = grid_positions(4, 4)                       # 16 positions
    prob = DewdropNetwork(m, 16; input = 30.0, tspan = (0.0, 50.0), positions = pos)
    sol = solve(prob, FixedStep(0.1))
    @test sol.positions == pos
    @test sol[:all].positions == pos                 # SubSolution carries the position slice
    # default: no positions → nothing
    @test solve(DewdropNetwork(m, 16; input = 30.0, tspan = (0.0, 50.0)), FixedStep(0.1)).positions === nothing

    # builder concatenates per-population positions (grid E + random I)
    nb = network(; tspan = (0.0, 50.0))
    population!(nb, :E, m, 8; input = 30.0, positions = grid_positions(4, 2))
    population!(nb, :I, m, 4; input = 30.0, positions = random_positions(4, (4.0, 4.0); seed = UInt64(1)))
    prob2 = build(nb)
    @test length(prob2.positions) == 12
    sol2 = solve(prob2, FixedStep(0.1))
    @test sol2.positions == prob2.positions
    @test sol2[:I].positions == prob2.positions[9:12]   # per-subpop coordinates (for spatial measures)
    @test length(sol2[:E].positions) == 8
end

@testset "recording restricted to a subpop (of = :E)" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    N = 100
    prob = DewdropNetwork(m, N; subpops = (E = 1:50, I = 51:100), input = 30.0, tspan = (0.0, 60.0))
    sol = solve(
        prob, FixedStep(0.1); record = (
            vfull = Trace(:V),
            vI = Trace(:V; of = :I),
            sE = Spikes(of = :E),
        )
    )
    @test size(sol.record.vI.data, 1) == 50             # only the I subpop's rows
    @test sol.record.vI.data == sol.record.vfull.data[51:100, :]   # same values, restricted
    @test size(sol.record.sE.data, 1) == 50             # E subpop spike raster
    @test_throws Exception solve(prob, FixedStep(0.1); record = (bad = Trace(:V; of = :nope),))
end

@testset "raster by subpop (rebased ids)" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    N = 100
    prob = DewdropNetwork(m, N; subpops = (E = 1:50, I = 51:100), input = 30.0, tspan = (0.0, 100.0))
    sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),))
    t_all, id_all = raster(sol)
    t_I, id_I = raster(sol; of = :I)
    @test !isempty(id_I)
    @test all(1 .<= id_I .<= 50)                  # ids rebased into 1:|I|
    # the I subpop's spike total matches the absolute-id events in 51:100
    @test length(id_I) == count(in(51:100), id_all)
end
