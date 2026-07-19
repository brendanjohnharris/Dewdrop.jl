using Dewdrop
using Test
using CairoMakie
using TimeseriesMakie

# The TimeseriesMakie extension: base neural recipes (spikeraster/psth/ratemap) live in TimeseriesMakie
# and are specialised here for Dewdrop solutions; traceplot/phaseplane reuse traces/trajectory;
# positionplot/connectivity are Dewdrop-only. The oracle is a full RENDER: each recipe is drawn into a
# figure and saved (a recipe that merely builds but fails to draw is caught by `save`). Artifacts go to
# the gitignored test/plots/ for visual inspection.

const MPLOTDIR = joinpath(@__DIR__, "plots")
isdir(MPLOTDIR) || mkpath(MPLOTDIR)
_saved(fig, name) = (p = joinpath(MPLOTDIR, name); save(p, fig); isfile(p))

@testset "Makie recipes" begin
    dt = 0.1

    @testset "base recipes on plain arrays" begin
        times = [1.0, 2.0, 2.5, 5.0, 6.0]
        ids = [1, 2, 1, 3, 2]
        fig = Figure()
        spikeraster!(Axis(fig[1, 1]), times, ids)                     # flat (times, ids)
        spikeraster!(Axis(fig[1, 2]), [[1.0, 2.0], [3.0], Float64[]]) # vector-of-vectors
        @test _saved(fig, "unit_spikeraster.png")

        S = [rand() < 0.2 for _ in 1:8, _ in 1:50]                    # Neuron × Time Bool mask
        fig = Figure()
        spikeraster!(Axis(fig[1, 1]), S)                              # Bool matrix
        psth!(Axis(fig[1, 2]), times; binwidth = 1.0)
        ratemap!(Axis(fig[2, 1]), S; binwidth = 5)
        ratemap!(Axis(fig[2, 2]), collect(1:50) .* dt, S)             # (times, raster)
        @test _saved(fig, "unit_base_recipes.png")
    end

    @testset "spike/rate on a DewdropSolution" begin
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 2.0)
        N, NE = 40, 30
        w(pre) = pre ≤ NE ? 0.6 : -1.2
        conn = fixed_prob(CPU(), N, N, 0.08; weight = w, delay = steps(1), seed = 0x01, allow_self = false)
        prob = DewdropNetwork(
            m, N; input = 25.0, tspan = (0.0, 300.0),   # suprathreshold (Vθ = 20)
            projection = Projection(DeltaSynapse(), conn), subpops = (E = 1:NE, I = (NE + 1):N)
        )
        sol = solve(prob, FixedStep(dt); record = (spikes = Spikes(), v = Trace(:V)))

        fig = Figure()
        spikeraster!(Axis(fig[1, 1]), sol)                            # L3 adapter
        spikeraster!(Axis(fig[1, 2]), sol; sortby = :rate)            # sorted
        psth!(Axis(fig[2, 1]), sol; binwidth = 5.0)                   # via pooled spike times
        ratemap!(Axis(fig[2, 2]), sol; binwidth = 10)                 # via _spike_raster
        @test _saved(fig, "sol_spike_rate.png")

        @test _saved(plot(sol).figure, "sol_plottype_raster.png")     # plottype → raster
        @test _saved(traceplot(sol, :v).figure, "sol_traces.png")     # reuse `traces`

        fig = Figure()
        spikeraster!(Axis(fig[1, 1]), sol[:E])                        # SubSolution (E only)
        @test _saved(fig, "sol_subpop_raster.png")
    end

    @testset "phase plane (AdEx V-w)" begin
        m = AdEx(;
            C = 200.0, gL = 10.0, EL = -70.0, VT = -50.0, ΔT = 2.0, Vr = -58.0,
            Vpeak = 0.0, a = 2.0, b = 60.0, τw = 120.0, tref = 2.0
        )
        sol = solve(
            DewdropNetwork(m, 3; input = 500.0, tspan = (0.0, 400.0)), FixedStep(dt);
            record = (V = Trace(:V), w = Trace(:w))
        )
        @test _saved(phaseplane(sol; vars = (:V, :w), neuron = 1).figure, "adex_phaseplane.png")
    end

    @testset "positions + connectivity" begin
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 2.0)
        N = 60
        pos = random_positions(N, (1.0, 1.0); seed = UInt(7))            # per-dim side lengths
        conn = fixed_prob(CPU(), N, N, 0.05; weight = 0.5, delay = steps(1), seed = 0x02, allow_self = false)
        prob = DewdropNetwork(
            m, N; input = 1.2, tspan = (0.0, 100.0),
            projection = Projection(DeltaSynapse(), conn), positions = pos
        )
        sol = solve(prob, FixedStep(dt))
        @test _saved(positionplot(sol).figure, "positions_rate.png")          # color by rate
        @test _saved(positionplot(sol; color = :type).figure, "positions_type.png")

        fig = Figure()
        connectivity!(Axis(fig[1, 1]), conn)                          # SparseCSR
        connectivity!(Axis(fig[1, 2]), prob)                          # DewdropNetwork
        @test _saved(fig, "connectivity.png")
    end
end
