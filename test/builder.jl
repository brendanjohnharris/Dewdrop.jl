using Dewdrop
using Test

# M2 --- a fluent builder for E/I networks: one population of NE+NI neurons with named
# :E (1:NE) and :I (NE+1:end) subpopulations, projections added by `connect!`, an external
# drive by `drive!`, assembled with `build`. Removes the manual fixed_prob / sources /
# Projection boilerplate that Brunel and Vogels–Abbott otherwise need.
@testset "network builder API" begin
    m = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 5.0)

    nb = network(m, 80, 20; arch = Dewdrop.CPU(), tspan = (0.0, 200.0))
    @test nb isa Dewdrop.NetworkBuilder
    project!(nb, :E, ConductanceSynapse(τ = 5.0, Erev = 0.0); p = 0.1, weight = 0.6, delay = steps(1), seed = UInt64(1))
    project!(nb, :I, ConductanceSynapse(τ = 10.0, Erev = -80.0); p = 0.1, weight = 6.7, delay = steps(1), seed = UInt64(2))
    drive!(nb, PoissonDrive(; rate = 6.0, weight = 0.1, seed = UInt64(7)))
    prob = build(nb)

    @test prob isa DewdropNetwork
    @test prob.n == 100                         # NE + NI
    @test length(prob.projections) == 2
    @test prob.drive !== nothing

    # the :E projection draws only from excitatory neurons (1..80): I neurons emit no E-edges
    e_conn = prob.projections[1].conn
    n_i_out = 0
    for pre in 81:100
        Dewdrop.for_each_post(e_conn, pre) do post, w, d
            n_i_out += 1
        end
    end
    @test n_i_out == 0

    # the assembled problem runs (and the hot loop is concrete: the builder boundary is the
    # only dynamic point), producing some activity
    sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),))
    @test sol.nsteps == 2000
    @test sum(sol.spike_count) > 0

    # :all sources span the whole population
    nb2 = network(m, 80, 20; arch = Dewdrop.CPU(), tspan = (0.0, 50.0))
    project!(nb2, :all, CurrentSynapse(τ = 5.0); p = 0.05, weight = 1.0, delay = steps(1), seed = UInt64(3))
    @test length(build(nb2).projections) == 1

    # the 2-arg builder now registers the :E / :I subpops on the problem (for the reference API)
    @test build(nb2).subpops.E == 1:80
    @test build(nb2).subpops.I == 81:100
end

# Fluent multi-population builder (Phase A): `network(; tspan)` accumulates named populations via
# `population!`, projections via `project!(src => dst, …)`, an external drive by `drive!`, assembled
# by `build` into a flat DewdropNetwork with a subpop registry. Same-type groups with differing
# parameters are merged into one `Heterogeneous` model (block per-neuron arrays).
@testset "fluent multi-population builder" begin
    @testset "named populations → registry + runs" begin
        nb = network(; arch = Dewdrop.CPU(), tspan = (0.0, 200.0))
        mE = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
        mI = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
        population!(nb, :E, mE, 80; input = 30.0)
        population!(nb, :I, mI, 20; input = 30.0)
        project!(nb, :E => :I, CurrentSynapse(τ = 5.0); p = 0.1, weight = 1.0, delay = steps(1), seed = UInt64(1))
        drive!(nb, PoissonDrive(; rate = 1.0, weight = 0.1, seed = UInt64(9)))
        prob = build(nb)
        @test prob isa DewdropNetwork
        @test prob.n == 100
        @test prob.subpops.E == 1:80
        @test prob.subpops.I == 81:100
        sol = solve(prob, FixedStep(0.1))
        @test sum(sol.spike_count) > 0
        @test firing_rate(sol, :E) == firing_rate(sol)[1:80]
    end

    @testset "src => dst projection wires only E → I" begin
        nb = network(; arch = Dewdrop.CPU(), tspan = (0.0, 50.0))
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
        population!(nb, :E, m, 50)
        population!(nb, :I, m, 50)
        project!(nb, :E => :I, CurrentSynapse(τ = 5.0); p = 0.3, weight = 1.0, delay = steps(1), seed = UInt64(2))
        prob = build(nb)
        conn = prob.projections[1].conn
        pairs = Tuple{Int, Int}[]
        for pre in 1:100
            Dewdrop.for_each_post(conn, pre) do post, w, d
                push!(pairs, (pre, post))
            end
        end
        @test !isempty(pairs)
        @test all(pr -> pr[1] in 1:50 && pr[2] in 51:100, pairs)   # only E(1:50) → I(51:100)
    end

    @testset "same-type groups with differing params → Heterogeneous" begin
        # E adapts (b = 1.0), I does not (b = 0.0): one AdaptLIF type, block per-neuron `b`
        base = (; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0, a = 0.0, τw = 150.0)
        nb = network(; arch = Dewdrop.CPU(), tspan = (0.0, 1000.0))
        population!(nb, :E, AdaptLIF(; base..., b = 1.0), 50; input = 20.0)
        population!(nb, :I, AdaptLIF(; base..., b = 0.0), 50; input = 20.0)
        prob = build(nb)
        @test prob.model isa Heterogeneous
        sol = solve(prob, FixedStep(0.1))
        # the non-adapting I half fires more than the adapting E half (spike-frequency adaptation)
        @test sum(sol.spike_count[1:50]) < sum(sol.spike_count[51:100])
    end

    @testset "different model TYPES → MultiModel (AdEx-E + LIF-I)" begin
        nb = network(; arch = Dewdrop.CPU(), tspan = (0.0, 300.0))
        population!(nb, :E, AdEx(; C = 200.0, gL = 10.0, EL = -70.0, VT = -50.0, ΔT = 2.0, Vr = -58.0,
            Vpeak = 0.0, a = 2.0, b = 60.0, τw = 120.0, tref = 2.0), 40; input = 700.0)
        population!(nb, :I, LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 2.0), 40;
            input = 400.0)
        prob = build(nb)
        @test prob.model isa MultiModel
        @test prob.subpops.E == 1:40
        @test prob.subpops.I == 41:80
        sol = solve(prob, FixedStep(0.05))
        @test sum(firing_rate(sol, :E)) > 0          # AdEx excitatory group fires
        @test sum(firing_rate(sol, :I)) > 0          # LIF inhibitory group fires
    end
end
