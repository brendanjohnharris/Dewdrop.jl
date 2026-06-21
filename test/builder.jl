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
    project!(nb, :E, ConductanceSynapse(τ = 5.0, Erev = 0.0); p = 0.1, weight = 0.6, delay = 1, seed = UInt64(1))
    project!(nb, :I, ConductanceSynapse(τ = 10.0, Erev = -80.0); p = 0.1, weight = 6.7, delay = 1, seed = UInt64(2))
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
    project!(nb2, :all, CurrentSynapse(τ = 5.0); p = 0.05, weight = 1.0, delay = 1, seed = UInt64(3))
    @test length(build(nb2).projections) == 1
end
