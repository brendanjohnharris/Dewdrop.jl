using Dewdrop
using Test

# Robustness guards: an empty projection must be a no-op
# rather than crash; the clock-driven engine cannot deliver a same-step (delay 0) synapse; and
# the Poisson drive must reject a per-step mean λ = rate*dt past the sampler's underflow cliff
# (the units trap that 1000×-overdrove the canonical Brunel AI drive).
@testset "edge cases + correctness guards" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)

    @testset "empty projection is a no-op (no crash)" begin
        empty_conn = fixed_prob(Dewdrop.CPU(), 50, 50, 0.0; weight = 1.0, delay = steps(1), seed = UInt64(1))
        @test Dewdrop.nedges(empty_conn) == 0
        se = solve(DewdropNetwork(m, 50; input = 2.0, tspan = (0.0, 5.0),
                projection = Projection(CurrentSynapse(τ = 5.0), empty_conn)), FixedStep(0.1))
        s0 = solve(DewdropNetwork(m, 50; input = 2.0, tspan = (0.0, 5.0)), FixedStep(0.1))
        @test se.spike_count == s0.spike_count
    end

    @testset "delay must be ≥ 1 step" begin
        @test_throws ArgumentError fixed_prob(Dewdrop.CPU(), 10, 10, 0.5; weight = 1.0, delay = steps(0), seed = UInt64(1))
        @test fixed_prob(Dewdrop.CPU(), 10, 10, 0.5; weight = 1.0, delay = steps(1), seed = UInt64(1)) isa SparseCSR
    end

    @testset "Poisson drive λ underflow guard" begin
        bad = DewdropNetwork(m, 10; input = 0.0, tspan = (0.0, 1.0),
            drive = PoissonDrive(rate = 20000.0, weight = 0.1, seed = UInt64(1)))   # rate*dt = 2000
        @test_throws ArgumentError init(bad, FixedStep(0.1))
        ok = DewdropNetwork(m, 10; input = 0.0, tspan = (0.0, 1.0),
            drive = PoissonDrive(rate = 20.0, weight = 0.1, seed = UInt64(1)))       # rate*dt = 2
        @test init(ok, FixedStep(0.1)) isa Dewdrop.DewdropIntegrator
    end
end
