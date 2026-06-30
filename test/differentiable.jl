using Dewdrop
using Dewdrop: Differentiable, duration
using Test
using ForwardDiff

# The `Differentiable` backend swaps the discontinuous spike for a smooth fast-sigmoid surrogate + soft
# reset + real-valued count, so a scalar loss is back-propagatable to the model parameters through the
# whole time loop (surrogate-gradient training / gradient-based fitting). It is a SEPARATE step path; the
# default backends stay bit-identical (covered by backends.jl `Serial ≡ Fused`).

@testset "differentiable backend (surrogate-gradient training)" begin
    # an unconnected LIF population driven above threshold; tref = 0 (conventional for surrogate training).
    build(R) = DewdropNetwork(LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = R, tref = 0.0),
        8; input = 30.0, tspan = (0.0, 300.0))
    rate(R; β = 25.0) = let s = solve(build(R), FixedStep(0.1); backend = Differentiable(β = β))
        sum(s.spike_count) / (8 * duration(s))
    end

    @testset "default eltype is unchanged (the bit-identical guarantee)" begin
        s = solve(build(1.0), FixedStep(0.1))                  # default (Auto) backend
        @test eltype(s.spike_count) == Int                     # Bool/Int preserved for every non-Differentiable run
    end

    @testset "runs with a real-valued surrogate count" begin
        s = solve(build(1.0), FixedStep(0.1); backend = Differentiable())
        @test eltype(s.spike_count) <: AbstractFloat           # surrogate count is real-valued (not Int)
        @test all(isfinite, s.spike_count)
        @test rate(1.0) > 0                                    # the population actually fires (surrogate)
    end

    @testset "surrogate rate is monotone in R" begin
        @test rate(0.9) < rate(1.0) < rate(1.2)                # higher R → higher V∞ → higher rate
    end

    @testset "ForwardDiff gradient matches finite differences (through the real solve)" begin
        R0 = 1.0
        g_ad = ForwardDiff.derivative(rate, R0)
        h = 1e-4
        g_fd = (rate(R0 + h) - rate(R0 - h)) / (2h)
        @test isfinite(g_ad) && g_ad > 0
        @test g_ad ≈ g_fd rtol = 1e-5
    end

    @testset "gradient descent reduces a fit-to-rate loss" begin
        target = 0.6 * rate(1.0)                               # a reachable target
        loss(R) = (rate(R) - target)^2
        R = 1.0
        l0 = loss(R)
        for _ in 1:20
            R -= 6.0 * ForwardDiff.derivative(loss, R)
        end
        @test loss(R) < l0 / 5                                 # ≥5× loss reduction
    end

    @testset "error paths (CPU-only, unconnected, canonical for now)" begin
        conn = fixed_prob(Dewdrop.CPU(), 8, 8, 0.2; weight = 0.5, delay = steps(1), seed = UInt64(1))
        connected = DewdropNetwork(LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 0.0),
            8; input = 30.0, tspan = (0.0, 10.0), projection = Projection(DeltaSynapse(), conn))
        @test_throws Exception init(connected, FixedStep(0.1); backend = Differentiable())  # surrogate scatter is the next step
    end
end
