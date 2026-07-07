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
    build(R) = DewdropNetwork(
        LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = R, tref = 0.0),
        8; input = 30.0, tspan = (0.0, 300.0)
    )
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
        h = 1.0e-4
        g_fd = (rate(R0 + h) - rate(R0 - h)) / (2h)
        @test isfinite(g_ad) && g_ad > 0
        @test g_ad ≈ g_fd rtol = 1.0e-5
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

    @testset "connected surrogate-gradient training (trainable weights)" begin
        # a recurrent CUBA net at surrogate-training params. The whole net is built at eltype T so a
        # ForwardDiff `Dual` flows through the state, the synapse accumulators, the delay ring, and the
        # surrogate-WEIGHTED scatter — gradients reach the synaptic weights. Impossible before the scatter.
        function netspikes(; R, w, N = 12, β = 25.0)
            T = promote_type(typeof(R), typeof(w))
            m = LIF(; τ = T(20.0), EL = T(0.0), Vθ = T(20.0), Vr = T(0.0), R = T(R), tref = T(0.0))
            conn = fixed_prob(Dewdrop.CPU(), N, N, 0.3; weight = T(w), delay = steps(1), seed = UInt64(1), allow_self = false)
            prob = DewdropNetwork(m, N; input = T(30.0), tspan = (0.0, 150.0),
                projections = (Projection(CurrentSynapse(; τ = T(5.0)), conn),))
            return sum(solve(prob, FixedStep(0.1); backend = Differentiable(; β = β)).spike_count)
        end
        @test netspikes(; R = 1.0, w = 0.3) > 0                              # the connected net fires
        # gradient w.r.t. a synaptic WEIGHT flows through the surrogate-weighted scatter; matches finite diff
        gw = ForwardDiff.derivative(w -> netspikes(; R = 1.0, w = w), 0.3)
        h = 1.0e-4
        gw_fd = (netspikes(; R = 1.0, w = 0.3 + h) - netspikes(; R = 1.0, w = 0.3 - h)) / (2h)
        @test isfinite(gw)
        @test abs(gw) > 0
        @test gw ≈ gw_fd rtol = 1.0e-4
        # gradient w.r.t. a neuron parameter through the connected path
        gR = ForwardDiff.derivative(R -> netspikes(; R = R, w = 0.3), 1.0)
        @test isfinite(gR) && gR > 0
    end

    @testset "error paths (CPU-only, no plastic projections, canonical for now)" begin
        conn = fixed_prob(Dewdrop.CPU(), 8, 8, 0.2; weight = 0.5, delay = steps(1), seed = UInt64(1))
        plastic = DewdropNetwork(
            LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 0.0),
            8; input = 30.0, tspan = (0.0, 10.0),
            projection = Projection(DeltaSynapse(), conn; plasticity = STDP(; Aplus = 0.01, Aminus = 0.01, τplus = 20.0, τminus = 20.0))
        )
        @test_throws Exception init(plastic, FixedStep(0.1); backend = Differentiable())  # plastic scatter is a separate path
    end
end
