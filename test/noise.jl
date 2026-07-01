using Dewdrop
using Test
using Adapt
using JLArrays

# SDE noise: an additive-voltage white-noise source (`WhiteNoise`) attached to the
# network, integrated by the EXACT Ornstein--Uhlenbeck discretization (the noise analogue of the
# engine's exact drift propagator). Opt-in and compiling away when absent (the `Nothing`
# strong-zero idiom). The increment is the counter-based Gaussian `draw_normal`, keyed by
# (seed, step, neuron) --- pure, allocation-free, GPU-safe, reproducible.

_meanvar(x) = (μ = sum(x) / length(x); (μ, sum(y -> (y - μ)^2, x) / length(x)))

@testset "draw_normal (counter-based Gaussian)" begin
    seed = UInt64(0x00C0FFEE)
    # purity: identical inputs -> identical draw
    @test Dewdrop.draw_normal(Float64, seed, 5, 3) === Dewdrop.draw_normal(Float64, seed, 5, 3)
    # distinct step/entity -> (almost surely) distinct draw
    @test Dewdrop.draw_normal(Float64, seed, 5, 3) != Dewdrop.draw_normal(Float64, seed, 5, 4)
    @test Dewdrop.draw_normal(Float64, seed, 5, 3) != Dewdrop.draw_normal(Float64, seed, 6, 3)
    # 4-arg ≡ 5-arg(batch = 0): the scalar path is the batch-0 column
    @test Dewdrop.draw_normal(Float64, seed, 5, 3) === Dewdrop.draw_normal(Float64, seed, 5, 3, 0)

    # standard-normal statistics over many independent draws
    N = 200_000
    s = [Dewdrop.draw_normal(Float64, seed, 1, i) for i in 1:N]
    μ, v = _meanvar(s)
    @test abs(μ) < 0.02
    @test abs(v - 1.0) < 0.03

    # distinct batches -> independent (uncorrelated) streams
    a = [Dewdrop.draw_normal(Float64, seed, 1, i, 0) for i in 1:N]
    b = [Dewdrop.draw_normal(Float64, seed, 1, i, 1) for i in 1:N]
    μa = _meanvar(a)[1]; μb = _meanvar(b)[1]
    cab = sum((a .- μa) .* (b .- μb)) / N
    @test abs(cab) < 0.02

    # parametric float type + finiteness (the log(0) guard holds)
    @test Dewdrop.draw_normal(Float32, seed, 1, 1) isa Float32
    @test all(isfinite, (Dewdrop.draw_normal(Float32, seed, k, 1) for k in 1:10_000))

    # zero-allocation hot path (both float widths)
    Dewdrop.draw_normal(Float32, UInt64(0x1234), 1, 1)
    Dewdrop.draw_normal(Float64, UInt64(0x1234), 1, 1)
    @test @allocated(Dewdrop.draw_normal(Float32, UInt64(0x1234), 2, 3)) == 0
    @test @allocated(Dewdrop.draw_normal(Float64, UInt64(0x1234), 2, 3)) == 0
end

@testset "WhiteNoise OU stationary variance (exact, dt-invariant)" begin
    # A purely subthreshold LIF (threshold pushed to +∞, no input) is an Ornstein--Uhlenbeck
    # process with stationary variance σ²τ/2. The exact-OU noise scaling reproduces this EXACTLY
    # at any dt; the literal Euler--Maruyama scaling σ√dt would be dt-biased --- the dt-invariance
    # assertion is what pins the discretization choice.
    τ = 20.0
    m = LIF(; τ = τ, EL = -65.0, Vθ = 1.0e6, Vr = -65.0, R = 1.0, tref = 0.0)
    σ = 0.5
    analytic = σ^2 * τ / 2
    vars = Float64[]
    for dt in (0.1, 0.05)
        prob = DewdropNetwork(
            m, 200; input = 0.0, tspan = (0.0, 2000.0),
            noise = WhiteNoise(σ; seed = UInt64(42))
        )
        sol = solve(prob, FixedStep(dt); record = (V = Trace(:V),))
        V = sol.record.V.data
        burn = size(V, 2) ÷ 5
        _, vv = _meanvar(vec(V[:, (burn + 1):end]))
        push!(vars, vv)
        @test isapprox(vv, analytic; rtol = 0.1)
    end
    @test isapprox(vars[1], vars[2]; rtol = 0.08)   # dt-invariance (selects OU over EM)
end

@testset "WhiteNoise: off ≡ no-noise; reproducible; raises rate" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    dt, tend = 0.1, 500.0
    # the noise = nothing path is identical to a build with no noise kwarg (regression guard)
    a = solve(DewdropNetwork(m, 40; input = 0.0, tspan = (0.0, tend)), FixedStep(dt))
    b = solve(DewdropNetwork(m, 40; input = 0.0, tspan = (0.0, tend), noise = nothing), FixedStep(dt))
    @test a.spike_count == b.spike_count

    # reproducible: same seed -> identical run
    p = DewdropNetwork(m, 40; input = 15.0, tspan = (0.0, tend), noise = WhiteNoise(2.0; seed = UInt64(7)))
    @test solve(p, FixedStep(dt)).spike_count == solve(p, FixedStep(dt)).spike_count

    # subthreshold mean (V∞ = 17 < Vθ = 20) -> silent without noise, fires with it; more σ -> more
    sub = 17.0
    r0 = sum(solve(DewdropNetwork(m, 200; input = sub, tspan = (0.0, tend)), FixedStep(dt)).spike_count)
    r1 = sum(
        solve(
            DewdropNetwork(
                m, 200; input = sub, tspan = (0.0, tend),
                noise = WhiteNoise(1.0; seed = UInt64(1))
            ), FixedStep(dt)
        ).spike_count
    )
    r2 = sum(
        solve(
            DewdropNetwork(
                m, 200; input = sub, tspan = (0.0, tend),
                noise = WhiteNoise(2.0; seed = UInt64(1))
            ), FixedStep(dt)
        ).spike_count
    )
    @test r0 == 0
    @test r2 > r1 > 0
end

@testset "WhiteNoise: CPU broadcast ≡ JLArray fused; allocation-free" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    dt = 0.1
    prob = DewdropNetwork(
        m, 64; input = 16.0, tspan = (0.0, 50.0),
        noise = WhiteNoise(2.0; seed = UInt64(99))
    )
    # draw_normal is a pure function and JLArrays is CPU-backed, so the fused megakernel matches
    # the broadcast path (same libm exp/log/cos/sqrt) --- bit-identical spikes, V within ULPs.
    cpu = init(prob, FixedStep(dt))
    gpu = adapt(JLArray, init(prob, FixedStep(dt)))
    for _ in 1:500
        step!(cpu); step!(gpu)
    end
    @test Array(gpu.spike_count) == cpu.spike_count
    @test Array(gpu.state.state.V) ≈ cpu.state.state.V
    @test sum(cpu.spike_count) > 0   # the regime actually fires (noise-driven)

    # allocation-free hot loop with noise active
    warm = init(prob, FixedStep(dt)); step!(warm)
    @test @allocated(step!(warm)) == 0
end

@testset "WhiteNoise: batched ≡ scalar reference (per-column streams)" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    dt = 0.1
    prob = DewdropNetwork(
        m, 32; input = 16.0, tspan = (0.0, 80.0),
        noise = WhiteNoise(2.0; seed = UInt64(5))
    )
    B = 4
    # streams all-zero -> every column shares the scalar (batch-0) noise draw, so each column
    # equals the scalar solve (the bit-exact ensemble reference, as for the Poisson drive).
    bsol = solve(prob, FixedStep(dt); batch = B, streams = zeros(Int, B))
    ssol = solve(prob, FixedStep(dt))
    for b in 1:B
        @test bsol.spike_count[:, b] == ssol.spike_count
    end
    # default streams (0:B-1) -> independent columns (not all identical to column 1)
    dsol = solve(prob, FixedStep(dt); batch = B)
    @test any(dsol.spike_count[:, b] != dsol.spike_count[:, 1] for b in 2:B)
end
