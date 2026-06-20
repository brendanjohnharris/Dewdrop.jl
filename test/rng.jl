using Dewdrop
using Test

# M0 contract 4 --- counter-based RNG keyed by (step, entity): a *pure* function,
# so draws are identical regardless of thread count or iteration order. This fixes
# the numerics and must be locked before any golden-seed regression test.
@testset "counter-based RNG" begin
    seed = UInt64(0xDEADBEEF)

    # purity: identical inputs -> identical draw
    @test Dewdrop.draw_uniform(Float64, seed, 5, 3) === Dewdrop.draw_uniform(Float64, seed, 5, 3)

    # distinct entity or step -> (almost surely) distinct draw
    @test Dewdrop.draw_uniform(Float64, seed, 5, 3) != Dewdrop.draw_uniform(Float64, seed, 5, 4)
    @test Dewdrop.draw_uniform(Float64, seed, 5, 3) != Dewdrop.draw_uniform(Float64, seed, 6, 3)

    # uniform range [0, 1)
    for i in 1:50
        u = Dewdrop.draw_uniform(Float64, seed, 1, i)
        @test 0.0 <= u < 1.0
    end

    # order independence: per-entity value does not depend on iteration order
    forward = [Dewdrop.draw_uniform(Float64, seed, 7, i) for i in 1:64]
    backward = reverse([Dewdrop.draw_uniform(Float64, seed, 7, i) for i in 64:-1:1])
    @test forward == backward

    # thread independence: parallel fill == sequential fill (guards shared state)
    N = 4096
    seq = [Dewdrop.draw_uniform(Float64, seed, 2, i) for i in 1:N]
    par = Vector{Float64}(undef, N)
    Threads.@threads for i in 1:N
        par[i] = Dewdrop.draw_uniform(Float64, seed, 2, i)
    end
    @test par == seq

    # parametric float type
    @test Dewdrop.draw_uniform(Float32, seed, 1, 1) isa Float32
    @test 0.0f0 <= Dewdrop.draw_uniform(Float32, seed, 1, 1) < 1.0f0

    # zero-allocation guard: the whole counter-RNG reproducibility contract relies on
    # the mutable Philox being SROA'd to registers (no heap box). Guard both float
    # widths so a compiler-version regression that defeats escape analysis fails CI.
    # Use literal seeds here, NOT the captured `seed` above: the earlier
    # `Threads.@threads` closure boxes `seed`, which would make this measure test-scope
    # boxing rather than `draw_uniform` itself.
    Dewdrop.draw_uniform(Float32, UInt64(0x1234), 1, 1)  # warm
    Dewdrop.draw_uniform(Float64, UInt64(0x1234), 1, 1)
    @test @allocated(Dewdrop.draw_uniform(Float32, UInt64(0x1234), 2, 3)) == 0
    @test @allocated(Dewdrop.draw_uniform(Float64, UInt64(0x1234), 2, 3)) == 0
end

# Counter-based Poisson sampling (inverse-CDF from one uniform) for per-neuron external
# drive --- pure, one uniform per sample, GPU-kernel-safe.
@testset "counter-based Poisson sampling" begin
    seed = UInt64(7)
    λ = 3.0

    # inverse-CDF basics
    @test Dewdrop.poisson_count(2.0, 0.0) == 0          # u = 0 → count 0
    @test Dewdrop.poisson_count(2.0, 0.99) > 2          # far tail → large count
    @test Dewdrop.poisson_count(0.0, 0.5) == 0          # λ = 0 → always 0

    # determinism + non-negativity
    @test Dewdrop.draw_poisson(λ, seed, 1, 1) === Dewdrop.draw_poisson(λ, seed, 1, 1)

    # mean ≈ variance ≈ λ over many independent samples
    N = 100_000
    s = [Dewdrop.draw_poisson(λ, seed, 1, i) for i in 1:N]
    @test all(≥(0), s)
    mean = sum(s) / N
    @test abs(mean - λ) < 0.05
    var = sum(x -> (x - mean)^2, s) / N
    @test abs(var - λ) < 0.1

    # zero allocation on the hot path
    Dewdrop.draw_poisson(2.0, UInt64(1), 1, 1)
    @test @allocated(Dewdrop.draw_poisson(2.0, UInt64(1), 2, 3)) == 0
end
