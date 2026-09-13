using Dewdrop
using Test
using Statistics

# Counter-based RNG keyed by (step, entity): a *pure* function,
# so draws are identical regardless of thread count or iteration order. This fixes
# the numerics and must be locked before any fixed-seed regression test.
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

    # zero-allocation guard: the whole counter-RNG reproducibility guarantee relies on
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
# drive: pure, one uniform per sample, GPU-kernel-safe.
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

    # zero allocation in the inner loop
    Dewdrop.draw_poisson(2.0, UInt64(1), 1, 1)
    @test @allocated(Dewdrop.draw_poisson(2.0, UInt64(1), 2, 3)) == 0
end

# Each builder keys its draws on a `(step, entity)` pair taken from whatever indices it has to hand, so
# `random_positions` (neuron, dim), `distance_prob` (pre, post) and `fixed_prob` (gap, pre) all land in
# one key space. Sharing a seed across them would correlate geometry with connectivity; each entry point
# mixes in its own tag (src/RNG.jl).
@testset "counter-RNG domain separation across builders" begin
    seed = UInt64(12345)
    tags = (
        Dewdrop.DOMAIN_POSITIONS, Dewdrop.DOMAIN_DISTPROB, Dewdrop.DOMAIN_DISTCOUNT,
        Dewdrop.DOMAIN_FIXEDPROB, Dewdrop.DOMAIN_WEIGHTS, Dewdrop.DOMAIN_POISSON,
        Dewdrop.DOMAIN_INITIALV, Dewdrop.DOMAIN_NOISE, Dewdrop.DOMAIN_DRIVE,
        Dewdrop.DOMAIN_INHOMPOISSON,
    )
    @test length(unique(tags)) == length(tags)
    ks = [Dewdrop.domain_seed(seed, t) for t in tags]
    @test length(unique(ks)) == length(ks)
    # the shared `(step, entity)` slot now yields a different draw at every call site
    @test length(unique(Dewdrop.draw_uniform(Float64, k, 5, 1) for k in ks)) == length(ks)
    @test Dewdrop.domain_seed(seed, first(tags)) == Dewdrop.domain_seed(seed, first(tags))   # pure

    # the concrete case: one seed reused for placement and for wiring
    pos = random_positions(64, (1.0, 1.0); seed = seed)
    conn = distance_prob(
        Dewdrop.CPU(), pos; kernel = gaussian_kernel(0.3), weight = 1.0,
        delay = steps(1), seed = seed
    )
    @test Dewdrop.nedges(conn) > 0
    @test pos == random_positions(64, (1.0, 1.0); seed = seed)          # still reproducible
    @test pos != random_positions(64, (1.0, 1.0); seed = UInt64(999))   # still seed-dependent
    # a neuron\'s coordinate is no longer the draw that decides its first connections
    @test pos != random_positions(64, (1.0, 1.0); seed = Dewdrop.domain_seed(seed, Dewdrop.DOMAIN_DISTPROB))
end

# The per-step stimuli collide harder than the builders do: `WhiteNoise` and `PoissonDrive` both key on
# `(step, neuron)` and both default to `seed = 0`, and `draw_normal` builds its Box-Muller radius from
# the very word `draw_uniform` returns. Undefended, the noise magnitude is then a deterministic
# decreasing function of the drive's draw (measured cor = -0.67), not an independent process.
@testset "per-step stimuli draw on separated streams" begin
    n, d = WhiteNoise(0.5), PoissonDrive(rate = 10.0, weight = 0.5)
    ip = InhomogeneousPoisson(5.0; weight = 0.5)
    @test length(unique((n.seed, d.seed, ip.seed))) == 3        # one user seed → three streams
    @test n.seed != UInt64(0)                                   # and none is the raw seed
    pts = [(s, i) for s in 1:200, i in 1:50]
    u = [Dewdrop.draw_uniform(Float64, d.seed, s, i) for (s, i) in pts]
    z = [Dewdrop.draw_normal(Float64, n.seed, s, i) for (s, i) in pts]
    @test abs(cor(vec(u), abs.(vec(z)))) < 0.05
    # the shared-word signature of the collision: |z| == sqrt(-2log u) * |cos θ| ≤ sqrt(-2log u)
    @test !all(abs.(vec(z)) .<= sqrt.(-2 .* log.(vec(u))) .+ 1.0e-9)
end
