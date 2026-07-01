using Dewdrop
using Test

# Batching (src/BlockBatch.jl): run B network members together via `batch(...)` → `solve` → `BatchSolution`.
# Three execution modes, auto-routed by what varies (and forceable via `mode=`):
#   :shared:   fused shared-CSR ensemble (one connectome, B (N,B) columns; vary input/v0/seed).
#   :multirun: B separate scalar solves SHARING the connectome array (vary the model; Mode-A memory, no kernel).
#   :block:    block-diagonal stack into one network (distinct topology).
# With NO drive the members are deterministic, so every mode must give the SAME per-member result.

_lif() = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
_mk(I; seed = 1) = (
    nb = network(; tspan = (0.0, 50.0));
    population!(nb, :E, _lif(), 6; input = I);
    project!(nb, :E => :E, DeltaSynapse(); p = 0.5, weight = 0.3, delay = steps(2), seed = UInt64(seed), allow_self = true);
    build(nb)
)
_solve(net) = solve(net, FixedStep(0.1); progress = false)

@testset "batching" begin
    @testset "block-diagonal engine: each block ≡ that member standalone (deterministic)" begin
        n1, n2 = _mk(0.3), _mk(0.5)
        bd = Dewdrop._block_diagonal([n1, n2])
        @test bd.n == 12
        sol = _solve(bd)
        @test sol.spike_count[1:6] == _solve(n1).spike_count
        @test sol.spike_count[7:12] == _solve(n2).spike_count
        @test sum(_solve(n1).spike_count) != sum(_solve(n2).spike_count)   # the members really differ
    end

    @testset "block-diagonal carries EACH member's streaming drive (not just member 1's)" begin
        # Distinct-topology members that DEPEND on an external Poisson drive (input = 0): block-stacking must
        # offset every member's `PoissonSource` into its own block. Reusing member 1's drive (the bug) leaves
        # members 2..B undriven → silent. Each member's batched result must equal its standalone, bit-for-bit.
        mkd(; cseed, dseed) = (
            nb = network(; tspan = (0.0, 50.0));
            population!(nb, :E, _lif(), 8; input = 0.0);
            project!(
                nb, :E => :E, DeltaSynapse(); p = 0.3, weight = 0.5, delay = steps(2),
                seed = UInt64(cseed), allow_self = false
            );
            drive!(
                nb, :E, DeltaSynapse(); rate = 150.0, n_ext = 10, p = 0.5, weight = 1.2,
                delay = steps(1), seed = UInt64(dseed)
            );
            build(nb)
        )
        n1, n2 = mkd(cseed = 1, dseed = 101), mkd(cseed = 2, dseed = 202)
        s1, s2 = _solve(n1).spike_count, _solve(n2).spike_count
        @test sum(s2) > 0                       # member 2 is active standalone (its drive sustains it)
        @test s1 != s2                          # the members really differ
        bs = solve(batch([n1, n2]), FixedStep(0.1); progress = false)
        @test bs.mode == :block
        @test bs[1] == s1                       # member 1 was already correct
        @test bs[2] == s2                       # the fix: member 2 driven by ITS OWN source, not member 1's
    end

    @testset "batch input forms + per-member addressing (sol[b])" begin
        bs = solve(batch([_mk(0.3), _mk(0.5)]), FixedStep(0.1); progress = false)
        @test nmembers(bs) == 2
        @test bs[1] == _solve(_mk(0.3)).spike_count
        @test bs[2] == _solve(_mk(0.5)).spike_count
        @test firing_rate(bs, 1) == _solve(_mk(0.3)).spike_count ./ 50.0
        # generator
        @test nmembers(solve(batch((b, i) -> _mk(0.2 + 0.1 * i), nothing; n = 3), FixedStep(0.1); progress = false)) == 3
        # model-parameter sweep (zipped + cartesian)
        bsw = solve(batch(_mk(0.3); τ = [10.0, 20.0, 40.0]), FixedStep(0.1); progress = false)
        @test nmembers(bsw) == 3 && bsw[1] != bsw[3]
        @test nmembers(batch(_lif(); τ = [10.0, 20.0], Vθ = [-50.0, -45.0], cartesian = true)) == 4
    end

    @testset "auto mode routing by what varies" begin
        base = _mk(0.3)
        @test solve(batch(base; τ = [10.0, 20.0]), FixedStep(0.1); progress = false).mode == :fused          # shared conn, vary model (uniform type)
        @test solve(batch(base; input = [0.25, 0.35]), FixedStep(0.1); progress = false).mode == :shared      # shared model+conn, vary input
        @test solve(batch([_mk(0.3; seed = 1), _mk(0.3; seed = 2)]), FixedStep(0.1); progress = false).mode == :block  # distinct topology
    end

    @testset "all modes agree on per-member results (no drive → deterministic)" begin
        base = _mk(0.3)
        # shared connectome, different models → :multirun, :fused, :block must all match per member
        bf = batch([base, Dewdrop._apply_sweep(base, (; τ = 40.0))])
        sm = solve(bf, FixedStep(0.1); mode = :multirun, progress = false)
        sf = solve(bf, FixedStep(0.1); mode = :fused, progress = false)
        sblk = solve(bf, FixedStep(0.1); mode = :block, progress = false)
        @test sm[1] == sblk[1] && sm[2] == sblk[2]
        @test sf[1] == sblk[1] && sf[2] == sblk[2]               # fused Mode A agrees too
        # shared model+connectome, different input → :shared (Mode 0) vs :block must match per member
        bi = batch(base; input = [0.3, 0.5])
        ss, sblk2 = solve(bi, FixedStep(0.1); mode = :shared, progress = false),
            solve(bi, FixedStep(0.1); mode = :block, progress = false)
        @test ss[1] == sblk2[1] && ss[2] == sblk2[2]
    end

    @testset "threaded multi-run ≡ sequential; mixed model types → :multirun" begin
        base = _mk(0.3)
        bf = batch([base, Dewdrop._apply_sweep(base, (; τ = 40.0))])
        seq = solve(bf, FixedStep(0.1); mode = :multirun, threads = false, progress = false)
        par = solve(bf, FixedStep(0.1); mode = :multirun, threads = true, progress = false)
        @test seq[1] == par[1] && seq[2] == par[2]      # threaded == sequential (bit-identical, Serial ≡ Fused)

        # different model TYPES sharing the connectome → auto :multirun (fused needs a uniform type)
        adex = AdEx(;
            C = 281.0, gL = 30.0, EL = -70.0, VT = -50.0, ΔT = 2.0, Vr = -60.0, Vpeak = -40.0,
            a = 4.0, b = 80.0, τw = 144.0, tref = 2.0
        )
        adexnet = DewdropNetwork(
            adex, base.n; input = base.input, tspan = base.tspan, arch = base.arch,
            schedule = base.schedule, projections = base.projections, drive = base.drive, noise = base.noise,
            subpops = base.subpops, positions = base.positions, projlabels = base.projlabels
        )
        mixed = solve(batch([base, adexnet]), FixedStep(0.1); progress = false)
        @test mixed.mode == :multirun && nmembers(mixed) == 2
    end

    @testset "mismatched projection structure errors clearly" begin
        a = network(; tspan = (0.0, 20.0)); population!(a, :E, _lif(), 4; input = 0.3)
        @test_throws ErrorException Dewdrop._block_diagonal([_mk(0.3), build(a)])
    end
end
