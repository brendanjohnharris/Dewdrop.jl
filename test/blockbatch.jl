using Dewdrop
using Test
using InteractiveUtils: subtypes

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

    @testset "block-diagonal keeps each member's synapse and learning rule" begin
        # The merged projection carries ONE synapse object, so members differing in a synaptic constant
        # would all run member 1's kinetics; and `plasticity` was dropped from the rebuilt Projection
        # entirely, silently turning a plastic member static.
        mkτ(τ; rule = nothing) = (
            nb = network(; tspan = (0.0, 200.0));
            population!(nb, :E, _lif(), 8; input = 0.45);
            project!(
                nb, :E => :E, CurrentSynapse(; τ = τ); p = 0.5, weight = 4.0, delay = steps(2),
                seed = UInt64(11), allow_self = true, plasticity = rule
            );
            build(nb)
        )
        solo = [sum(_solve(mkτ(1.0)).spike_count), sum(_solve(mkτ(40.0)).spike_count)]
        @test solo[1] != solo[2]                        # the synaptic constant really matters here
        bs = solve(batch([mkτ(1.0), mkτ(40.0)]), FixedStep(0.1); mode = :block, progress = false)
        @test [sum(bs[1]), sum(bs[2])] == solo

        # a plastic projection survives the stack: `Projection(syn, conn)` dropped the rule outright,
        # silently turning a plastic member static
        stdp = STDP(; Aplus = 0.02, Aminus = 0.005, τplus = 20.0, τminus = 20.0, wmin = 0.0, wmax = 8.0)
        bd = Dewdrop._block_diagonal([mkτ(5.0; rule = stdp), mkτ(5.0; rule = stdp)])
        @test all(p -> p.plasticity == stdp, bd.projections)
        psolo = sum(_solve(mkτ(5.0; rule = stdp)).spike_count)
        pb = solve(batch([mkτ(5.0; rule = stdp), mkτ(5.0; rule = stdp)]), FixedStep(0.1); mode = :block, progress = false)
        @test sum(pb[1]) == psolo && sum(pb[2]) == psolo
    end

    @testset "a wrapped (Heterogeneous) model routes away from :fused" begin
        # `:fused` diffs `fieldnames(typeof(model))`, but `_resolve_member` matches those keys against the
        # LEAF model's fields: for a wrapper they match nothing, so every member would silently run
        # member 1's parameters. Such a sweep must route to :multirun and stay correct there.
        h(vθ) = Heterogeneous(_lif(); Vθ = fill(vθ, 6))
        nh(vθ) = DewdropNetwork(h(vθ), 6; input = 0.35, tspan = (0.0, 50.0))
        solo = [sum(_solve(nh(-50.0)).spike_count), sum(_solve(nh(-58.0)).spike_count)]
        @test solo[1] != solo[2]                        # the members really differ
        bs = solve(batch([nh(-50.0), nh(-58.0)]), FixedStep(0.1); progress = false)
        @test bs.mode == :multirun
        @test [sum(bs[1]), sum(bs[2])] == solo
        @test [sum(bs[1]), sum(bs[2])] ==
            [sum(x) for x in solve(batch([nh(-50.0), nh(-58.0)]), FixedStep(0.1); mode = :multirun, threads = true, progress = false).spike_counts]
        # forcing it is refused rather than quietly wrong
        @test_throws ErrorException solve(batch([nh(-50.0), nh(-58.0)]), FixedStep(0.1); mode = :fused, progress = false)

        # a plain scalar model still takes :fused, and is still correct there
        ns(vθ) = DewdropNetwork(
            LIF(; τ = 20.0, EL = -70.0, Vθ = vθ, Vr = -60.0, R = 100.0, tref = 2.0),
            6; input = 0.35, tspan = (0.0, 50.0)
        )
        solos = [sum(_solve(ns(-50.0)).spike_count), sum(_solve(ns(-58.0)).spike_count)]
        bf = solve(batch([ns(-50.0), ns(-58.0)]), FixedStep(0.1); progress = false)
        @test bf.mode == :fused && [sum(bf[1]), sum(bf[2])] == solos
    end

    @testset "every mode carries the members' `stimuli`" begin
        # `stimulate!`-attached stimuli live in their own field, separate from `input`/`drive`/`noise`;
        # a rebuild that forgets it runs the member with no stimulus at all, silently and without error.
        ta = TimedArray(fill(0.3, 501); as = :current)
        mkst(I) = DewdropNetwork(_lif(), 6; input = I, tspan = (0.0, 50.0), stimuli = (ta,))
        solo = [sum(_solve(mkst(0.0)).spike_count), sum(_solve(mkst(0.05)).spike_count)]
        @test solo[1] > 0 && solo[2] > solo[1]          # the stimulus really drives the members
        for mode in (:shared, :multirun, :block)
            bs = solve(batch([mkst(0.0), mkst(0.05)]), FixedStep(0.1); mode = mode, progress = false)
            @test [sum(bs[1]), sum(bs[2])] == solo
        end
        bi = solve(batch(mkst(0.0); input = [0.0, 0.05]), FixedStep(0.1); progress = false)
        @test [sum(bi[1]), sum(bi[2])] == solo          # the input-sweep rebuild carries it too
    end

    @testset "a Float32 model solves in every mode" begin
        # `duration` is `nsteps * dt`, so it takes the MEMBER float type; a `Float64`-typed field would
        # reject every Float32 model after the whole simulation had already run.
        m32 = LIF(; τ = 20.0f0, EL = -70.0f0, Vθ = -50.0f0, Vr = -60.0f0, R = 100.0f0, tref = 2.0f0)
        n32(I) = DewdropNetwork(m32, 6; input = Float32(I), tspan = (0.0f0, 50.0f0))
        for mode in (:shared, :multirun, :block)
            bs = solve(batch([n32(0.3), n32(0.5)]), FixedStep(0.1f0); mode = mode, progress = false)
            @test nmembers(bs) == 2
            @test bs.duration isa Float32
            @test eltype(firing_rate(bs, 1)) === Float32
        end
    end

    @testset "spike counts come back host-resident in every mode" begin
        # `:shared`/`:fused` collect; `:block`/`:multirun` must too, or a device run returns device
        # vectors (and does not even construct, since the field is a `Vector{Vector{T}}`).
        for mode in (:shared, :multirun, :block)
            bs = solve(batch([_mk(0.3), _mk(0.3)]), FixedStep(0.1); mode = mode, progress = false)
            @test bs.spike_counts isa Vector{Vector{Int}}
        end
    end
end

# A drive carrying its own member-local wiring cannot be shared across the block: reusing member 1\'s
# `extconn` would leave members 2..B undriven, silently and with no error.
@testset "a replayed spike source is offset per member, not shared" begin
    arch = Dewdrop.CPU()
    ec = fixed_prob(arch, 3, 6, 1.0; weight = 1.0, delay = steps(1), seed = UInt64(1))
    ssa = SpikeSourceArray(DeltaSynapse(), ec, trues(3, 20))
    @test !Dewdrop._block_mergeable(ssa)                  # matches its PoissonSource sibling
    @test !Dewdrop._block_mergeable(PoissonSource(DeltaSynapse(), ec; rate = 50.0, seed = UInt64(1)))
    off = Dewdrop._offset_synapse(ssa, 6, 12, arch)
    @test off !== ssa                                     # wiring shifted into the member\'s block
    @test off.spikes === ssa.spikes                       # replay pattern shared verbatim
    @test Dewdrop.npost(off.extconn) == 12
    @test Array(off.extconn.post) == Array(ec.post) .+ 6
end

@testset "every member of a block batch keeps its own replayed drive" begin
    arch = Dewdrop.CPU()
    N, n_ext, dt, tspan = 4, 2, 0.1, (0.0, 40.0)
    nsteps = round(Int, (tspan[2] - tspan[1]) / dt)
    pat = trues(n_ext, nsteps)                            # both virtual sources fire every step
    function mk()
        ec = fixed_prob(arch, n_ext, N, 1.0; weight = 40.0, delay = steps(1), seed = UInt64(3))
        return DewdropNetwork(
            LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0), N;
            input = 0.0, tspan = tspan,
            projections = (Projection(SpikeSourceArray(DeltaSynapse(), ec, pat), Dewdrop._empty_csr(arch, N)),)
        )
    end
    bs = solve(batch([mk(), mk()]), FixedStep(dt); mode = :block, progress = false)
    @test nmembers(bs) == 2
    @test all(b -> sum(bs[b]) > 0, 1:2)                   # member 2 must not be silently undriven
    @test sum(bs[1]) == sum(bs[2])                        # identical members, identical drive
end

# A source synapse carries its own member-local wiring, so plasticity must refuse to wrap it and
# block-stacking must refuse to share it. Both follow from `is_source_synapse`; the wiring offset
# cannot be derived and needs a method per type. This fails if a new model misses either.
@testset "every synapse model is classified, and every source model can be offset" begin
    arch = Dewdrop.CPU()
    ec = fixed_prob(arch, 3, 6, 1.0; weight = 1.0, delay = steps(1), seed = UInt64(1))
    sources = (
        PoissonSource(DeltaSynapse(), ec; rate = 10.0, seed = UInt64(1)),
        SpikeSourceArray(DeltaSynapse(), ec, trues(3, 8)),
    )
    ordinary = (
        CurrentSynapse(τ = 5.0), DeltaSynapse(), ConductanceSynapse(τ = 5.0, Erev = 0.0),
        DualExpSynapse(τr = 1.0, τd = 5.0, Erev = 0.0),
        FrozenDualExpSynapse(τr = 1.0, τd = 5.0, Erev = 0.0),
    )
    for s in sources
        @test Dewdrop.is_source_synapse(s)
        @test !Dewdrop._plastic_wrappable(s)                  # derived
        @test !Dewdrop._block_mergeable(s)                    # derived
        @test Dewdrop._offset_synapse(s, 6, 12, arch) !== s   # defined per type, so check it exists
    end
    for s in ordinary
        @test !Dewdrop.is_source_synapse(s)
        @test Dewdrop._plastic_wrappable(s) && Dewdrop._block_mergeable(s)
        @test Dewdrop._offset_synapse(s, 6, 12, arch) === s   # nothing internal to shift
    end
    # a newly added synapse model has to be classified above, or this fails
    covered = union(Set(nameof(typeof(s)) for s in sources), Set(nameof(typeof(s)) for s in ordinary))
    @test Set(nameof.(subtypes(Dewdrop.AbstractSynapseModel))) == covered
end
