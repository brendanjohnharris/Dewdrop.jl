using Dewdrop
using Test

# Execution backends (Auto/Serial/Fused/Turbo): HOW the per-step engine runs, orthogonal to the
# CPU/GPU architecture. `Serial` = the per-phase broadcast (bit-reproducible baseline); `Fused` = the
# tight single-pass loop (CPU) / megakernel (GPU), bit-identical and faster; `Turbo` = SIMD
# (LoopVectorization ext, opt-in); `Auto` = the advisor picks. See src/Backends.jl.

@testset "execution backends" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    N = 64
    conn = fixed_prob(Dewdrop.CPU(), N, N, 0.1; weight = 0.5, delay = steps(2), seed = UInt64(1))
    prob = DewdropNetwork(
        m, N; input = 30.0, tspan = (0.0, 100.0),
        projection = Projection(DeltaSynapse(), conn)
    )   # exact weights → scatter order-independent

    @testset "types exported + are SimBackends" begin
        for b in (Dewdrop.Auto(), Serial(), Fused(), Turbo())
            @test b isa SimBackend
        end
    end

    @testset "Serial ≡ Fused on CPU (bit-identical)" begin
        ss = solve(prob, FixedStep(0.1); backend = Serial())
        sf = solve(prob, FixedStep(0.1); backend = Fused())
        @test sum(ss.spike_count) > 0
        @test sf.spike_count == ss.spike_count           # same spikes
        @test sf.state.state.V == ss.state.state.V       # bit-identical V (exact-weight delta scatter)
    end

    # A synapse whose `_syn_membrane` reads `v` (today only `FrozenDualExpSynapse`) evaluates
    # `g·(Erev − V)` itself, so it is the only case that can see a different V on the two paths. Both
    # must use the START-of-step V, with :jump deliveries and :kick stimuli applied afterwards
    # (BrainPy's `sum_current_inputs` before `sum_delta_inputs`; Brian2's `groups` slot before
    # `synapses`). The delta-only test above cannot catch this: delta never reads v.
    @testset "Serial ≡ Fused for a V-reading synapse (start-of-step V)" begin
        fz = FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
        conn2 = fixed_prob(Dewdrop.CPU(), N, N, 0.1; weight = 0.5, delay = steps(3), seed = UInt64(7))
        both(p) = (solve(p, FixedStep(0.1); backend = Serial()), solve(p, FixedStep(0.1); backend = Fused()))

        kick = DewdropNetwork(
            m, N; input = 30.0, tspan = (0.0, 100.0),
            projection = Projection(fz, conn), stimuli = (FunctionalKick((i, t) -> 0.05),)
        )
        sk, fk = both(kick)
        @test sum(sk.spike_count) > 0
        @test sk.spike_count == fk.spike_count
        @test sk.state.state.V == fk.state.state.V

        # …and with an instantaneous jump alongside it, in both projection orders.
        first_fz = DewdropNetwork(m, N; input = 30.0, tspan = (0.0, 100.0),
            projections = (Projection(fz, conn), Projection(DeltaSynapse(), conn2)))
        first_delta = DewdropNetwork(m, N; input = 30.0, tspan = (0.0, 100.0),
            projections = (Projection(DeltaSynapse(), conn2), Projection(fz, conn)))
        sa, fa = both(first_fz)
        sb, fb = both(first_delta)
        @test sa.state.state.V == fa.state.state.V
        @test sb.state.state.V == fb.state.state.V
        @test sa.state.state.V == sb.state.state.V     # projection order must not matter
        @test sa.spike_count == sb.spike_count
    end

    @testset "Auto resolution" begin
        @test Dewdrop._resolve_backend(Serial(), prob) isa Serial   # explicit passes through
        @test Dewdrop._resolve_backend(Dewdrop.Auto(), prob) isa Fused      # canonical CPU → work-aware tight loop
        big = DewdropNetwork(m, 20_000; input = 30.0, tspan = (0.0, 1.0))
        @test Dewdrop._resolve_backend(Dewdrop.Auto(), big) isa Fused       # large net → Fused (threads itself)
        hm = Heterogeneous(m; Vθ = fill(m.Vθ, N))
        hp = DewdropNetwork(hm, N; input = 30.0, tspan = (0.0, 50.0))
        @test Dewdrop._resolve_backend(Dewdrop.Auto(), hp) isa Fused        # hetero → Fused
    end

    @testset "step=:fused deprecated alias maps to Fused" begin
        a = solve(prob, FixedStep(0.1); step = :fused)
        b = solve(prob, FixedStep(0.1); backend = Fused())
        @test a.spike_count == b.spike_count
        @test solve(prob, FixedStep(0.1); step = :auto).spike_count == solve(prob, FixedStep(0.1)).spike_count
        @test_throws Exception init(prob, FixedStep(0.1); step = :bogus)
    end

    @testset "Serial rejects heterogeneous; Fused/Auto accept" begin
        hm = Heterogeneous(m; Vθ = fill(m.Vθ, N))
        hp = DewdropNetwork(hm, N; input = 30.0, tspan = (0.0, 50.0))
        @test_throws Exception init(hp, FixedStep(0.1); backend = Serial())
        @test sum(solve(hp, FixedStep(0.1); backend = Fused()).spike_count) == 128   # Fused works
        @test sum(solve(hp, FixedStep(0.1)).spike_count) == 128                       # Auto → Fused
    end

    # (the Turbo backend, numerics + with-extension error paths, is tested in test/turbo.jl,
    # which loads LoopVectorization; it must run AFTER this file, which assumes the ext is dormant.)

    @testset "allocation-free Serial dense phases" begin
        # unconnected net → no sparse scatter (the threaded scatter allocates the @threads tasks at
        # >1 thread regardless of backend); this isolates the Serial dense-phase allocation-free guarantee.
        unconn = DewdropNetwork(m, N; input = 30.0, tspan = (0.0, 50.0))
        warm = init(unconn, FixedStep(0.1); backend = Serial())
        step!(warm)
        @test @allocated(step!(warm)) == 0
    end
end

# The guard against recording an accumulator a backend never materialises matched a bare `Trace`
# only, so wrapping the same variable in an `Aggregate` slipped past it and returned silent zeros.
@testset "accumulator-record guard sees through Aggregate" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    prob = DewdropNetwork(m, 3; input = 0.5, tspan = (0.0, 20.0))
    for spec in (Trace(:itot), Aggregate(Trace(:itot), :sum), Aggregate(Trace(:gtot), :mean))
        @test_throws ArgumentError solve(
            prob, FixedStep(0.1); backend = Differentiable(), record = (r = spec,)
        )
    end
    @test solve(prob, FixedStep(0.1); backend = Differentiable(),   # a non-accumulator aggregate is fine
                record = (r = Aggregate(Trace(:V), :mean),)) isa Any
end
