using Dewdrop
using Test

# Deferred network spec (src/NetworkSpec.jl): an immutable, run-parameter-free description that
# materialises into a `DewdropNetwork` only at solve time, when `dt`/`tspan` are known. Two reps under
# `AbstractNetworkSpec`: `freeze(builder)` (structured) and `defer(constructor; kw...)` (thunk).
# Behaviour: `materialize(spec, FixedStep(dt); tspan)` is a pure (spec, run-params) → DewdropNetwork, and
# `solve(spec, …)` is bit-identical to solving the materialised network.

_lif() = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)

# a deferred build function: takes the captured kwargs PLUS the injected `tspan`/`dt` (it may ignore dt)
_mknet(; n, I, tspan, dt) = DewdropNetwork(_lif(), n; input = I, tspan = tspan)

@testset "network spec (deferred materialisation)" begin
    @testset "freeze(builder): materialise ≡ build; solve ≡ built solve (bit-identical)" begin
        mk() = (
            nb = network(; tspan = (0.0, 50.0));
            population!(nb, :E, _lif(), 6; input = 0.3);
            population!(nb, :I, _lif(), 4; input = 0.3);
            project!(nb, :E => :I, DeltaSynapse(); p = 1.0, weight = 0.5, delay = steps(1), seed = UInt64(1));
            nb
        )
        spec = freeze(mk())
        @test spec isa Dewdrop.AbstractNetworkSpec

        built = build(mk())
        mat = materialize(spec, FixedStep(0.1); tspan = (0.0, 50.0))
        @test mat isa DewdropNetwork
        @test mat.n == built.n
        @test keys(mat.subpops) == keys(built.subpops)

        s_spec = solve(spec, FixedStep(0.1))                # uses the frozen builder's default tspan
        s_built = solve(built, FixedStep(0.1))
        @test s_spec.spike_count == s_built.spike_count
        @test s_spec.state.state.V == s_built.state.state.V
    end

    @testset "freeze: tspan default + override (reuse across durations)" begin
        nb = network(; tspan = (0.0, 50.0))
        population!(nb, :E, _lif(), 5; input = 0.3)
        spec = freeze(nb)
        @test solve(spec, FixedStep(0.1)).nsteps == 500                 # default tspan = 50 ms
        @test solve(spec, FixedStep(0.1); tspan = (0.0, 100.0)).nsteps == 1000   # override
    end

    @testset "defer(constructor): materialise + solve ≡ eager call" begin
        spec = defer(_mknet; n = 8, I = 0.3)
        @test spec isa Dewdrop.AbstractNetworkSpec
        mat = materialize(spec, FixedStep(0.1); tspan = (0.0, 40.0))
        @test mat isa DewdropNetwork && mat.n == 8

        s_spec = solve(spec, FixedStep(0.1); tspan = (0.0, 40.0))
        s_eager = solve(_mknet(; n = 8, I = 0.3, tspan = (0.0, 40.0), dt = 0.1), FixedStep(0.1))
        @test s_spec.spike_count == s_eager.spike_count
        @test s_spec.state.state.V == s_eager.state.state.V
    end

    @testset "deferred spec needs tspan (no captured default) → clear error" begin
        spec = defer(_mknet; n = 5, I = 0.3)
        @test_throws ArgumentError solve(spec, FixedStep(0.1))
    end

    @testset "batch=B flows through a spec (Mode-0 shared-CSR ensemble)" begin
        nb = network(; tspan = (0.0, 30.0))
        population!(nb, :E, _lif(), 10; input = 0.3)
        spec = freeze(nb)
        sol = solve(spec, FixedStep(0.1); batch = 4)
        @test size(sol.spike_count) == (10, 4)
    end

    @testset "build(spec; dt, tspan) escape hatch returns a DewdropNetwork" begin
        spec = defer(_mknet; n = 7, I = 0.3)
        net = build(spec; dt = 0.1, tspan = (0.0, 20.0))
        @test net isa DewdropNetwork && net.n == 7
    end

    @testset "show renders specs without dumping arrays" begin
        rich(x) = sprint((io, y) -> show(io, MIME"text/plain"(), y), x; context = :color => false)
        nb = network(; tspan = (0.0, 50.0))
        population!(nb, :E, _lif(), 6)
        population!(nb, :I, _lif(), 4)
        project!(nb, :E => :I, DeltaSynapse(); p = 1.0, weight = 0.5, delay = steps(1), seed = UInt64(1))
        ss = rich(freeze(nb))
        @test occursin("NetworkSpec", ss)
        @test occursin("E", ss) && occursin("projections", ss)
        ds = rich(defer(_mknet; n = 3, I = 0.3))
        @test occursin("NetworkSpec", ds)
    end
end

@testset "delay in physical time (ms), resolved at the solve dt" begin
    mk(d) = (
        nb = network(; tspan = (0.0, 20.0)); population!(nb, :E, _lif(), 4; input = 0.3);
        project!(nb, :E => :E, DeltaSynapse(); p = 1.0, weight = 0.0, delay = d, seed = UInt64(1), allow_self = true);
        build(nb)
    )
    maxd(net, dt) = maximum(init(net, FixedStep(dt)).syns[1].conn.delay)

    @testset "plain number = ms → step count tracks dt, physical delay fixed" begin
        @test maxd(mk(1.0), 0.1) == Dewdrop._ms_to_steps(1.0, 0.1)
        @test maxd(mk(1.0), 0.05) == Dewdrop._ms_to_steps(1.0, 0.05)         # same 1 ms, twice the steps
        @test maxd(mk(1.0), 0.05) == 2 * maxd(mk(1.0), 0.1)
        @test eltype(init(mk(1.0), FixedStep(0.1)).syns[1].conn.delay) <: Integer   # resolved to steps
    end

    @testset "steps(n) = exact step count, dt-independent" begin
        @test maxd(mk(steps(7)), 0.1) == 7
        @test maxd(mk(steps(7)), 0.99) == 7
        @test_throws ArgumentError steps(0)        # must deliver in a later step
    end
end

@testset "post-build connectivity adjust hook" begin
    nb = network(; tspan = (0.0, 20.0))
    population!(nb, :E, _lif(), 5; input = 0.3)
    population!(nb, :I, _lif(), 5; input = 0.3)
    project!(
        nb, :E => :I, DeltaSynapse(); p = 1.0, weight = 0.5, delay = steps(1), seed = UInt64(1),
        adjust = c -> (c.weight .*= 2)
    )                # doubles the built weights in place
    net = build(nb)
    @test all(==(1.0), net.projections[1].conn.weight)   # 0.5 doubled post-build

    # a 2-arg adjuster `(conn, ctx)` additionally receives the resolved `(; sources, targets)` ranges;
    # `correlate_weights` uses ctx.targets to normalise the in-degree over the destination sub-population.
    nb2 = network(; tspan = (0.0, 20.0))
    population!(nb2, :E, _lif(), 8; input = 0.3)
    population!(nb2, :I, _lif(), 8; input = 0.3)
    project!(
        nb2, :E => :I, DeltaSynapse(); p = 0.6, weight = 1.0, delay = steps(1), seed = UInt64(2),
        adjust = correlate_weights(0.1; seed = UInt64(7))
    )
    w = build(nb2).projections[1].conn.weight
    @test !all(==(1.0), w) && all(>(0.0), w)             # in-degree-normalised, not the raw 1.0
    @test build(nb2).projections[1].conn.weight == w     # deterministic / reproducible
end
