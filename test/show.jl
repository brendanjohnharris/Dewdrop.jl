using Dewdrop
using Test

# Hierarchical `show` (src/Show.jl): every Dewdrop object renders cleanly in the REPL, reflecting
# the structure it actually has --- a leaf model as a flat parameter sheet, a composite model or a
# whole network/solution as an indented tree. All assertions use `:color => false` for deterministic,
# escape-code-free strings, and lean on robust substrings (not exact column positions).

rich(x) = sprint((io, y) -> show(io, MIME"text/plain"(), y), x; context = :color => false)
flat(x) = sprint(show, x; context = :color => false)          # the compact 2-arg form

# a @neuron-defined model must be declared at top level (structs can't be defined inside a testset)
@neuron _ShowLIF begin
    @parameters τ = 20.0 EL = -60.0 Vθ = -50.0 Vr = -60.0 R = 1.0 tref = 5.0
    @state V refrac
    @asymptote EL + R * I
    @resistance R
    @timeconstant τ
    @threshold V ≥ Vθ
    @reset Vr
    @refractory tref
end

@testset "show (models, synapses, composites)" begin
    @testset "leaf neuron model: aligned params + units + state footer" begin
        m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
        s = rich(m)
        @test startswith(s, "LIF{Float64}")
        @test occursin("τ", s) && occursin("20.0", s)
        @test occursin("ms", s)                       # τ / tref are times
        @test occursin("mV", s)                       # EL/Vθ/Vr are voltages
        @test occursin("GΩ", s)                       # R is a resistance
        @test occursin("state: V, refrac", s)
        @test count(==('\n'), s) ≥ 6                  # header + 6 params + footer (multi-line)
    end

    @testset "AdEx leaf renders all 11 params with units" begin
        m = AdEx(;
            C = 281.0, gL = 30.0, EL = -70.6, VT = -50.4, ΔT = 2.0, Vr = -70.6,
            Vpeak = -40.0, a = 4.0, b = 80.5, τw = 144.0, tref = 0.0
        )
        s = rich(m)
        @test startswith(s, "AdEx{Float64}")
        for f in ("C", "gL", "EL", "VT", "ΔT", "Vr", "Vpeak", "a", "b", "τw", "tref")
            @test occursin(f, s)
        end
        @test occursin("pF", s) && occursin("nS", s) && occursin("pA", s)   # capacitance/conductance/current
        @test occursin("state: V, refrac, w", s)
    end

    @testset "compact 2-arg form is a single named line" begin
        m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
        c = flat(m)
        @test !occursin('\n', c)
        @test startswith(c, "LIF{Float64}(")
        @test occursin("τ=20.0", c)
    end

    @testset "@neuron model: generic render, no invented units" begin
        m = _ShowLIF()
        s = rich(m)
        @test startswith(s, "_ShowLIF{Float64}")
        @test occursin("state: V, refrac", s)
        @test !occursin("mV", s) && !occursin("ms", s)     # no dimension metadata → bare numbers
    end

    @testset "synapse models: params + COBA/CUBA/delta tag" begin
        cs = ConductanceSynapse(; τ = 5.0, Erev = 0.0)
        s = rich(cs)
        @test startswith(s, "ConductanceSynapse{Float64}")
        @test occursin("COBA", s)
        @test occursin("Erev", s) && occursin("mV", s)
        @test occursin("τ", s) && occursin("ms", s)
        @test occursin("COBA", rich(DualExpSynapse(; τr = 0.5, τd = 2.0, Erev = 0.0)))
        @test occursin("CUBA", rich(CurrentSynapse(; τ = 5.0)))
        @test occursin("delta", rich(DeltaSynapse()))
    end

    @testset "MultiModel: one tree row per group with its range + model" begin
        mm = MultiModel(
            [
                AdEx(;
                    C = 281.0, gL = 30.0, EL = -70.6, VT = -50.4, ΔT = 2.0, Vr = -70.6,
                    Vpeak = -40.0, a = 4.0, b = 80.5, τw = 144.0, tref = 0.0
                ),
                LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 2.0),
            ], [8, 2]
        )
        s = rich(mm)
        @test occursin("MultiModel", s)
        @test occursin("2 groups", s) && occursin("N=10", s)
        @test occursin("1:8", s) && occursin("9:10", s)
        @test occursin("AdEx{Float64}", s) && occursin("LIF{Float64}", s)
        @test occursin("├─", s) && occursin("└─", s)        # tree connectors
    end

    @testset "Heterogeneous: base + per-neuron override SUMMARIES (no array dump)" begin
        base = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
        h = Heterogeneous(base; Vθ = collect(range(-52.0, -48.0; length = 50)), R = fill(100.0, 50))
        s = rich(h)
        @test occursin("Heterogeneous", s)
        @test occursin("per-neuron", s)
        @test occursin("base", s) && occursin("LIF{Float64}", s)
        @test occursin("Vθ", s) && occursin("R", s)
        @test occursin("50", s)                              # array length in the summary
        @test occursin("⟨", s) && occursin("…", s)           # min…max summary markers
        @test count(==('\n'), s) < 12                        # bounded: arrays summarised, never dumped
    end

    @testset "connectivity + projection summaries" begin
        conn = fixed_prob(
            Dewdrop.CPU(), 10, 10, 1.0; weight = 0.5, delay = steps(5), seed = UInt64(1),
            sources = 1:6, targets = 7:10
        )
        cs = rich(conn)
        @test occursin("SparseCSR", cs)
        @test occursin("edges", cs)
        p = Projection(DeltaSynapse(), conn)
        ps = rich(p)
        @test occursin("Projection", ps)
        @test occursin("DeltaSynapse", ps)
        @test occursin("edges", ps)
    end
end

_lif5() = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)

@testset "show (network, builder, solution)" begin
    @testset "DewdropNetwork: populations + projections tree, recovered labels" begin
        conn = fixed_prob(
            Dewdrop.CPU(), 5, 5, 1.0; weight = 0.5, delay = steps(1), seed = UInt64(1),
            sources = 1:3, targets = 4:5
        )
        net = DewdropNetwork(
            _lif5(), 5; input = 0.0, tspan = (0.0, 10.0),
            projections = (Projection(DeltaSynapse(), conn),), subpops = (E = 1:3, I = 4:5)
        )
        s = rich(net)
        @test startswith(s, "DewdropNetwork")
        @test occursin("N=5", s)
        @test occursin("populations", s)
        @test occursin("E", s) && occursin("I", s)
        @test occursin("1:3", s) && occursin("4:5", s)
        @test occursin("projections", s)
        @test occursin("E → I", s)            # label recovered by matching conn ranges to subpops
        @test occursin("DeltaSynapse", s)
        @test occursin("├─", s)
        c = flat(net)
        @test !occursin('\n', c) && occursin("DewdropNetwork", c)
    end

    @testset "builder: exact named projection labels + drive; NetworkBuilder render" begin
        nb = network(; tspan = (0.0, 10.0))
        population!(nb, :E, _lif5(), 3)
        population!(nb, :I, _lif5(), 2)
        project!(nb, :E => :I, DeltaSynapse(); p = 1.0, weight = 0.5, delay = steps(1), seed = UInt64(1))
        drive!(nb, PoissonDrive(; rate = 0.02, weight = 0.5, seed = UInt64(2)))

        sb = rich(nb)                          # the unbuilt builder
        @test startswith(sb, "NetworkBuilder")
        @test occursin("unbuilt", sb)
        @test occursin("populations", sb) && occursin("projections", sb)
        @test occursin("E → I", sb)
        @test occursin("drive", sb) && occursin("PoissonDrive", sb)

        net = build(nb)
        @test net.projlabels == ((:E => :I),)   # endpoints preserved onto the network
        sn = rich(net)
        @test occursin("E → I", sn)            # exact, not recovered
        @test occursin("PoissonDrive", sn)
    end

    @testset "DewdropSolution / SubSolution / BatchedSolution" begin
        net = DewdropNetwork(_lif5(), 10; input = 0.3, tspan = (0.0, 50.0), subpops = (E = 1:6, I = 7:10))
        sol = solve(net, FixedStep(0.1); record = (spikes = Spikes(),), progress = false)
        s = rich(sol)
        @test startswith(s, "DewdropSolution")
        @test occursin("N=10", s)
        @test occursin("ms", s)                # duration / dt in ms
        @test occursin("populations", s)
        @test occursin("Hz", s)                # firing-rate summary always shown (decision b)
        @test occursin("spikes", s)            # recorded monitor listed
        @test sum(sol.spike_count) > 0         # sanity: the net actually fired

        sub = rich(sol[:E])
        @test occursin("SubSolution", sub) && occursin("E", sub)

        bsol = solve(net, FixedStep(0.1); batch = 4, progress = false)
        sbatch = rich(bsol)
        @test occursin("BatchedSolution", sbatch)
        @test occursin("4", sbatch)            # batch size
    end
end

_lifS() = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)

@testset "show (batching + delay types)" begin
    @testset "steps(n) round-trips" begin
        @test flat(steps(5)) == "steps(5)"
    end

    @testset "BatchedModel: base + per-member override summaries (no array dump)" begin
        bm = Dewdrop.BatchedModel(_lifS(), (τ = [10.0, 20.0, 30.0],))
        s = rich(bm)
        @test occursin("BatchedModel", s)
        @test occursin("per-member", s) && occursin("B=3", s)
        @test occursin("base", s) && occursin("LIF{Float64}", s)
        @test occursin("τ", s) && occursin("⟨", s)        # min…max summary, not a dump
        @test count(==('\n'), s) < 8
    end

    @testset "NetworkBatch + BatchSolution" begin
        nb = network(; tspan = (0.0, 30.0)); population!(nb, :E, _lifS(), 5; input = 0.3)
        b = batch(build(nb); τ = [10.0, 20.0])
        @test occursin("NetworkBatch", rich(b)) && occursin("2 member", rich(b))

        bs = solve(b, FixedStep(0.1); progress = false)
        ss = rich(bs)
        @test occursin("BatchSolution", ss)
        @test occursin("2 member", ss) && occursin("mode", ss) && occursin("Hz", ss)
        @test !occursin('\n', flat(bs)) && occursin("BatchSolution", flat(bs))
    end
end
