using Dewdrop
using Test
using Unitful

# The Unitful boundary extension. Physical-unit inputs are converted + stripped to Dewdrop's
# coherent canonical float system (ms, mV, nS, pA, pF, GΩ, kHz) at construction, so the engine
# state stays plain isbits floats (the GPU contract) while the API accepts real units.
@testset "Unitful boundary extension" begin
    @testset "neuron + synapse construction strips to canonical floats" begin
        mu = LIF(; τ = 20u"ms", EL = -60u"mV", Vθ = -50u"mV", Vr = -60u"mV", R = 0.1u"GΩ", tref = 5u"ms")
        mb = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 5.0)
        @test mu isa LIF{Float64}
        @test (mu.τ, mu.EL, mu.Vθ, mu.Vr, mu.R, mu.tref) == (mb.τ, mb.EL, mb.Vθ, mb.Vr, mb.R, mb.tref)
        # unit CONVERSION, not just stripping: seconds → ms, MΩ → GΩ
        m2 = LIF(; τ = 0.02u"s", EL = -60u"mV", Vθ = -50u"mV", Vr = -60u"mV", R = 100u"MΩ", tref = 5u"ms")
        @test m2.τ ≈ 20.0 && m2.R ≈ 0.1
        # synapses
        su = ConductanceSynapse(; τ = 5u"ms", Erev = 0u"mV")
        @test su isa ConductanceSynapse{Float64} && su.τ == 5.0 && su.Erev == 0.0
        @test CurrentSynapse(; τ = 5u"ms").τ == 5.0
    end

    @testset "wrong dimension is rejected at the boundary" begin
        @test_throws Exception LIF(; τ = 20u"mV", EL = -60u"mV", Vθ = -50u"mV", Vr = -60u"mV", R = 0.1u"GΩ", tref = 5u"ms")
        @test_throws Exception ConductanceSynapse(; τ = 5u"ms", Erev = 0u"ms")
        @test_throws Exception PoissonDrive(; rate = 20u"mV", weight = 0.1u"mV", seed = 1)
    end

    @testset "drive / step / network / weights" begin
        # the per-ms-vs-Hz units trap is now impossible: the rate carries its unit and converts
        d1 = PoissonDrive(; rate = 20u"kHz", weight = 0.1u"mV", seed = 1)
        d2 = PoissonDrive(; rate = 20000u"Hz", weight = 0.1u"mV", seed = 1)
        @test d1.rate ≈ 20.0 && d2.rate ≈ 20.0 && d1.weight == 0.1
        @test FixedStep(0.1u"ms").dt == 0.1
        m = LIF(; τ = 20u"ms", EL = 0u"mV", Vθ = 20u"mV", Vr = 10u"mV", R = 1u"GΩ", tref = 2u"ms")
        prob = DewdropNetwork(m, 10; input = 0.5u"pA", tspan = (0u"ms", 50u"ms"))
        @test prob.tspan == (0.0, 50.0) && prob.input == 0.5
        # connectivity weights stripped by their OWN dimension (delta→mV, COBA→nS, CUBA→pA)
        @test all(≈(0.1), fixed_prob(Dewdrop.CPU(), 10, 10, 0.5; weight = 0.1u"mV", delay = steps(1), seed = UInt64(1)).weight)
        @test all(≈(6.0), fixed_prob(Dewdrop.CPU(), 10, 10, 0.5; weight = 6u"nS", delay = steps(1), seed = UInt64(1)).weight)
    end

    @testset "end-to-end: unitful network == bare-canonical network" begin
        function buildnet(unitful)
            m = unitful ? LIF(; τ = 20u"ms", EL = 0u"mV", Vθ = 20u"mV", Vr = 10u"mV", R = 1u"GΩ", tref = 2u"ms") :
                LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
            nb = network(m, 80, 20; arch = Dewdrop.CPU(), tspan = unitful ? (0u"ms", 200u"ms") : (0.0, 200.0))
            project!(nb, :E, DeltaSynapse(); p = 0.1, weight = unitful ? 0.5u"mV" : 0.5, delay = steps(1), seed = UInt64(1))
            drive!(nb, PoissonDrive(; rate = unitful ? 0.3u"kHz" : 0.3, weight = unitful ? 0.5u"mV" : 0.5, seed = UInt64(2)))
            return build(nb)
        end
        su = solve(buildnet(true), FixedStep(0.1u"ms"); record = (spikes = Spikes(),))
        sb = solve(buildnet(false), FixedStep(0.1); record = (spikes = Spikes(),))
        @test su.spike_count == sb.spike_count
    end

    @testset "dimensional-soundness GUARANTEE (engine code run on units)" begin
        # _coba_step IS the voltage-coupled core dynamics. Running the ACTUAL engine function on
        # canonical-unit Quantities certifies (a) it is dimensionally consistent (no DimensionError)
        # and yields a VOLTAGE, and (b) the canonical system is coherent: the stripped-float result,
        # re-attached as mV, equals the units-typed result. A dimensional bug (e.g. a dropped R)
        # would throw here; a scale/coherence error would fail the ≈.
        Vu = Dewdrop._coba_step(-55u"mV", -60u"mV", 0.1u"GΩ", 20u"ms", 6u"nS", 100u"pA", 0.1u"ms")
        @test dimension(Vu) == dimension(u"mV")
        Vs = Dewdrop._coba_step(-55.0, -60.0, 0.1, 20.0, 6.0, 100.0, 0.1)
        @test ustrip(u"mV", Vu) ≈ Vs
        # the pure LIF asymptote EL + R·I is coherent in the same system
        @test ustrip(u"mV", -60u"mV" + 0.1u"GΩ" * 100u"pA") ≈ -60.0 + 0.1 * 100.0
    end
end
