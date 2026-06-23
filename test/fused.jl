using Dewdrop
using Test
using JLArrays
using GPUArrays
using Adapt: adapt

# M6 Tier-1 --- the fused device megakernel (src/Fused.jl). On a GPU backend the canonical
# schedule's dense per-neuron phases (deliver + drive + accumulate + membrane + decay +
# threshold + reset + count) collapse into ONE KernelAbstractions launch, the CPU path keeps
# its broadcast phases. This suite pins the fused path to be NUMERICALLY IDENTICAL to the
# broadcast path WITHOUT a GPU: JLArrays' `JLBackend <: KernelAbstractions.GPU`, so adapting an
# integrator to `JLArray` routes `step!` through the fused kernel, and `allowscalar(false)`
# proves it never falls back to host scalar indexing. (The real-CUDA equivalence + throughput
# win are in test/cuda.jl, guarded by a functional GPU.)

# Run a problem on the CPU (broadcast) and under JLArrays (fused) and return both solutions.
function _cpu_vs_fused(prob, alg; kw...)
    cpu = solve(prob, alg; kw...)
    ig = adapt(JLArray, Dewdrop.init(prob, alg; kw...))   # JLArray ⇒ JLBackend ⇒ fused step!
    # a non-CPU backend ⇒ `step!` routes through `_fused_step!` (not the broadcast phases)
    @test typeof(Dewdrop.get_backend(ig.state.state.V)) !== typeof(Dewdrop.get_backend(zeros(1)))
    Dewdrop.solve!(ig)
    return cpu, Dewdrop.DewdropSolution(ig)
end

@testset "fused device megakernel ≡ broadcast (JLArrays, no GPU)" begin
    GPUArrays.allowscalar(false)                          # the fused path must never scalar-index
    rec() = (spikes = Spikes(), V = Trace(:V), rate = Aggregate(Spikes(), sum))

    @testset "delta multi-projection (E/I) + Poisson drive + monitors" begin
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
        ce = fixed_prob(Dewdrop.CPU(), 120, 120, 0.1; weight = 0.5, delay = steps(5), seed = UInt64(1), sources = 1:90, allow_self = false)
        ci = fixed_prob(Dewdrop.CPU(), 120, 120, 0.1; weight = -1.0, delay = steps(5), seed = UInt64(2), sources = 91:120, allow_self = false)
        prob = DewdropNetwork(m, 120; input = 0.0, tspan = (0.0, 80.0), arch = Dewdrop.CPU(),
            projections = (Projection(DeltaSynapse(), ce), Projection(DeltaSynapse(), ci)),
            drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = UInt64(3)))
        c, g = _cpu_vs_fused(prob, FixedStep(0.1); record = rec())
        @test Array(g.spike_count) == c.spike_count                  # bit-identical (delta + counter RNG)
        @test g.record.spikes.data == c.record.spikes.data
        @test g.record.rate.data == c.record.rate.data
        @test g.record.V.data ≈ c.record.V.data
    end

    @testset "COBA conductance synapses + randomized init" begin
        mc = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 5.0)
        ce = fixed_prob(Dewdrop.CPU(), 150, 150, 0.08; weight = 0.6, delay = steps(1), seed = UInt64(1), sources = 1:120)
        ci = fixed_prob(Dewdrop.CPU(), 150, 150, 0.08; weight = 6.7, delay = steps(1), seed = UInt64(2), sources = 121:150)
        prob = DewdropNetwork(mc, 150; input = 0.0, tspan = (0.0, 120.0), arch = Dewdrop.CPU(),
            projections = (Projection(ConductanceSynapse(τ = 5.0, Erev = 0.0), ce),
                Projection(ConductanceSynapse(τ = 10.0, Erev = -80.0), ci)),
            drive = PoissonDrive(rate = 6.0, weight = 0.1, seed = UInt64(7)))
        c, g = _cpu_vs_fused(prob, FixedStep(0.1); v0 = (-60.0, -50.0), record = (spikes = Spikes(),))
        # JLArrays runs the kernel on CPU libm, so the exp-based COBA step matches bit-for-bit.
        @test Array(g.spike_count) == c.spike_count
    end

    @testset "CUBA current synapse" begin
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 1.0)
        ce = fixed_prob(Dewdrop.CPU(), 80, 80, 0.1; weight = 1.2, delay = steps(3), seed = UInt64(4), allow_self = false)
        prob = DewdropNetwork(m, 80; input = 0.0, tspan = (0.0, 60.0), arch = Dewdrop.CPU(),
            projection = Projection(CurrentSynapse(τ = 5.0), ce),
            drive = PoissonDrive(rate = 30.0, weight = 0.8, seed = UInt64(5)))
        c, g = _cpu_vs_fused(prob, FixedStep(0.1); record = (spikes = Spikes(),))
        @test Array(g.spike_count) == c.spike_count
        @test g.record.spikes.data == c.record.spikes.data
    end

    @testset "unconnected driven population (empty projection tuple fuses too)" begin
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 15.0, Vr = 0.0, R = 1.0, tref = 2.0)
        prob = DewdropNetwork(m, 64; input = 0.0, tspan = (0.0, 50.0), arch = Dewdrop.CPU(),
            drive = PoissonDrive(rate = 40.0, weight = 0.6, seed = UInt64(9)))
        c, g = _cpu_vs_fused(prob, FixedStep(0.1); record = (V = Trace(:V), spikes = Spikes()))
        @test Array(g.spike_count) == c.spike_count
        @test g.record.V.data ≈ c.record.V.data
    end
end

# `step = :fused` runs the fused megakernel on the NATIVE CPU (KA.CPU threads the per-neuron ndrange)
# instead of the broadcast phases --- the large-N optimisation the advisor suggests. The Array scatter
# is unchanged and the dense kernel uses the same CPU libm, so it is BIT-IDENTICAL to the default
# broadcast path even multi-threaded (exact weights ⇒ order-independent atomic scatter).
@testset "native CPU step = :fused ≡ broadcast" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 2.0)
    ce = fixed_prob(Dewdrop.CPU(), 160, 160, 0.1; weight = 0.5, delay = steps(5), seed = UInt64(1), sources = 1:120, allow_self = false)
    ci = fixed_prob(Dewdrop.CPU(), 160, 160, 0.1; weight = -1.0, delay = steps(5), seed = UInt64(2), sources = 121:160, allow_self = false)
    prob = DewdropNetwork(m, 160; input = 0.0, tspan = (0.0, 100.0),
        projections = (Projection(DeltaSynapse(), ce), Projection(DeltaSynapse(), ci)),
        drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = UInt64(3)))
    bcast = solve(prob, FixedStep(0.1); advise = false, record = (spikes = Spikes(), V = Trace(:V)))
    fused = solve(prob, FixedStep(0.1); step = :fused, advise = false, record = (spikes = Spikes(), V = Trace(:V)))
    @test sum(bcast.spike_count) > 0
    @test fused.spike_count == bcast.spike_count            # bit-identical (exact weights + same libm)
    @test fused.state.state.V == bcast.state.state.V
    @test fused.record.spikes.data == bcast.record.spikes.data
    @test fused.record.V.data == bcast.record.V.data

    # a multi-state model (AdEx) also fuses on the native CPU
    ad = AdEx(; C = 200.0, gL = 10.0, EL = -70.0, VT = -50.0, ΔT = 2.0, Vr = -58.0, Vpeak = -40.0,
        a = 0.0, b = 40.0, τw = 100.0, tref = 0.0)
    padex = DewdropNetwork(ad, 200; input = 400.0, tspan = (0.0, 60.0))
    @test solve(padex, FixedStep(0.1); step = :fused, advise = false).spike_count ==
        solve(padex, FixedStep(0.1); advise = false).spike_count

    @test_throws ArgumentError Dewdrop.init(prob, FixedStep(0.1); step = :bogus)
end
