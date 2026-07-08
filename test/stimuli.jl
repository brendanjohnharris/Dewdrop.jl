using Dewdrop
using Test
using JLArrays
using GPUArrays
using Statistics
using Adapt: adapt

# Unified stimulus seam (src/Stimuli.jl): every external input is an `AbstractStimulus` applied at a compile-time
# point (:current / :conductance / :kick / :noise). This file covers the families BEYOND the three legacy inputs
# (input / drive / noise, which the whole suite already exercises byte-identically): FunctionalCurrent/Kick (live
# f(t) / f(i,t)), TimedArray (tabulated, indexed by step), the analytic shapes, InhomogeneousPoisson, and
# SpikeSourceArray. The oracle is BACKEND AGREEMENT — Serial == Fused == batched-column-0 (stream 0) — plus
# JLArrays under `allowscalar(false)` to prove the device megakernel path never scalar-indexes.

_slif() = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 2.0)
const SARCH = Dewdrop.CPU()

@testset "unified stimulus seam" begin

    @testset "FunctionalCurrent: f(t) and f(i, t), Serial ≡ Fused" begin
        N = 30; dt = 0.1; tspan = (0.0, 100.0)
        # f(t): a suprathreshold sinusoid → spikes; both backends agree bit-for-bit
        p = DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan,
            stimuli = sinusoid(amplitude = 10.0, freq = 0.05, offset = 22.0))
        cs = solve(p, FixedStep(dt); backend = Serial()).spike_count
        cf = solve(p, FixedStep(dt); backend = Fused()).spike_count
        @test cs == cf
        @test sum(cf) > 0
        # f(i, t): a per-neuron gradient of constant inputs → monotone rate across neurons
        p2 = DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan,
            stimuli = FunctionalCurrent((i, t) -> 15.0 + 0.5 * i))
        c2s = solve(p2, FixedStep(dt); backend = Serial()).spike_count
        c2f = solve(p2, FixedStep(dt); backend = Fused()).spike_count
        @test c2s == c2f
        @test issorted(c2f)                              # higher i → higher input → ≥ rate
    end

    @testset "TimedArray ≡ matching FunctionalCurrent (drift-free step indexing)" begin
        N = 20; dt = 0.1; tspan = (0.0, 80.0); nsteps = 800
        data = [22.0 + 8.0 * sin(0.05 * k) for k in 0:(nsteps - 1)]   # value at step k (0-based)
        ta = DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan, stimuli = TimedArray(data))
        fc = DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan,
            stimuli = FunctionalCurrent(t -> 22.0 + 8.0 * sin(0.05 * round(Int, t / dt))))
        cta = solve(ta, FixedStep(dt); backend = Fused()).spike_count
        cfc = solve(fc, FixedStep(dt); backend = Fused()).spike_count
        @test cta == cfc                                 # TimedArray[k] == f(t = k·dt): the step index lines up
        @test solve(ta, FixedStep(dt); backend = Serial()).spike_count == cta
        # per-neuron matrix form (N × nsteps)
        mat = repeat(reshape(data, 1, :), N, 1)
        tam = DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan, stimuli = TimedArray(mat))
        @test solve(tam, FixedStep(dt); backend = Fused()).spike_count == cta
    end

    @testset "TimedArray shape validation" begin
        N = 5; dt = 0.1; tspan = (0.0, 10.0)             # nsteps = 100
        @test_throws ArgumentError solve(
            DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan, stimuli = TimedArray(zeros(50))), FixedStep(dt))
        @test_throws ArgumentError solve(
            DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan, stimuli = TimedArray(zeros(3, 100))), FixedStep(dt))
    end

    @testset "analytic input shapes evaluate correctly" begin
        r = ramp(t1 = 10.0, to = 5.0)                    # 0 at t=0, 5 at t=10, flat after
        @test r.f(1, 0.0) ≈ 0.0
        @test r.f(1, 5.0) ≈ 2.5
        @test r.f(1, 20.0) ≈ 5.0
        s = step_input(amplitude = 3.0, t0 = 4.0)
        @test s.f(1, 3.9) == 0.0 && s.f(1, 4.1) == 3.0
        sn = sinusoid(amplitude = 2.0, freq = 0.25, offset = 1.0)
        @test sn.f(1, 0.0) ≈ 1.0 && sn.f(1, 1.0) ≈ 3.0   # sin(2π·0.25·1) = sin(π/2) = 1
        pl = pulses(amplitude = 1.0, period = 10.0, width = 2.0)
        @test pl.f(1, 1.0) == 1.0 && pl.f(1, 5.0) == 0.0 && pl.f(1, 11.0) == 1.0
    end

    @testset "InhomogeneousPoisson: targeting, f(t) rate, batched col0 ≡ scalar" begin
        N = 6; dt = 0.1; tspan = (0.0, 200.0)
        rate = [400.0, 0.0, 0.0, 0.0, 0.0, 0.0]          # only neuron 1 driven (targeting via a zero rate)
        p = DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan,
            stimuli = InhomogeneousPoisson(rate; weight = 25.0, seed = UInt64(7)))
        cs = solve(p, FixedStep(dt); backend = Serial()).spike_count
        cf = solve(p, FixedStep(dt); backend = Fused()).spike_count
        @test cs == cf
        @test cf[1] > 0 && all(==(0), cf[2:end])
        @test solve(p, FixedStep(dt)).spike_count == cf                  # reproducible (counter RNG)
        b = solve(p, FixedStep(dt); batch = 3, streams = fill(0, 3))     # stream 0 → scalar reference
        @test b.spike_count[:, 1] == cf
        @test b.spike_count[:, 2] == cf                                  # identical streams → identical columns
        # time-varying rate function; Serial ≡ Fused (relies on the drift-free `t`)
        pf = DewdropNetwork(_slif(), N; input = 15.0, tspan = tspan,
            stimuli = InhomogeneousPoisson(t -> 50.0 + t; weight = 1.0, seed = UInt64(9)))
        @test solve(pf, FixedStep(dt); backend = Serial()).spike_count ==
            solve(pf, FixedStep(dt); backend = Fused()).spike_count
    end

    @testset "SpikeSourceArray: replay, deterministic, generic synapse, batched" begin
        N = 4; dt = 0.1; tspan = (0.0, 200.0); nsteps = 2000
        extconn = Dewdrop.SparseCSR(SARCH, [(1, 1, 25.0, 1)]; npre = 1, npost = N)   # 1 source → neuron 1
        spikes = fill(false, 1, nsteps); spikes[1, 10:10:end] .= true                # one spike every 1 ms
        mk(syn) = DewdropNetwork(_slif(), N; input = 0.0, tspan = tspan,
            projections = (Projection(SpikeSourceArray(syn, extconn, spikes), Dewdrop._empty_csr(SARCH, N)),))
        cs = solve(mk(DeltaSynapse()), FixedStep(dt); backend = Serial()).spike_count
        cf = solve(mk(DeltaSynapse()), FixedStep(dt); backend = Fused()).spike_count
        @test cs == cf
        @test cf[1] > 0 && all(==(0), cf[2:end])                        # targeting: only neuron 1
        @test solve(mk(DeltaSynapse()), FixedStep(dt)).spike_count == cf   # deterministic (no RNG)
        @test solve(mk(CurrentSynapse(τ = 5.0)), FixedStep(dt)).spike_count[1] > 0   # generic over synapse
        b = solve(mk(DeltaSynapse()), FixedStep(dt); batch = 2, streams = fill(0, 2))
        @test b.spike_count[:, 1] == cf && b.spike_count[:, 2] == cf     # replay shared across columns
    end

    @testset "combining stimuli (input + drive + functional + timed), all backends" begin
        N = 15; dt = 0.1; tspan = (0.0, 100.0); nsteps = 1000
        p = DewdropNetwork(_slif(), N; input = fill(10.0, N), tspan = tspan,
            drive = PoissonDrive(rate = 30.0, weight = 0.5, seed = UInt64(3)),
            stimuli = (ramp(t1 = 80.0, to = 8.0), TimedArray(fill(4.0, nsteps))))
        cs = solve(p, FixedStep(dt); backend = Serial()).spike_count
        cf = solve(p, FixedStep(dt); backend = Fused()).spike_count
        @test cs == cf && sum(cf) > 0
        b = solve(p, FixedStep(dt); batch = 2, streams = fill(0, 2))
        @test b.spike_count[:, 1] == cf
    end

    @testset "builder: stimulate! and freeze round-trip" begin
        N = 20; dt = 0.1; tspan = (0.0, 100.0)
        nb = network(; tspan = tspan)
        population!(nb, :E, _slif(), N; input = 0.0)
        stimulate!(nb, sinusoid(amplitude = 10.0, freq = 0.05, offset = 22.0))
        stimulate!(nb, TimedArray(fill(2.0, 1000)))
        prob = build(nb)
        @test length(prob.stimuli) == 2
        cs = solve(prob, FixedStep(dt); backend = Serial()).spike_count
        cf = solve(prob, FixedStep(dt); backend = Fused()).spike_count
        @test cs == cf && sum(cf) > 0
        fb = freeze(nb)                                          # the spec carries the stimuli
        probf = Dewdrop.materialize(fb, FixedStep(dt))
        @test length(probf.stimuli) == 2
        @test solve(probf, FixedStep(dt)).spike_count == cf
    end

    @testset "GPU-readiness: new stimuli under JLArray + allowscalar(false)" begin
        GPUArrays.allowscalar(false)
        N = 40; dt = 0.1; tspan = (0.0, 60.0); nsteps = 600
        data = [22.0 + 6.0 * sin(0.05 * k) for k in 0:(nsteps - 1)]
        rate = fill(200.0, N)
        extconn = Dewdrop.SparseCSR(SARCH, [(1, 3, 25.0, 1)]; npre = 1, npost = N)
        spikes = fill(false, 1, nsteps); spikes[1, 50:50:end] .= true
        p = DewdropNetwork(_slif(), N; input = fill(5.0, N), tspan = tspan,
            drive = PoissonDrive(rate = 20.0, weight = 0.4, seed = UInt64(1)),
            noise = WhiteNoise(0.5; seed = UInt64(2)),
            stimuli = (FunctionalCurrent(t -> 3.0), TimedArray(data),
                InhomogeneousPoisson(rate; weight = 0.3, seed = UInt64(5))),
            projections = (Projection(SpikeSourceArray(DeltaSynapse(), extconn, spikes), Dewdrop._empty_csr(SARCH, N)),))
        cpu = init(p, FixedStep(dt); backend = Fused())
        gpu = adapt(JLArray, init(p, FixedStep(dt); backend = Fused()))
        @test gpu.state.state.V isa JLArray
        for _ in 1:nsteps
            step!(cpu); step!(gpu)
        end
        @test Array(gpu.spike_count) == cpu.spike_count   # device megakernel path ≡ CPU fused, no scalar indexing
        GPUArrays.allowscalar(true)
    end

end # testset
