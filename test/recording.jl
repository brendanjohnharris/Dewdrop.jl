using Dewdrop
using Test
using Statistics
using Adapt
using JLArrays
using GPUArrays

# M4 --- the monitor (recording) framework. `record = (name = spec, ...)` materialises a
# NamedTuple of monitors that stage into arch-resident window buffers flushed to host stores;
# `sol.record.<name>` holds the result. Specs: Trace (any state/synaptic/accumulator var),
# Spikes (raster), Aggregate (a scalar reduction per step), Probe (an arbitrary function), each
# optionally over a unit subset and/or strided. The `:record` slot runs after `:reset`, so
# traces are the post-reset state.
@testset "monitor (recording) framework" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    dt, tend = 0.1, 100.0
    prob = DewdropNetwork(m, 4; input = 0.5, tspan = (0.0, tend))      # supra-rheobase
    nsteps = round(Int, tend / dt)

    sol = solve(prob, FixedStep(dt); record = (spikes = Spikes(), V = Trace(:V)))

    @testset "traces + spikes: shapes, parity, post-reset snapshot" begin
        @test size(sol.record.V.data) == (4, nsteps)
        @test size(sol.record.spikes.data) == (4, nsteps)
        @test eltype(sol.record.spikes.data) == Bool
        @test vec(sum(sol.record.spikes.data; dims = 2)) == sol.spike_count   # raster ⇔ counts
        @test all(v -> m.EL - 1e-6 ≤ v < m.Vθ, sol.record.V.data)             # post-reset, sub-θ
        @test sol.record.V.data[:, end] == sol.state.state.V                  # last column = final
    end

    @testset "default: nothing recorded except the spike count" begin
        sol0 = solve(prob, FixedStep(dt))
        @test sol0.record == NamedTuple()
        @test sol0.spike_count == sol.spike_count        # the count is always-on, recording-independent
    end

    @testset "raster over the spikes monitor" begin
        times, ids = raster(sol)
        @test length(times) == sum(sol.record.spikes.data)
        @test all(i -> 1 ≤ i ≤ 4, ids) && all(t -> 0 < t ≤ tend, times)
        @test_throws Exception raster(solve(prob, FixedStep(dt)))             # no spikes recorded
    end

    @testset "unit subset" begin
        sub = solve(prob, FixedStep(dt); record = (V = Trace(:V; of = [1, 3]),))
        @test size(sub.record.V.data) == (2, nsteps)
        @test sub.record.V.data == sol.record.V.data[[1, 3], :]
        @test sub.record.V.idx == [1, 3]
    end

    @testset "aggregates (scalar reduction per step)" begin
        agg = solve(prob, FixedStep(dt); record = (
            rate = Aggregate(Spikes(), sum), meanV = Aggregate(Trace(:V), :mean)))
        @test size(agg.record.rate.data) == (1, nsteps)
        @test vec(agg.record.rate.data) == vec(sum(sol.record.spikes.data; dims = 1))
        @test vec(agg.record.meanV.data) ≈ vec(mean(sol.record.V.data; dims = 1))
    end

    @testset "stride" begin
        st = solve(prob, FixedStep(dt); record = (V = Trace(:V; every = 5),))
        @test size(st.record.V.data, 2) == cld(nsteps, 5)
        @test st.record.V.data == sol.record.V.data[:, 1:5:nsteps]
    end

    @testset "other state vars, synaptic vars, and a probe" begin
        @test all(≥(-1e-9), solve(prob, FixedStep(dt); record = (r = Trace(:refrac),)).record.r.data)
        # a CUBA projection exposes a synaptic current column
        conn = fixed_prob(Dewdrop.CPU(), 4, 4, 0.5; weight = 1.0, delay = 2, seed = UInt64(1))
        cprob = DewdropNetwork(m, 4; input = 0.5, tspan = (0.0, tend),
            projection = Projection(CurrentSynapse(τ = 5.0), conn))
        cs = solve(cprob, FixedStep(dt); record = (I = Trace(:Isyn; projection = 1),))
        @test size(cs.record.I.data) == (4, nsteps)
        # a probe records an arbitrary derived quantity (here the population mean V)
        pr = solve(prob, FixedStep(dt); record = (mV = Probe(integ -> [mean(integ.state.state.V)]; n = 1),))
        @test vec(pr.record.mV.data) ≈ vec(mean(sol.record.V.data; dims = 1))
    end

    @testset "GPU-safe under allowscalar(false): traces, spikes, in-kernel aggregate" begin
        GPUArrays.allowscalar(false)
        gpu = adapt(JLArray, init(prob, FixedStep(dt);
            record = (V = Trace(:V), spikes = Spikes(), rate = Aggregate(Spikes(), sum))))
        @test gpu.monitors.V.buf.window isa JLArray
        for _ in 1:50
            step!(gpu)
        end
        Dewdrop._finalize_all!(gpu.monitors)
        @test sum(gpu.monitors.rate.buf.store) ≥ 0       # ran without scalar-indexing errors
    end
end
