using Dewdrop
using Test
using Statistics
using Adapt
using JLArrays
using GPUArrays

# The monitor (recording) framework. `record = (name = spec, ...)` materialises a
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
        @test all(v -> m.EL - 1.0e-6 ≤ v < m.Vθ, sol.record.V.data)             # post-reset, sub-θ
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
        agg = solve(
            prob, FixedStep(dt); record = (
                rate = Aggregate(Spikes(), sum), meanV = Aggregate(Trace(:V), :mean),
            )
        )
        @test size(agg.record.rate.data) == (1, nsteps)
        @test vec(agg.record.rate.data) == vec(sum(sol.record.spikes.data; dims = 1))
        @test vec(agg.record.meanV.data) ≈ vec(mean(sol.record.V.data; dims = 1))
    end

    @testset "stride" begin
        st = solve(prob, FixedStep(dt); record = (V = Trace(:V; every = 5),))
        @test size(st.record.V.data, 2) == cld(nsteps, 5)
        @test st.record.V.data == sol.record.V.data[:, 1:5:nsteps]
    end

    @testset "windowed flush across multiple windows" begin
        long = DewdropNetwork(m, 2; input = 0.5, tspan = (0.0, 130.0))   # 1300 steps > 1024 window
        ls = solve(long, FixedStep(dt); record = (V = Trace(:V),))
        @test size(ls.record.V.data, 2) == 1300
        @test ls.record.V.data[:, end] == ls.state.state.V             # final partial window flushed
        @test all(v -> m.EL - 1.0e-6 ≤ v < m.Vθ, ls.record.V.data)       # every window flushed correctly
    end

    @testset "other state vars, synaptic vars, and a probe" begin
        @test all(≥(-1.0e-9), solve(prob, FixedStep(dt); record = (r = Trace(:refrac),)).record.r.data)
        # a CUBA projection exposes a synaptic current column
        conn = fixed_prob(Dewdrop.CPU(), 4, 4, 0.5; weight = 1.0, delay = steps(2), seed = UInt64(1))
        cprob = DewdropNetwork(
            m, 4; input = 0.5, tspan = (0.0, tend),
            projection = Projection(CurrentSynapse(τ = 5.0), conn)
        )
        cs = solve(cprob, FixedStep(dt); record = (I = Trace(:Isyn; projection = 1),))
        @test size(cs.record.I.data) == (4, nsteps)
        # a probe records an arbitrary derived quantity (here the population mean V)
        pr = solve(prob, FixedStep(dt); record = (mV = Probe(integ -> [mean(integ.state.state.V)]; n = 1),))
        @test vec(pr.record.mV.data) ≈ vec(mean(sol.record.V.data; dims = 1))
    end

    # Time-axis reduction mirrors the unit axis: `every` SELECTS steps (a stride), `bin` REDUCES
    # them (events sum, signals average). Striding an event record is refused outright, since it
    # keeps the spike bit at one step in k and silently drops the rest.
    @testset "time-axis reduction: bin vs every" begin
        ref = solve(prob, FixedStep(dt); record = (s = Spikes(), v = Trace(:V), p = Aggregate(Spikes(), sum)))
        tot = sum(ref.record.s.data)
        @test tot > 10                                    # the fixture fires: nothing below is vacuous

        b1 = solve(prob, FixedStep(dt); record = (s = Spikes(bin = 1), v = Trace(:V; bin = 1)))
        @test b1.record.s.data == ref.record.s.data       # bin = 1 changes nothing
        @test b1.record.v.data == ref.record.v.data
        @test eltype(b1.record.s.data) === Bool

        for k in (2, 5, 20)
            r = solve(
                prob, FixedStep(dt);
                record = (s = Spikes(bin = k), p = Aggregate(Spikes(), sum; bin = k))
            )
            @test eltype(r.record.s.data) === UInt16      # Bool cannot hold a count
            @test size(r.record.s.data, 2) == fld(nsteps, k)      # complete windows only
            @test sum(r.record.s.data) == tot             # every spike survives binning
            @test sum(r.record.p.data) == tot
            @test r.record.s.data == UInt16.(Dewdrop.coarsegrain(ref.record.s.data, k))   # TimeseriesBase exports one too
            @test r.record.s.every == k && r.record.s.bin == k    # `every` = steps per column
        end

        bv = solve(
            prob, FixedStep(dt); record = (
                v = Trace(:V; bin = 10),
                a = Aggregate(Trace(:V), :mean; bin = 10),
            )
        )
        want = [
            mean(ref.record.v.data[i, ((c - 1) * 10 + 1):(c * 10)])
                for i in 1:size(ref.record.v.data, 1), c in 1:fld(nsteps, 10)
        ]
        @test bv.record.v.data ≈ want                     # a binned signal is the window mean
        @test vec(bv.record.a.data) ≈ vec(mean(want; dims = 1))

        sv = solve(prob, FixedStep(dt); record = (v = Trace(:V; every = 10),))
        @test sv.record.v.data == ref.record.v.data[:, 1:10:end]   # `every` still strides a signal

        @test_throws ArgumentError Spikes(every = 5)
        @test_throws ArgumentError Aggregate(Spikes(), sum; every = 5)
        @test_throws ArgumentError Spikes(every = 5, bin = 5)      # exclusive
        @test_throws ArgumentError Trace(:V; every = 5, bin = 5)
        @test_throws ArgumentError Aggregate(Trace(:V; bin = 2), :mean)   # inner knobs are ignored
        @test Aggregate(Trace(:V), :mean; every = 5) isa Aggregate        # signals may stride

        # Spike TIMES do not survive binning, so the statistics built on them refuse the record.
        rb = solve(prob, FixedStep(dt); record = (s = Spikes(bin = 10),))
        @test_throws ErrorException raster(rb)
        @test_throws ErrorException cv_isi(rb)
        @test !isempty(first(raster(ref; name = :s)))

        B = 3                                             # the batched path mirrors the scalar one
        ba = solve(
            prob, FixedStep(dt); batch = B, streams = fill(0, B),
            record = (s = Spikes(), p = Aggregate(Spikes(), sum))
        )
        bb = solve(
            prob, FixedStep(dt); batch = B, streams = fill(0, B),
            record = (s = Spikes(bin = 5), p = Aggregate(Spikes(), sum; bin = 5))
        )
        @test eltype(bb.record.s.data) === UInt16
        @test size(bb.record.s.data, 3) == fld(nsteps, 5)
        for col in 1:B
            @test sum(ba.record.s.data[:, col, :]) == sum(bb.record.s.data[:, col, :]) > 0
            @test sum(ba.record.p.data[col, :]) == sum(bb.record.p.data[col, :])
        end
    end

    @testset "GPU-safe under allowscalar(false): traces, spikes, in-kernel aggregate" begin
        GPUArrays.allowscalar(false)
        gpu = adapt(
            JLArray, init(
                prob, FixedStep(dt);
                record = (V = Trace(:V), spikes = Spikes(), rate = Aggregate(Spikes(), sum))
            )
        )
        @test gpu.monitors.V.buf.window isa JLArray
        for _ in 1:50
            step!(gpu)
        end
        Dewdrop._finalize_all!(gpu.monitors)
        @test gpu.monitors.rate.buf.flushed == 50         # one aggregate column per step reached the host store
        @test all(<(0), view(gpu.monitors.V.buf.store, :, 1:50))   # the 50 recorded columns hold real voltages
    end
end

# The reducer becomes a type parameter and the two paths disagree on an unsupported one: the CPU
# `_finalize` has no method (a MethodError mid-solve) while the GPU kernel tests only `R === :mean`,
# so anything else silently computes a SUM. Refuse it where it is written instead.
@testset "Aggregate validates its reducer" begin
    @test Aggregate(Trace(:V), :sum).reducer === :sum
    @test Aggregate(Trace(:V), sum).reducer === :sum
    @test Aggregate(Trace(:V), :mean).reducer === :mean
    @test_throws ArgumentError Aggregate(Trace(:V), :median)
    @test_throws ArgumentError Aggregate(Trace(:V), maximum)
end

# The device aggregate is a real parallel reduction into a scratch slot, then a one-element fold into
# the open column. Sizes below and above one workgroup exercise both regimes, binned and not.
@testset "device aggregate reduction matches the CPU reduction" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    rec() = (
        rate = Aggregate(Spikes(), sum), mv = Aggregate(Trace(:V), :mean),
        rb = Aggregate(Spikes(), sum; bin = 5), mb = Aggregate(Trace(:V), :mean; bin = 5),
    )
    for N in (7, 300, 5000)
        prob = DewdropNetwork(m, N; input = 0.35, tspan = (0.0, 60.0))
        ref = solve(prob, FixedStep(0.1); record = rec())
        got = solve!(adapt(JLArray, init(prob, FixedStep(0.1); record = rec())))
        @test Array(got.record.rate.data) == Array(ref.record.rate.data)   # integer sums: exact
        @test Array(got.record.rb.data) == Array(ref.record.rb.data)
        @test Array(got.record.mv.data) ≈ Array(ref.record.mv.data) rtol = 1.0e-4
        @test Array(got.record.mb.data) ≈ Array(ref.record.mb.data) rtol = 1.0e-4
    end
end

# The fused step keeps the accumulators in registers, so their array stores exist only for the
# monitors that read them back. Eliding those stores must not change what a recording sees.
@testset ":itot/:gtot are materialised exactly when recorded" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    prob = DewdropNetwork(m, 64; input = 0.5, tspan = (0.0, 20.0))
    for bk in (Serial(), Fused(), Dewdrop.Auto())
        sol = solve(prob, FixedStep(0.1); backend = bk, record = (it = Trace(:itot), gt = Trace(:gtot)))
        @test all(≈(0.5), sol.record.it.data)     # the constant input, materialised
        @test all(iszero, sol.record.gt.data)     # no conductance synapses
    end
    @test Dewdrop._accum_sinks(init(prob, FixedStep(0.1); backend = Fused())) === (nothing, nothing)
    rec = init(prob, FixedStep(0.1); backend = Fused(), record = (it = Trace(:itot),))
    @test Dewdrop._accum_sinks(rec) === (rec.itot, rec.gtot)
    @test !Dewdrop._any_reads_accum(typeof(NamedTuple()))
    # a Probe's `f` is opaque, so it counts as reading them: `Probe(integ -> integ.itot)` recorded
    # zeros under Fused when the stores were elided on the monitor type alone.
    pr = (p = Probe(integ -> copy(integ.itot); n = 64),)
    @test Dewdrop._any_reads_accum(typeof(init(prob, FixedStep(0.1); record = pr).monitors))
    for bk in (Serial(), Fused(), Dewdrop.Auto())
        sol = solve(prob, FixedStep(0.1); backend = bk, record = pr)
        @test all(≈(0.5), sol.record.p.data)
    end
end

# `Aggregate` reduces over the units of a per-unit source. `Probe` carries `every`/`bin` too, so the
# stride check alone admitted it and it died later on `inner.of`, a field `Probe` does not have.
@testset "Aggregate rejects a non-per-unit inner spec" begin
    @test_throws ArgumentError Aggregate(Probe(integ -> integ.itot; n = 1), :mean)
    @test Aggregate(Trace(:V), :mean) isa Aggregate          # the supported inners still construct
    @test Aggregate(Spikes(), :sum) isa Aggregate
end
