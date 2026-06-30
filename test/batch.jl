using Dewdrop
using Test
using JLArrays
using GPUArrays
using Statistics
using Adapt: adapt

# Ensemble (tensor) batching (src/Batch.jl): run B independent network instances at once, sharing
# connectivity STRUCTURE. The correctness check is a SCALAR REFERENCE: with the per-column drive
# stream forced to 0 (the scalar default), batched column b must equal a scalar solve configured
# with that column's input/v0 --- BIT-IDENTICALLY (delta synapses + counter RNG are exact). With
# distinct streams the columns are independent realizations. No GPU is needed: KA kernels run on
# the CPU backend, and JLArrays' allowscalar(false) proves the batched path never scalar-indexes.
# (Real-CUDA correctness + the ensemble speedup are in test/cuda.jl, behind a functional GPU.)

const ARCH = Dewdrop.CPU()
_lif() = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
function _ei_delta(input; N = 100, T = 80.0, arch = ARCH)
    ce = fixed_prob(arch, N, N, 0.1; weight = 0.5, delay = steps(5), seed = UInt64(1), sources = 1:(4N ÷ 5), allow_self = false)
    ci = fixed_prob(arch, N, N, 0.1; weight = -1.0, delay = steps(5), seed = UInt64(2), sources = (4N ÷ 5 + 1):N, allow_self = false)
    return DewdropNetwork(_lif(), N; input = input, tspan = (0.0, T), arch = arch,
        projections = (Projection(DeltaSynapse(), ce), Projection(DeltaSynapse(), ci)),
        drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = UInt64(3)))
end
_colmat(vals, N) = repeat(reshape(collect(Float64, vals), 1, :), N, 1)   # (N,B) per-column constant input
_ncols_expected(T) = round(Int, T / 0.1)                                 # recorded columns at dt=0.1, every=1

@testset "ensemble (tensor) batching" begin
    @testset "RNG batch axis: 4-arg ≡ 5-arg(0); distinct batches independent" begin
        a = [Dewdrop.draw_uniform(Float64, UInt64(7), s, e) for s in 0:40 for e in 1:40]
        b = [Dewdrop.draw_uniform(Float64, UInt64(7), s, e, 0) for s in 0:40 for e in 1:40]
        @test a == b                                                     # batch 0 reproduces scalar bits
        s0 = [Dewdrop.draw_uniform(Float64, UInt64(7), s, 3, 0) for s in 0:300]
        s1 = [Dewdrop.draw_uniform(Float64, UInt64(7), s, 3, 1) for s in 0:300]
        @test s0 != s1 && abs(cor(s0, s1)) < 0.2                         # independent streams
        @test [Dewdrop.draw_poisson(2.0, UInt64(7), s, 3) for s in 0:50] ==
              [Dewdrop.draw_poisson(2.0, UInt64(7), s, 3, 0) for s in 0:50]
    end

    @testset "bit-exact scalar reference (delta E/I + drive), input sweep" begin
        N, B = 100, 6
        inputs = [(b - 1) * 0.4 for b in 1:B]
        scal = [solve(_ei_delta(inputs[b]; N), FixedStep(0.1)).spike_count for b in 1:B]
        bs = solve(_ei_delta(0.0; N), FixedStep(0.1); batch = B, input = _colmat(inputs, N), streams = fill(0, B))
        @test size(bs.spike_count) == (N, B)
        @test all(bs.spike_count[:, b] == scal[b] for b in 1:B)          # every column == its scalar run
        @test size(firing_rate(bs)) == (N, B)
    end

    @testset "bit-exact reference for COBA and CUBA synapses" begin
        # Weights are exactly representable (multiples of powers of two), so the scatter's
        # ring-slot accumulation is ORDER-INDEPENDENT --- the bit-exact reference then holds even when
        # the scalar reference scatter runs multithreaded (its atomic path reorders the FP sum;
        # exact weights make any order give the same float). The batched scatter is deterministic
        # regardless (it threads over the disjoint batch axis with no atomics).
        N, B, T = 90, 4, 90.0
        inputs = [(b - 1) * 0.25 for b in 1:B]
        # COBA
        cobaprob(input) = let mc = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 5.0)
            ce = fixed_prob(ARCH, N, N, 0.1; weight = 0.5, delay = steps(1), seed = UInt64(1), sources = 1:72)
            ci = fixed_prob(ARCH, N, N, 0.1; weight = 4.0, delay = steps(1), seed = UInt64(2), sources = 73:N)
            DewdropNetwork(mc, N; input = input, tspan = (0.0, T), arch = ARCH,
                projections = (Projection(ConductanceSynapse(τ = 5.0, Erev = 0.0), ce),
                    Projection(ConductanceSynapse(τ = 10.0, Erev = -80.0), ci)),
                drive = PoissonDrive(rate = 6.0, weight = 0.1, seed = UInt64(7)))
        end
        sc = [solve(cobaprob(inputs[b]), FixedStep(0.1); v0 = -60.0).spike_count for b in 1:B]
        bc = solve(cobaprob(0.0), FixedStep(0.1); batch = B, input = _colmat(inputs, N), streams = fill(0, B), v0 = -60.0)
        @test all(bc.spike_count[:, b] == sc[b] for b in 1:B)
        # CUBA
        cubaprob(input) = DewdropNetwork(_lif(), N; input = input, tspan = (0.0, T), arch = ARCH,
            projection = Projection(CurrentSynapse(τ = 5.0), fixed_prob(ARCH, N, N, 0.1; weight = 1.0, delay = steps(3), seed = UInt64(4), allow_self = false)),
            drive = PoissonDrive(rate = 25.0, weight = 0.5, seed = UInt64(5)))
        su = [solve(cubaprob(inputs[b]), FixedStep(0.1)).spike_count for b in 1:B]
        bu = solve(cubaprob(0.0), FixedStep(0.1); batch = B, input = _colmat(inputs, N), streams = fill(0, B))
        @test all(bu.spike_count[:, b] == su[b] for b in 1:B)
    end

    @testset "independent drive streams: columns differ, stream-0 ≡ scalar" begin
        N, B = 100, 6
        bs = solve(_ei_delta(0.5; N), FixedStep(0.1); batch = B, streams = 0:(B - 1))
        sc0 = solve(_ei_delta(0.5; N), FixedStep(0.1)).spike_count
        @test bs.spike_count[:, 1] == sc0                                # stream 0 == scalar default
        @test all(bs.spike_count[:, b] != bs.spike_count[:, 1] for b in 2:B)
    end

    @testset "per-column random v0: shared scalar reference + independence" begin
        N, B = 80, 5
        # shared v0 scalar (no random) reference, input sweep
        inputs = [(b - 1) * 0.5 for b in 1:B]
        scal = [solve(_ei_delta(inputs[b]; N), FixedStep(0.1); v0 = 5.0).spike_count for b in 1:B]
        bs = solve(_ei_delta(0.0; N), FixedStep(0.1); batch = B, input = _colmat(inputs, N), streams = fill(0, B), v0 = 5.0)
        @test all(bs.spike_count[:, b] == scal[b] for b in 1:B)
        # random per-column v0 → distinct initial conditions per column
        br = solve(_ei_delta(0.5; N), FixedStep(0.1); batch = B, streams = fill(0, B), v0 = (0.0, 20.0))
        @test all(br.spike_count[:, b] != br.spike_count[:, 1] for b in 2:B)   # v0 drawn on the batch axis
        # per-neuron VECTOR v0 (length N) is broadcast across the batch: every column == scalar(vec)
        vec0 = [5.0 + 8.0 * (i / N) for i in 1:N]
        sv = solve(_ei_delta(0.5; N), FixedStep(0.1); v0 = vec0).spike_count
        bv = solve(_ei_delta(0.5; N), FixedStep(0.1); batch = B, streams = fill(0, B), v0 = vec0)
        @test all(bv.spike_count[:, b] == sv for b in 1:B)
        # a mis-sized matrix v0 errors at init (shape guard), not silently
        @test_throws DimensionMismatch solve(_ei_delta(0.5; N), FixedStep(0.1); batch = B, v0 = zeros(N, B + 1))
        @test_throws ArgumentError solve(_ei_delta(0.5; N), FixedStep(0.1); batch = B, streams = 0:(B - 2))
    end

    @testset "batched monitors bit-exact vs scalar (Spikes / Trace / subset / Aggregate)" begin
        N, B, T = 60, 5, 60.0
        inputs = [(b - 1) * 0.5 for b in 1:B]
        rec() = (spikes = Spikes(), V = Trace(:V), sub = Trace(:V; of = 10:20),
            rate = Aggregate(Spikes(), sum), mv = Aggregate(Trace(:V), :mean))
        scal = [solve(_ei_delta(inputs[b]; N, T), FixedStep(0.1); record = rec()) for b in 1:B]
        bs = solve(_ei_delta(0.0; N, T), FixedStep(0.1); batch = B, input = _colmat(inputs, N), streams = fill(0, B), record = rec())
        @test size(bs.record.spikes.data) == (N, B, _ncols_expected(T))
        @test all(bs.record.spikes.data[:, b, :] == scal[b].record.spikes.data for b in 1:B)
        @test all(bs.record.V.data[:, b, :] == scal[b].record.V.data for b in 1:B)
        @test all(bs.record.sub.data[:, b, :] == scal[b].record.sub.data for b in 1:B)
        @test all(bs.record.rate.data[b, :] == vec(scal[b].record.rate.data) for b in 1:B)
        @test all(bs.record.mv.data[b, :] ≈ vec(scal[b].record.mv.data) for b in 1:B)
    end


    @testset "JLArrays (no GPU) under allowscalar(false): batched ≡ scalar" begin
        GPUArrays.allowscalar(false)
        N, B = 70, 4
        inputs = [(b - 1) * 0.5 for b in 1:B]
        scal = [solve(_ei_delta(inputs[b]; N), FixedStep(0.1); record = (spikes = Spikes(),)).spike_count for b in 1:B]
        ig = adapt(JLArray, Dewdrop.init(_ei_delta(0.0; N), FixedStep(0.1); batch = B, input = _colmat(inputs, N), streams = fill(0, B), record = (spikes = Spikes(),)))
        @test typeof(Dewdrop.get_backend(ig.state.state.V)) !== typeof(Dewdrop.get_backend(zeros(1)))
        Dewdrop.solve!(ig)
        bs = Dewdrop.BatchedSolution(ig)
        @test all(Array(bs.spike_count)[:, b] == scal[b] for b in 1:B)
        @test bs.record.spikes.data isa Array{Bool, 3}                   # store came back host-resident
    end

    @testset "compacted scatter (scatter = :compacted) ≡ edge reference, per column" begin
        N, B = 120, 5
        inputs = [(b - 1) * 0.4 for b in 1:B]
        scal = [solve(_ei_delta(inputs[b]; N), FixedStep(0.1)).spike_count for b in 1:B]
        # CPU (KA.CPU): batched compacted matches the scalar reference bit-for-bit (exact delta weights)
        bc = solve(_ei_delta(0.0; N), FixedStep(0.1); batch = B, input = _colmat(inputs, N), streams = fill(0, B), scatter = :compacted)
        @test all(bc.spike_count[:, b] == scal[b] for b in 1:B)
        # JLArrays under allowscalar(false): the per-column compaction (active (N,B), na (B,)) engages
        ig = adapt(JLArray, Dewdrop.init(_ei_delta(0.0; N), FixedStep(0.1); batch = B, input = _colmat(inputs, N), streams = fill(0, B), scatter = :compacted))
        @test ig.compaction isa Dewdrop.BatchedCompactionScratch
        Dewdrop.solve!(ig)
        jb = Dewdrop.BatchedSolution(ig)
        @test all(Array(jb.spike_count)[:, b] == scal[b] for b in 1:B)
        @test_throws ArgumentError Dewdrop.init(_ei_delta(0.0; N), FixedStep(0.1); batch = B, scatter = :nope)
    end

    @testset "periodic sync is a barrier: result-invariant to sync_every" begin
        N, B = 100, 4
        a = solve(_ei_delta(0.3; N), FixedStep(0.1); batch = B, streams = 0:(B - 1), sync_every = 0).spike_count
        b = solve(_ei_delta(0.3; N), FixedStep(0.1); batch = B, streams = 0:(B - 1), sync_every = 7).spike_count
        @test a == b                                                     # syncing changes timing, never results
        # scalar path too (sync_every default vs disabled)
        s0 = solve(_ei_delta(0.3; N), FixedStep(0.1); sync_every = 0).spike_count
        s1 = solve(_ei_delta(0.3; N), FixedStep(0.1); sync_every = 5).spike_count
        @test s0 == s1
    end
end
