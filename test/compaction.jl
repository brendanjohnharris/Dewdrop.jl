using Dewdrop
using Test
using JLArrays
using GPUArrays
using Adapt: adapt

# Compacted scatter (src/Compaction.jl): the opt-in device fast path that processes only ACTIVE
# synapses (compactify spiking neurons → 2-level launch over active×maxdeg). It must deposit the
# SAME set of contributions as the edge-parallel scatter. With exactly-representable weights the
# ring accumulation is order-independent, so compacted == edge == the serial CPU reference
# BIT-FOR-BIT. No GPU needed: JLArrays routes the device step through the compacted path, and
# allowscalar(false) proves the `na` host read uses a bulk copy (not a scalar index).

const ARCH = Dewdrop.CPU()
_lif() = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)

# run a problem under JLArrays with a given scatter mode (→ JLBackend → the device step)
function _jl_run(prob, alg; scatter, kw...)
    ig = adapt(JLArray, Dewdrop.init(prob, alg; scatter = scatter, kw...))
    Dewdrop.solve!(ig)
    return Dewdrop.DewdropSolution(ig)
end

@testset "compacted scatter" begin
    GPUArrays.allowscalar(false)

    @testset "compacted ≡ edge ≡ CPU reference (delta E/I + drive, exact weights)" begin
        N = 300
        ce = fixed_prob(ARCH, N, N, 0.1; weight = 0.5, delay = 5, seed = UInt64(1), sources = 1:(4N ÷ 5), allow_self = false)
        ci = fixed_prob(ARCH, N, N, 0.1; weight = -1.0, delay = 5, seed = UInt64(2), sources = (4N ÷ 5 + 1):N, allow_self = false)
        prob = DewdropNetwork(_lif(), N; input = 0.0, tspan = (0.0, 80.0), arch = ARCH,
            projections = (Projection(DeltaSynapse(), ce), Projection(DeltaSynapse(), ci)),
            drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = UInt64(3)))
        ref = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),), advise = false)
        comp = _jl_run(prob, FixedStep(0.1); scatter = :compacted, record = (spikes = Spikes(),))
        edge = _jl_run(prob, FixedStep(0.1); scatter = :edge, record = (spikes = Spikes(),))
        @test comp.spike_count isa JLArray || Array(comp.spike_count) == ref.spike_count
        @test Array(comp.spike_count) == ref.spike_count           # compacted == serial CPU
        @test Array(comp.spike_count) == Array(edge.spike_count)   # compacted == edge-parallel
        @test Array(comp.record.spikes.data) == ref.record.spikes.data

        # `scatter = :compacted` is HONOURED on a native CPU Array backend too (not a silent no-op):
        # the broadcast propagate routes through the same compaction seam (KA.CPU compaction kernels).
        cpu_comp = solve(prob, FixedStep(0.1); scatter = :compacted, advise = false)
        @test cpu_comp.spike_count == ref.spike_count
    end

    @testset "compacted ≡ CPU reference for COBA (exact weights)" begin
        N = 200
        mc = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 5.0)
        ce = fixed_prob(ARCH, N, N, 0.08; weight = 0.5, delay = 1, seed = UInt64(1), sources = 1:160)
        ci = fixed_prob(ARCH, N, N, 0.08; weight = 4.0, delay = 1, seed = UInt64(2), sources = 161:N)
        prob = DewdropNetwork(mc, N; input = 0.0, tspan = (0.0, 100.0), arch = ARCH,
            projections = (Projection(ConductanceSynapse(τ = 5.0, Erev = 0.0), ce),
                Projection(ConductanceSynapse(τ = 10.0, Erev = -80.0), ci)),
            drive = PoissonDrive(rate = 6.0, weight = 0.1, seed = UInt64(7)))
        ref = solve(prob, FixedStep(0.1); v0 = -60.0, advise = false).spike_count
        comp = _jl_run(prob, FixedStep(0.1); scatter = :compacted, v0 = -60.0)
        @test Array(comp.spike_count) == ref
    end

    @testset "empty + single-edge connectivity compact cleanly" begin
        # zero spikes / zero edges must not crash the compactify or the 2-level launch
        N = 64
        prob = DewdropNetwork(_lif(), N; input = 0.0, tspan = (0.0, 30.0), arch = ARCH,
            projection = Projection(DeltaSynapse(), fixed_prob(ARCH, N, N, 0.1; weight = 0.5, delay = 2, seed = UInt64(4))),
            drive = PoissonDrive(rate = 5.0, weight = 0.5, seed = UInt64(9)))
        comp = _jl_run(prob, FixedStep(0.1); scatter = :compacted)
        ref = solve(prob, FixedStep(0.1); advise = false).spike_count
        @test Array(comp.spike_count) == ref
    end

    @testset "scatter mode validation" begin
        prob = DewdropNetwork(_lif(), 8; input = 12.0, tspan = (0.0, 5.0), arch = ARCH)
        @test_throws ArgumentError Dewdrop.init(prob, FixedStep(0.1); scatter = :nonsense)
        @test solve(prob, FixedStep(0.1); scatter = :edge, advise = false) isa Dewdrop.DewdropSolution
    end

    @testset "scatter = :auto crossover resolution" begin
        # `_resolve_scatter` only reads each connectome's edge count + index eltype, so a UnitRange `post`
        # (O(1) memory) stands in for an arbitrarily large connectome --- no million-edge build needed.
        fakeconn(ne) = Dewdrop.SparseCSR([1, 1], 1:ne, Float64[], Int[], 1:1, 1, 1, 1)
        prob(arch, ne; plastic = nothing) = DewdropNetwork(_lif(), 1000; input = 0.0, tspan = (0.0, 1.0),
            arch = arch, projection = Projection(DeltaSynapse(), fakeconn(ne); plasticity = plastic))
        unconn(arch) = DewdropNetwork(_lif(), 1000; input = 0.0, tspan = (0.0, 1.0), arch = arch)
        res(s, arch, p) = Dewdrop._resolve_scatter(s, arch, p.projections)
        L2 = Dewdrop._l2_cache_bytes(Dewdrop.GPU())
        small = 100_000                                       # ~0.8 MB index footprint → L2-resident
        large = (L2 ÷ 2) ÷ sizeof(Int) + 1_000_000            # comfortably past the L2/2 crossover
        # GPU: small → :edge (the no-sync edge launch wins), large → :compacted (the edge rescan spills L2)
        @test res(:auto, Dewdrop.GPU(), prob(Dewdrop.GPU(), small)) === :edge
        @test res(:auto, Dewdrop.GPU(), prob(Dewdrop.GPU(), large)) === :compacted
        # CPU never compacts (its scatter already walks only spiking rows); unconnected → :edge
        @test res(:auto, Dewdrop.CPU(), prob(Dewdrop.CPU(), large)) === :edge
        @test res(:auto, Dewdrop.GPU(), unconn(Dewdrop.GPU())) === :edge
        # plastic projections must stay on :edge (compacted can't drive STDP potentiation)
        stdp = Dewdrop.STDP(; Aplus = 0.01, Aminus = 0.012, τplus = 20.0, τminus = 20.0)
        @test res(:auto, Dewdrop.GPU(), prob(Dewdrop.GPU(), large; plastic = stdp)) === :edge
        # an explicit mode always passes through unchanged
        @test res(:edge, Dewdrop.GPU(), prob(Dewdrop.GPU(), large)) === :edge
        @test res(:compacted, Dewdrop.GPU(), prob(Dewdrop.GPU(), small)) === :compacted
    end
end
