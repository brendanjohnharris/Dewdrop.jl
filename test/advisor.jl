using Dewdrop
using Test
using CUDA

# The performance advisor (src/Advisor.jl) only inspects METADATA (architecture, float type,
# connectivity index width, degree) and a firing fraction, so a GPU()-arch problem can be built
# with CPU-resident connectivity and probed WITHOUT a functional GPU --- which is exactly how this
# suite exercises every branch. Each distinct suggestion fires at most once per session (deduped),
# and the whole advisor is silenceable globally / per call.

# a GPU-arch problem whose connectivity lives on the host (advice reads it without touching a device)
function _gpuprob(; T = Float64, IT = Int, N = 4000, p = 0.1)
    m = LIF(; τ = T(20), EL = T(0), Vθ = T(20), Vr = T(10), R = T(1), tref = T(2))
    conn = fixed_prob(Dewdrop.CPU(), N, N, p; weight = T(0.5), delay = steps(1), seed = UInt64(1), index_type = IT)
    return DewdropNetwork(m, N; input = T(0), tspan = (T(0), T(10)), arch = Dewdrop.GPU(),
        projection = Projection(DeltaSynapse(), conn))
end

@testset "performance advisor" begin
    Dewdrop.set_advice!(true)

    @testset "static: Float64-on-GPU → Float32, Int64 indices → Int32 (fire once each)" begin
        prob = _gpuprob()
        Dewdrop.reset_advice!()
        @test_logs (:info, r"Float32") (:info, r"Int32") Dewdrop._advise_static(prob)
        @test_logs Dewdrop._advise_static(prob)              # deduped on the second call → silent
    end

    @testset "no false advice: an all-Float32/Int32 GPU problem is silent" begin
        prob = _gpuprob(; T = Float32, IT = Int32)
        Dewdrop.reset_advice!()
        @test_logs Dewdrop._advise_static(prob)              # nothing left to suggest
    end

    @testset "CPU problems get no GPU advice" begin
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
        conn = fixed_prob(Dewdrop.CPU(), 100, 100, 0.1; weight = 0.5, delay = steps(1), seed = UInt64(1))
        cpuprob = DewdropNetwork(m, 100; input = 0.0, tspan = (0.0, 10.0), arch = Dewdrop.CPU(),
            projection = Projection(DeltaSynapse(), conn))
        Dewdrop.reset_advice!()
        @test_logs Dewdrop._advise_static(cpuprob)
        @test_logs Dewdrop._advise_runtime(cpuprob, 0.001)
    end

    @testset "CPU: large multithreaded network → suggest backend = Turbo" begin
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
        big = DewdropNetwork(m, 12000; input = 0.0, tspan = (0.0, 1.0))   # N ≥ 10k, canonical schedule
        small = DewdropNetwork(m, 1000; input = 0.0, tspan = (0.0, 1.0))   # N < 10k
        if Threads.nthreads() > 1                                          # the advice is for multicore
            Dewdrop.reset_advice!()
            @test_logs (:info, r"Turbo") Dewdrop._advise_cpu(big)        # Auto already fuses → point to the SIMD backend
            @test_logs Dewdrop._advise_cpu(big)                          # deduped on the second call → silent
        end
        Dewdrop.reset_advice!()
        @test_logs Dewdrop._advise_cpu(small)                            # N too small → silent (any thread count)
    end

    @testset "runtime regimes pick the right specialised path" begin
        sparse_big = _gpuprob(; N = 4000, p = 0.1)           # nedges ≈ 1.6M, mean degree 400
        Dewdrop.reset_advice!()
        @test_logs (:info, r"compacted scatter") Dewdrop._advise_runtime(sparse_big, 0.005)  # sparse + large

        dense = _gpuprob(; N = 4000, p = 0.16)               # mean degree ≈ 640 (> 500)
        Dewdrop.reset_advice!()
        @test_logs (:info, r"gather/SpMV") Dewdrop._advise_runtime(dense, 0.10)               # dense + high firing

        small = _gpuprob(; N = 2000, p = 0.1)                # n < 5000, nedges < 1M
        Dewdrop.reset_advice!()
        @test_logs (:info, r"launch-bound") Dewdrop._advise_runtime(small, 0.005)             # small + quiet
    end

    @testset "silencing: set_advice!(false) suppresses everything" begin
        prob = _gpuprob()
        Dewdrop.reset_advice!()
        Dewdrop.set_advice!(false)
        @test_logs Dewdrop._advise_static(prob)
        @test_logs Dewdrop._advise_runtime(prob, 0.005)
        Dewdrop.set_advice!(true)
    end

    Dewdrop.set_advice!(false)   # restore the suite-wide default (set in runtests.jl)
end
