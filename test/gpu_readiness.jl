using Dewdrop
using Test
using Adapt
using StructArrays
using JLArrays
using GPUArrays

# The GPU-readiness suite turns "GPU-aware architecture" into enforced contracts,
# checkable on CPU with NO GPU present via JLArrays + allowscalar(false). This is the
# guard against the CPU-convenience "bolt-on" trap: it fails CI the moment the core
# acquires host-side scalar indexing, non-isbits state, non-movable structures, or
# order-dependent RNG.
@testset "GPU-readiness contracts" begin
    arch = Dewdrop.CPU()

    @testset "isbits SoA + adapt-movability (contracts 2, 5)" begin
        pop = Dewdrop.Population(arch, Float32, (:V, :ge, :gi), 16)
        @test isbitstype(eltype(pop.state))
        gpop = adapt(JLArray, pop)
        @test gpop.state.V isa JLArray{Float32}
        @test isbitstype(eltype(gpop.state))

        edges = [(i, mod1(i + 1, 8), 1.0f0, 1) for i in 1:8]
        conn = Dewdrop.SparseCSR(arch, edges; npre = 8, npost = 8)
        gconn = adapt(JLArray, conn)
        @test gconn.post isa JLArray
        @test isbitstype(eltype(gconn.weight))
        @test isbitstype(eltype(gconn.delay))
    end

    @testset "no host scalar indexing on the broadcast path (contract 3)" begin
        pop = Dewdrop.Population(arch, Float32, (:V, :ge), 32)
        gpop = adapt(JLArray, pop)
        GPUArrays.allowscalar(false)
        # a fused, exponential-Euler-shaped state update must run with scalar
        # indexing disallowed (Tier-1 dynamics path).
        gpop.state.ge .= 1.0f0
        @. gpop.state.V = gpop.state.V - 0.1f0 * gpop.state.V + gpop.state.ge
        @test all(==(1.0f0), Array(gpop.state.V))
    end

    @testset "RNG determinism across thread count (contract 4)" begin
        seed = UInt64(0x1234)
        N = 2048
        seq = [Dewdrop.draw_uniform(Float32, seed, 3, i) for i in 1:N]
        par = Vector{Float32}(undef, N)
        Threads.@threads for i in 1:N
            par[i] = Dewdrop.draw_uniform(Float32, seed, 3, i)
        end
        @test par == seq
    end

    @testset "hot-path type stability + allocation (RNG)" begin
        @test (@inferred Dewdrop.draw_uniform(Float64, UInt64(1), 1, 1)) isa Float64
        Dewdrop.draw_uniform(Float64, UInt64(1), 1, 1)  # warm
        @test @allocated(Dewdrop.draw_uniform(Float64, UInt64(1), 1, 1)) == 0
    end

    @testset "engine step! is GPU-ready (contract 3)" begin
        m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
        prob = DewdropNetwork(m, 64; input = 0.5, tspan = (0.0, 5.0))
        cpu = init(prob, FixedStep(0.1))
        gpu = adapt(JLArray, cpu)
        @test gpu.state.state.V isa JLArray
        @test gpu.spiked isa JLArray
        @test gpu.spike_count isa JLArray

        # the ENTIRE step (all dense phases) must run with scalar indexing disallowed,
        # and must agree step-for-step with the CPU run (deterministic, no atomics yet).
        GPUArrays.allowscalar(false)
        for _ in 1:50
            step!(gpu)
        end
        for _ in 1:50
            step!(cpu)
        end
        @test gpu.n == cpu.n
        @test Array(gpu.state.state.V) ≈ cpu.state.state.V
        @test Array(gpu.spike_count) == cpu.spike_count
    end
end
