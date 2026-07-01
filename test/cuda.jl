using Dewdrop
using Test
using Statistics
using CUDA

# The CUDA GPU backend. The architecture seam (`array_type(GPU()) = CuArray`) plus the
# Adapt-movable, KernelAbstractions-kernel engine means a network built with `arch = GPU()` runs
# entirely on the device with no kernel changes. The arch-seam check runs anywhere CUDA loads;
# the device-simulation checks are guarded by `CUDA.functional()` (they need a real GPU).
@testset "CUDA GPU backend" begin
    @testset "architecture seam (no functional GPU required)" begin
        @test Dewdrop.array_type(Dewdrop.GPU()) === CuArray          # type-level; no device needed
    end

    if !CUDA.functional()
        @info "CUDA not functional on this host: skipping GPU device simulations"
    else
        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)

        @testset "counter-based drive RNG is bit-identical GPU == CPU" begin
            cpu = [Dewdrop.draw_poisson(2.0, UInt64(7), 5, i) for i in 1:256]
            gpu = Array(Dewdrop.draw_poisson.(2.0, UInt64(7), 5, CuArray(1:256)))
            @test gpu == cpu                                         # a pure (seed,step,entity) fn
            @test Dewdrop.architecture(CuArray([1.0, 2.0])) === Dewdrop.GPU()
        end

        # build the SAME network on a given architecture (deterministic connectivity + drive)
        function ei_net(arch; T = 200.0)
            ce = fixed_prob(arch, 200, 200, 0.1; weight = 0.5, delay = steps(15), seed = UInt64(1), sources = 1:160, allow_self = false)
            ci = fixed_prob(arch, 200, 200, 0.1; weight = -1.0, delay = steps(15), seed = UInt64(2), sources = 161:200, allow_self = false)
            return DewdropNetwork(
                m, 200; input = 0.0, tspan = (0.0, T), arch = arch,
                projections = (Projection(DeltaSynapse(), ce), Projection(DeltaSynapse(), ci)),
                drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = UInt64(3))
            )
        end

        @testset "connected + driven net + monitors: GPU runs on-device and matches CPU" begin
            rec = (spikes = Spikes(), V = Trace(:V), rate = Aggregate(Spikes(), sum))
            g = solve(ei_net(Dewdrop.GPU()), FixedStep(0.1); record = rec)
            c = solve(ei_net(Dewdrop.CPU()), FixedStep(0.1); record = rec)
            # state ran on the device; monitor STORES come back host-resident
            @test g.record.V.data isa Matrix{Float64}
            @test g.record.spikes.data isa Matrix{Bool}
            # exact-weight delta synapses → the atomic scatter is order-independent here → bit-equal
            @test sum(g.spike_count) == sum(c.spike_count)
            @test g.record.rate.data == c.record.rate.data
            @test vec(sum(g.record.spikes.data; dims = 1)) == vec(g.record.rate.data)   # raster ⇔ aggregate
            tg, ig = raster(g)
            @test length(tg) == sum(g.spike_count) && all(x -> 1 ≤ x ≤ 200, ig)
        end

        @testset "COBA conductance synapses + randomized init on GPU" begin
            function coba(arch)
                mc = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 5.0)
                ce = fixed_prob(arch, 500, 500, 0.05; weight = 0.6, delay = steps(1), seed = UInt64(1), sources = 1:400)
                ci = fixed_prob(arch, 500, 500, 0.05; weight = 6.7, delay = steps(1), seed = UInt64(2), sources = 401:500)
                return DewdropNetwork(
                    mc, 500; input = 0.0, tspan = (0.0, 200.0), arch = arch,
                    projections = (
                        Projection(ConductanceSynapse(τ = 5.0, Erev = 0.0), ce),
                        Projection(ConductanceSynapse(τ = 10.0, Erev = -80.0), ci),
                    ),
                    drive = PoissonDrive(rate = 6.0, weight = 0.1, seed = UInt64(7))
                )
            end
            g = solve(coba(Dewdrop.GPU()), FixedStep(0.1); v0 = (-60.0, -50.0), record = (spikes = Spikes(),))
            c = solve(coba(Dewdrop.CPU()), FixedStep(0.1); v0 = (-60.0, -50.0), record = (spikes = Spikes(),))
            @test isapprox(sum(g.spike_count), sum(c.spike_count); rtol = 0.02)   # atomic-scatter tolerance
        end

        @testset "deterministic subthreshold LIF: GPU ≈ CPU exact propagator" begin
            # no scatter, no drive, sub-θ → the only GPU/CPU difference is `exp` ULP (CUDA
            # libdevice vs CPU libm), so the trace matches to floating-point tolerance.
            sg = solve(
                DewdropNetwork(m, 64; input = 15.0, tspan = (0.0, 100.0), arch = Dewdrop.GPU()),
                FixedStep(0.1); record = (V = Trace(:V),)
            )
            sc = solve(
                DewdropNetwork(m, 64; input = 15.0, tspan = (0.0, 100.0), arch = Dewdrop.CPU()),
                FixedStep(0.1); record = (V = Trace(:V),)
            )
            @test sum(sg.spike_count) == 0 == sum(sc.spike_count)   # sub-θ (V∞ = 15 < Vθ = 20)
            @test sg.record.V.data ≈ sc.record.V.data
        end

        @testset "MultiModel (AdEx-E + LIF-I) runs on-device + matches CPU" begin
            # multi-type populations: the fused launch loops the groups, launching the per-neuron
            # kernel once per (model, range). On the GPU each group is its own monomorphic launch over
            # the union SoA; with no synapses the run is deterministic, matching the CPU per-group path.
            mE = AdEx(;
                C = 200.0, gL = 10.0, EL = -70.0, VT = -50.0, ΔT = 2.0, Vr = -58.0, Vpeak = 0.0,
                a = 2.0, b = 60.0, τw = 120.0, tref = 2.0
            )
            mI = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 2.0)
            net(arch) = DewdropNetwork(
                Dewdrop.MultiModel([mE, mI], [64, 64]), 128;
                input = vcat(fill(700.0, 64), fill(400.0, 64)), tspan = (0.0, 150.0), arch = arch
            )
            g = solve(net(Dewdrop.GPU()), FixedStep(0.05))
            c = solve(net(Dewdrop.CPU()), FixedStep(0.05))
            @test sum(Array(g.spike_count)[1:64]) > 0                  # AdEx group fired on-device
            @test sum(Array(g.spike_count)[65:128]) > 0               # LIF group fired on-device
            @test Array(g.spike_count) == c.spike_count               # per-group launches ≡ CPU (no scatter)
            @test Array(g.state.state.w) ≈ c.state.state.w            # adaptation column matches (exp ULP)
        end

        @testset "Int32 indices + Float32 path on the GPU (scatter perf knobs)" begin
            function ei(::Type{T}, N, IT; Tend = 150.0) where {T}
                mm = LIF(; τ = T(20), EL = T(0), Vθ = T(20), Vr = T(10), R = T(1), tref = T(2))
                ce = fixed_prob(Dewdrop.GPU(), N, N, 0.1; weight = T(0.5), delay = steps(15), seed = UInt64(1), sources = 1:(4N ÷ 5), allow_self = false, index_type = IT)
                ci = fixed_prob(Dewdrop.GPU(), N, N, 0.1; weight = T(-1.0), delay = steps(15), seed = UInt64(2), sources = (4N ÷ 5 + 1):N, allow_self = false, index_type = IT)
                DewdropNetwork(
                    mm, N; input = T(0), tspan = (T(0), T(Tend)), arch = Dewdrop.GPU(),
                    projections = (Projection(DeltaSynapse(), ce), Projection(DeltaSynapse(), ci)),
                    drive = PoissonDrive(rate = T(20), weight = T(0.5), seed = UInt64(3))
                )
            end
            # Int32 indices: identical connectome ⇒ bit-identical results (the edge-parallel scatter
            # reads conn.src/post regardless of index width)
            g64 = solve(ei(Float64, 4000, Int), FixedStep(0.1))
            g32idx = solve(ei(Float64, 4000, Int32), FixedStep(0.1))
            @test eltype(ei(Float64, 64, Int32).projections[1].conn.post) == Int32
            @test Array(g64.spike_count) == Array(g32idx.spike_count)
            # Float32 state: same dynamics within floating-point tolerance
            gf32 = solve(ei(Float32, 4000, Int32), FixedStep(0.1f0))
            @test eltype(gf32.state.state.V) == Float32
            @test isapprox(sum(Array(gf32.spike_count)), sum(Array(g64.spike_count)); rtol = 0.05)
        end

        @testset "compacted scatter on the GPU: bit-identical to edge-parallel + faster when sparse" begin
            function ein(N, p; Tend = 100.0, w = 0.01)
                mm = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
                ce = fixed_prob(Dewdrop.GPU(), N, N, p; weight = w, delay = steps(15), seed = UInt64(1), sources = 1:(4N ÷ 5), allow_self = false)
                ci = fixed_prob(Dewdrop.GPU(), N, N, p; weight = -w, delay = steps(15), seed = UInt64(2), sources = (4N ÷ 5 + 1):N, allow_self = false)
                DewdropNetwork(
                    mm, N; input = 0.0, tspan = (0.0, Tend), arch = Dewdrop.GPU(),
                    projections = (Projection(DeltaSynapse(), ce), Projection(DeltaSynapse(), ci)),
                    drive = PoissonDrive(rate = 2.0, weight = 0.4, seed = UInt64(3))
                )
            end
            # exact-weight delta → the compacted and edge scatters deposit the same set in any order
            small = ein(2000, 0.1)
            @test Array(solve(small, FixedStep(0.1); scatter = :compacted, advise = false).spike_count) ==
                Array(solve(small, FixedStep(0.1); scatter = :edge, advise = false).spike_count)

            # large + sparse: compaction should beat edge-parallel
            big = ein(40_000, 0.05)
            solve(big, FixedStep(0.1); scatter = :edge, advise = false)        # warm
            solve(big, FixedStep(0.1); scatter = :compacted, advise = false)
            te = minimum(@elapsed(CUDA.@sync solve(big, FixedStep(0.1); scatter = :edge, advise = false)) for _ in 1:2)
            tc = minimum(@elapsed(CUDA.@sync solve(big, FixedStep(0.1); scatter = :compacted, advise = false)) for _ in 1:2)
            @info "compacted vs edge scatter (large, sparse)" edge_s = te compacted_s = tc speedup = te / tc
            @test tc < te                                                       # compaction wins here
        end

        @testset "fused megakernel makes the GPU faster than the CPU at scale" begin
            # The fused dense step + pipelined (no per-step sync) device stream removes the
            # launch-bound tax: at N = 20k the GPU runs several × the CPU throughput. Assert a
            # conservative ≥ 1.5× (the measured margin is ~5×) so a shared host stays non-flaky.
            N, T = 20_000, 100.0
            netp(arch) = DewdropNetwork(
                m, N; input = 0.0, tspan = (0.0, T), arch = arch,
                projections = (
                    Projection(
                        DeltaSynapse(),
                        fixed_prob(arch, N, N, 0.02; weight = 0.5, delay = steps(15), seed = UInt64(1), sources = 1:(4N ÷ 5), allow_self = false)
                    ),
                    Projection(
                        DeltaSynapse(),
                        fixed_prob(arch, N, N, 0.02; weight = -1.0, delay = steps(15), seed = UInt64(2), sources = (4N ÷ 5 + 1):N, allow_self = false)
                    ),
                ),
                drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = UInt64(3))
            )
            rec = (spikes = Spikes(),)
            solve(netp(Dewdrop.GPU()), FixedStep(0.1); record = rec)   # warm up (compile)
            solve(netp(Dewdrop.CPU()), FixedStep(0.1); record = rec)
            tg = minimum(@elapsed(CUDA.@sync solve(netp(Dewdrop.GPU()), FixedStep(0.1); record = rec)) for _ in 1:3)
            tc = minimum(@elapsed(solve(netp(Dewdrop.CPU()), FixedStep(0.1); record = rec)) for _ in 1:3)
            @info "fused throughput" N gpu_s = tg cpu_s = tc speedup = tc / tg
            @test tc / tg ≥ 1.5
        end

        @testset "ensemble batching on the GPU: bit-exact reference + speedup vs sequential" begin
            # B independent instances sharing one CSR, varying only per-column input; with the
            # drive stream forced to 0 each column must match a scalar GPU solve bit-for-bit.
            N, B, T = 500, 8, 100.0
            netp(input, arch) = DewdropNetwork(
                m, N; input = input, tspan = (0.0, T), arch = arch,
                projections = (
                    Projection(
                        DeltaSynapse(),
                        fixed_prob(arch, N, N, 0.1; weight = 0.5, delay = steps(15), seed = UInt64(1), sources = 1:(4N ÷ 5), allow_self = false)
                    ),
                    Projection(
                        DeltaSynapse(),
                        fixed_prob(arch, N, N, 0.1; weight = -1.0, delay = steps(15), seed = UInt64(2), sources = (4N ÷ 5 + 1):N, allow_self = false)
                    ),
                ),
                drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = UInt64(3))
            )
            inputs = [(b - 1) * 0.3 for b in 1:B]
            inMat = Dewdrop.on_architecture(Dewdrop.GPU(), repeat(reshape(inputs, 1, B), N, 1))
            scal = [Array(solve(netp(inputs[b], Dewdrop.GPU()), FixedStep(0.1)).spike_count) for b in 1:B]
            bs = solve(netp(0.0, Dewdrop.GPU()), FixedStep(0.1); batch = B, input = inMat, streams = fill(0, B))
            bc = Array(bs.spike_count)
            @test size(bc) == (N, B)
            @test all(bc[:, b] == scal[b] for b in 1:B)              # every GPU column == its scalar GPU run

            # speedup: one B-batched solve vs B sequential scalar solves (same total work)
            Nb, Bb, Tb = 2000, 24, 150.0
            pin = netp(0.0, Dewdrop.GPU())   # warm compile
            solve(pin, FixedStep(0.1); batch = 4, streams = 0:3)
            big = DewdropNetwork(
                m, Nb; input = 0.0, tspan = (0.0, Tb), arch = Dewdrop.GPU(),
                projections = (
                    Projection(
                        DeltaSynapse(),
                        fixed_prob(Dewdrop.GPU(), Nb, Nb, 0.05; weight = 0.5, delay = steps(15), seed = UInt64(1), sources = 1:(4Nb ÷ 5), allow_self = false)
                    ),
                    Projection(
                        DeltaSynapse(),
                        fixed_prob(Dewdrop.GPU(), Nb, Nb, 0.05; weight = -1.0, delay = steps(15), seed = UInt64(2), sources = (4Nb ÷ 5 + 1):Nb, allow_self = false)
                    ),
                ),
                drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = UInt64(3))
            )
            solve(big, FixedStep(0.1); batch = Bb, streams = 0:(Bb - 1))     # warm
            for k in 1:Bb
                solve(big, FixedStep(0.1))
            end                   # warm
            tbatch = minimum(@elapsed(CUDA.@sync solve(big, FixedStep(0.1); batch = Bb, streams = 0:(Bb - 1))) for _ in 1:2)
            tseq = minimum(
                @elapsed(
                        CUDA.@sync (
                            for k in 1:Bb
                                solve(big, FixedStep(0.1))
                        end
                        )
                    ) for _ in 1:2
            )
            @info "ensemble batching throughput" N = Nb B = Bb batched_s = tbatch sequential_s = tseq speedup = tseq / tbatch
            @test tseq / tbatch ≥ 3.0                                  # measured ~12×; conservative guard
        end
    end
end
