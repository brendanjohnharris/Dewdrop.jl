using Dewdrop
using Test
using Statistics

# M3 --- spatial / structured connectivity: positions + a distance kernel build distance-
# dependent, ring and grid topologies (with optional periodic boundaries) on the same SparseCSR
# the engine consumes.
@testset "spatial / structured connectivity" begin
    @testset "position layouts + distance" begin
        @test length(line_positions(10)) == 10
        @test grid_positions(3, 2) == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]
        @test length(ring_positions(8)) == 8
        @test Dewdrop.distance((0.0, 0.0), (3.0, 4.0), nothing) ≈ 5.0
        @test Dewdrop.distance((0.0,), (9.0,), (10.0,)) ≈ 1.0     # minimum-image wraparound
        @test Dewdrop.distance((0.0,), (9.0,), nothing) ≈ 9.0     # no wrap
        @test box_kernel(2.0)(1.0) == 1.0 && box_kernel(2.0)(3.0) == 0.0
        @test gaussian_kernel(1.0)(0.0) == 1.0 && exponential_kernel(1.0)(0.0) == 1.0
    end

    pos = line_positions(50; spacing = 1.0)

    @testset "distance-dependent connectivity (box kernel = local lattice)" begin
        conn = distance_prob(Dewdrop.CPU(), pos; kernel = box_kernel(3.0), weight = 1.0, delay = steps(1), seed = UInt64(1))
        @test conn isa SparseCSR && Dewdrop.nedges(conn) > 0
        maxd = Ref(0.0)
        self = Ref(0)
        for pre in 1:50
            Dewdrop.for_each_post(conn, pre) do post, w, d
                maxd[] = max(maxd[], abs(pos[pre][1] - pos[post][1]))
                post == pre && (self[] += 1)
            end
        end
        @test maxd[] ≤ 3.0          # all edges within the kernel radius
        @test self[] == 0           # no autapses
        # reproducible
        conn2 = distance_prob(Dewdrop.CPU(), pos; kernel = box_kernel(3.0), weight = 1.0, delay = steps(1), seed = UInt64(1))
        @test conn.post == conn2.post && conn.rowptr == conn2.rowptr
    end

    @testset "targets restricts the postsynaptic set" begin
        # E(1:25) → I(26:50) spatial projection: every edge lands in the target range
        conn = distance_prob(Dewdrop.CPU(), pos; kernel = box_kernel(10.0), weight = 1.0, delay = steps(1),
            seed = UInt64(1), sources = 1:25, targets = 26:50)
        posts = Int[]
        srcs = Int[]
        for pre in 1:50
            Dewdrop.for_each_post(conn, pre) do post, w, d
                push!(posts, post)
                push!(srcs, pre)
            end
        end
        @test !isempty(posts)
        @test all(in(26:50), posts)     # targets honoured
        @test all(in(1:25), srcs)       # sources honoured
    end

    @testset "periodic boundary wraps the seam" begin
        connp = distance_prob(Dewdrop.CPU(), pos; kernel = box_kernel(3.0), weight = 1.0,
            delay = steps(1), seed = UInt64(1), period = (50.0,))
        wrapped = Ref(false)
        Dewdrop.for_each_post(connp, 1) do post, w, d   # neuron 1 (x=0) reaches x≈49 across the seam
            post ≥ 48 && (wrapped[] = true)
        end
        @test wrapped[]
    end

    @testset "random positions (uniform in a box, reproducible)" begin
        pos = random_positions(200, (1.0, 2.0); seed = UInt64(1))
        @test length(pos) == 200
        @test all(p -> 0 ≤ p[1] < 1.0 && 0 ≤ p[2] < 2.0, pos)       # within the domain
        @test eltype(pos) == NTuple{2, Float64}
        @test pos == random_positions(200, (1.0, 2.0); seed = UInt64(1))   # reproducible
        @test pos != random_positions(200, (1.0, 2.0); seed = UInt64(2))   # seed-dependent
    end

    @testset "fixed-count distance connectivity (Gumbel-max top-k)" begin
        pos = line_positions(100; spacing = 1.0)
        conn = distance_fixed_count(Dewdrop.CPU(), pos; kernel = exponential_kernel(3.0), count = 500,
            weight = 1.0, delay = steps(1), seed = UInt64(1))
        @test conn isa SparseCSR
        @test Dewdrop.nedges(conn) == 500                           # EXACTLY count edges (the point)
        # reproducible: same seed → identical realised connectome
        c2 = distance_fixed_count(Dewdrop.CPU(), pos; kernel = exponential_kernel(3.0), count = 500,
            weight = 1.0, delay = steps(1), seed = UInt64(1))
        @test c2.post == conn.post && c2.rowptr == conn.rowptr
        # localised: the exponential kernel concentrates edges on nearby pairs
        dists = Float64[]
        for pre in 1:100
            Dewdrop.for_each_post(conn, pre) do post, w, d
                push!(dists, abs(pos[pre][1] - pos[post][1]))
            end
        end
        @test mean(dists) < 20.0                                    # σ=3 localises (uniform ≈ 33 on a 100-line)

        # box kernel → a hard connection radius
        cb = distance_fixed_count(Dewdrop.CPU(), pos; kernel = box_kernel(5.0), count = 300,
            weight = 1.0, delay = steps(1), seed = UInt64(2))
        @test Dewdrop.nedges(cb) == 300
        maxd = 0.0
        for pre in 1:100
            Dewdrop.for_each_post(cb, pre) do post, w, d
                maxd = max(maxd, abs(pos[pre][1] - pos[post][1]))
            end
        end
        @test maxd ≤ 5.0                                            # never connects beyond the kernel support

        # sources / targets restrict the sampled pairs (E → I projection)
        ct = distance_fixed_count(Dewdrop.CPU(), pos; kernel = exponential_kernel(5.0), count = 200,
            weight = 1.0, delay = steps(1), seed = UInt64(3), sources = 1:50, targets = 51:100)
        @test Dewdrop.nedges(ct) == 200
        pairs = Tuple{Int, Int}[]
        for pre in 1:100
            Dewdrop.for_each_post(ct, pre) do post, w, d
                push!(pairs, (pre, post))
            end
        end
        @test all(pr -> pr[1] in 1:50 && pr[2] in 51:100, pairs)
    end

    @testset "Gaussian kernel → local connectivity + runs end-to-end" begin
        cg = distance_prob(Dewdrop.CPU(), pos; kernel = gaussian_kernel(2.0), weight = 1.0, delay = steps(1), seed = UInt64(7))
        dists = Float64[]
        for pre in 1:50
            Dewdrop.for_each_post(cg, pre) do post, w, d
                push!(dists, abs(pos[pre][1] - pos[post][1]))
            end
        end
        @test mean(dists) < 5.0     # localised, not uniform (~16 for a 50-line)

        m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
        sol = solve(DewdropNetwork(m, 50; input = 1.5, tspan = (0.0, 50.0),
                projection = Projection(CurrentSynapse(τ = 5.0), cg)), FixedStep(0.1))
        @test sum(sol.spike_count) ≥ 0
    end
end
