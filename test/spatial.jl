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
        conn = distance_prob(Dewdrop.CPU(), pos; kernel = box_kernel(3.0), weight = 1.0, delay = 1, seed = UInt64(1))
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
        conn2 = distance_prob(Dewdrop.CPU(), pos; kernel = box_kernel(3.0), weight = 1.0, delay = 1, seed = UInt64(1))
        @test conn.post == conn2.post && conn.rowptr == conn2.rowptr
    end

    @testset "periodic boundary wraps the seam" begin
        connp = distance_prob(Dewdrop.CPU(), pos; kernel = box_kernel(3.0), weight = 1.0,
            delay = 1, seed = UInt64(1), period = (50.0,))
        wrapped = Ref(false)
        Dewdrop.for_each_post(connp, 1) do post, w, d   # neuron 1 (x=0) reaches x≈49 across the seam
            post ≥ 48 && (wrapped[] = true)
        end
        @test wrapped[]
    end

    @testset "Gaussian kernel → local connectivity + runs end-to-end" begin
        cg = distance_prob(Dewdrop.CPU(), pos; kernel = gaussian_kernel(2.0), weight = 1.0, delay = 1, seed = UInt64(7))
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
