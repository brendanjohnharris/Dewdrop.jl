using Dewdrop
using Test
using Adapt
using JLArrays

# Connectivity is an interface (`for_each_post`) backed by CSR
# arrays over *source* neurons, never a dense [post x pre] matrix. Per-synapse
# weight and delay live in CSR-parallel arrays. This keeps event-driven scatter and
# (later) procedural connectivity expressible, and is Adapt-movable to a device.
@testset "connectivity interface (CSR)" begin
    arch = Dewdrop.CPU()
    # edges as (pre, post, weight, delay_steps); 3 -> nothing
    #   1 -> (2, 0.5, 1), (3, 0.25, 2)
    #   2 -> (3, 1.0, 1)
    edges = [(1, 2, 0.5f0, 1), (1, 3, 0.25f0, 2), (2, 3, 1.0f0, 1)]
    conn = Dewdrop.SparseCSR(arch, edges; npre = 3, npost = 3)

    @test conn isa Dewdrop.AbstractConnectivity
    @test Dewdrop.npre(conn) == 3
    @test Dewdrop.npost(conn) == 3
    @test Dewdrop.nedges(conn) == 3

    # for_each_post iterates a presynaptic neuron's out-edges, in CSR order
    got = Tuple{Int, Float32, Int}[]
    Dewdrop.for_each_post(conn, 1) do post, w, d
        push!(got, (post, w, d))
    end
    @test got == [(2, 0.5f0, 1), (3, 0.25f0, 2)]

    # a source neuron with no out-edges iterates zero times
    n3 = Ref(0)
    Dewdrop.for_each_post(conn, 3) do post, w, d
        n3[] += 1
    end
    @test n3[] == 0

    # total weight delivered by pre 1 == 0.75
    acc = Ref(0.0f0)
    Dewdrop.for_each_post(conn, 1) do post, w, d
        acc[] += w
    end
    @test acc[] == 0.75f0

    # Adapt-movable: CSR-parallel arrays become device arrays
    gconn = adapt(JLArray, conn)
    @test gconn.post isa JLArray
    @test gconn.weight isa JLArray{Float32}
    @test gconn.delay isa JLArray
    @test Dewdrop.npre(gconn) == 3
end

@testset "empty connectivity" begin
    arch = Dewdrop.CPU()
    conn = Dewdrop.SparseCSR(arch, Tuple{Int, Int, Float32, Int}[]; npre = 4, npost = 4)
    @test Dewdrop.nedges(conn) == 0
    @test eltype(conn.weight) == Float32        # default weight type when no edges
    @test all(==(1), conn.rowptr)               # 1-based CSR, every row empty
    seen = Ref(0)
    for i in 1:4
        Dewdrop.for_each_post(conn, i) do post, w, d
            seen[] += 1
        end
    end
    @test seen[] == 0
end

@testset "edge-source array + Int32 indices" begin
    arch = Dewdrop.CPU()
    edges = [(1, 2, 0.5f0, 1), (1, 3, 0.25f0, 2), (2, 3, 1.0f0, 1)]
    conn = Dewdrop.SparseCSR(arch, edges; npre = 3, npost = 3)
    # `src` is the inverse of rowptr: src[e] is the presynaptic neuron owning edge e (parallel to post)
    @test conn.src == [1, 1, 2]
    @test all(conn.rowptr[conn.src[e]] ≤ e < conn.rowptr[conn.src[e] + 1] for e in 1:Dewdrop.nedges(conn))

    # Int32 indices: narrower rowptr/post/delay/src, same connectome + same src
    c32 = Dewdrop.SparseCSR(arch, edges; npre = 3, npost = 3, index_type = Int32)
    @test eltype(c32.rowptr) == Int32 && eltype(c32.post) == Int32
    @test eltype(c32.delay) == Int32 && eltype(c32.src) == Int32
    @test eltype(c32.weight) == Float32                      # weight type unchanged
    @test c32.rowptr == conn.rowptr && c32.post == conn.post && c32.src == conn.src

    # fixed_prob with index_type = Int32 yields the SAME connectome as Int64 (bit-identical sampling)
    a = fixed_prob(arch, 200, 200, 0.1; weight = 0.5, delay = steps(3), seed = UInt64(9))
    b = fixed_prob(arch, 200, 200, 0.1; weight = 0.5, delay = steps(3), seed = UInt64(9), index_type = Int32)
    @test a.post == b.post && a.rowptr == b.rowptr && a.src == b.src
    @test eltype(b.post) == Int32
end

# correlate_weights!: in-degree-normalised (1/√k) weights with relative Gaussian jitter, reproducible from
# the seed: the generic primitive spatial E/I models consume. See networkspec.jl for the curried
# `correlate_weights` used through the builder's `adjust` hook.
@testset "correlate_weights! (in-degree normalisation)" begin
    arch = Dewdrop.CPU()
    _mean(x) = sum(x) / length(x)
    mk() = fixed_prob(arch, 30, 30, 0.5; weight = 1.0, delay = steps(1), seed = UInt64(1))
    c = mk()
    correlate_weights!(c, 0.1; seed = UInt64(7))
    k = zeros(Int, 30)
    for p in c.post
        k[p] += 1
    end
    sum_k = sum(kk for kk in k if kk > 0)
    sum_sqrt = sum(sqrt(kk) for kk in k if kk > 0)
    J_rec = 0.1 * sum_k / sum_sqrt
    for p in unique(c.post)                              # mean weight into target p ≈ J_rec/√k[p]
        @test isapprox(_mean(c.weight[c.post .== p]), J_rec / sqrt(k[p]); rtol = 0.2)
    end
    c2 = mk()
    correlate_weights!(c2, 0.1; seed = UInt64(7))
    @test c.weight == c2.weight                          # reproducible from the seed

    # zero-in-degree convention. c3 has an isolated target (3); c2 is the same wiring with no empty target.
    mkc3() = Dewdrop.SparseCSR(arch, [(1, 1, 1.0, 1), (1, 2, 1.0, 1)]; npre = 1, npost = 3)  # target 3 unwired
    mkc2() = Dewdrop.SparseCSR(arch, [(1, 1, 1.0, 1), (1, 2, 1.0, 1)]; npre = 1, npost = 2)  # no empty target
    # default count_empty=true (BrainPy √max(k,1)): the empty target adds √1=1 to the denominator, so
    # Σ√max(k,1) = √1+√1+√1 = 3 vs √1+√1 = 2 → J_rec (and every weight) scales by 2/3.
    c3b = mkc3(); correlate_weights!(c3b, 0.2; targets = 1:3, seed = UInt64(3))
    c2b = mkc2(); correlate_weights!(c2b, 0.2; targets = 1:2, seed = UInt64(3))
    @test c3b.weight ≈ c2b.weight .* (2 / 3)
    # count_empty=false (principled): the empty target is irrelevant to the scale
    c3p = mkc3(); correlate_weights!(c3p, 0.2; targets = 1:3, seed = UInt64(3), count_empty = false)
    c2p = mkc2(); correlate_weights!(c2p, 0.2; targets = 1:2, seed = UInt64(3), count_empty = false)
    @test c3p.weight == c2p.weight
    @test c2b.weight == c2p.weight                        # no empty target → the two conventions agree
end

# correlate_weights! on a DEVICE connectome (GPU build path): the in-degree count + per-edge weight
# assignment are serial, so they run on a host copy of `post` with a single bulk write-back, never
# scalar-indexing the device array. A JLArray connectome (which bans scalar indexing, like CUDA) exercises
# this; the result must be identical to the host path.
@testset "correlate_weights! on a device connectome (no scalar indexing)" begin
    c = fixed_prob(Dewdrop.CPU(), 64, 64, 0.3; weight = 1.0, delay = steps(1), seed = UInt64(11))
    cdev = adapt(JLArray, c)
    correlate_weights!(cdev, 0.1; seed = UInt64(7))      # would scalar-index a device array if not host-staged
    correlate_weights!(c, 0.1; seed = UInt64(7))         # host reference (same edges + seed)
    @test cdev.weight isa JLArray
    @test Array(cdev.weight) == c.weight                 # bit-identical to the host path
    # sub-population target range on device, too
    cs = fixed_prob(Dewdrop.CPU(), 64, 64, 0.3; weight = 1.0, delay = steps(1), seed = UInt64(12), targets = 10:40)
    csdev = adapt(JLArray, cs)
    correlate_weights!(csdev, 0.2; targets = 10:40, seed = UInt64(8))
    correlate_weights!(cs, 0.2; targets = 10:40, seed = UInt64(8))
    @test Array(csdev.weight) == cs.weight
end
