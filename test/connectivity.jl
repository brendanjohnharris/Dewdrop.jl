using Dewdrop
using Test
using Adapt
using JLArrays

# M0 contract 5 --- connectivity is an interface (`for_each_post`) backed by CSR
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
    a = fixed_prob(arch, 200, 200, 0.1; weight = 0.5, delay = 3, seed = UInt64(9))
    b = fixed_prob(arch, 200, 200, 0.1; weight = 0.5, delay = 3, seed = UInt64(9), index_type = Int32)
    @test a.post == b.post && a.rowptr == b.rowptr && a.src == b.src
    @test eltype(b.post) == Int32
end
