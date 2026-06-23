using Dewdrop
using Test

# M2 --- random fixed-probability connectivity, the workhorse for balanced E/I networks.
# Each possible (pre, post) edge is present with probability `p`, sampled reproducibly from
# the counter-based RNG keyed by (pre, post). Weights/delays may be scalars or per-source
# functions (so excitatory vs inhibitory neurons get signed weights).
@testset "FixedProb connectivity" begin
    arch = Dewdrop.CPU()
    npre, npost, p = 200, 150, 0.1
    seed = UInt64(42)
    conn = Dewdrop.fixed_prob(arch, npre, npost, p; weight = 0.5f0, delay = steps(2), seed = seed)

    @test conn isa Dewdrop.SparseCSR
    @test Dewdrop.npre(conn) == npre
    @test Dewdrop.npost(conn) == npost
    # expected edge count ≈ p · npre · npost
    expected = p * npre * npost
    @test 0.9 * expected < Dewdrop.nedges(conn) < 1.1 * expected
    @test all(==(0.5f0), conn.weight)
    @test all(==(2), conn.delay)

    # reproducible: same seed → identical realised connectome
    conn2 = Dewdrop.fixed_prob(arch, npre, npost, p; weight = 0.5f0, delay = steps(2), seed = seed)
    @test conn.post == conn2.post
    @test conn.rowptr == conn2.rowptr
    # a different seed gives a different connectome
    conn3 = Dewdrop.fixed_prob(arch, npre, npost, p; weight = 0.5f0, delay = steps(2), seed = UInt64(43))
    @test conn3.post != conn2.post

    # no self-connections when disallowed
    rec = Dewdrop.fixed_prob(arch, 100, 100, 0.2; weight = 1.0f0, delay = steps(1), seed = seed, allow_self = false)
    self = 0
    for pre in 1:100
        Dewdrop.for_each_post(rec, pre) do post, w, d
            post == pre && (self += 1)
        end
    end
    @test self == 0

    # per-source signed weights (excitatory neurons +, inhibitory −)
    NE = 100
    ei = Dewdrop.fixed_prob(arch, 150, 150, 0.1; weight = pre -> pre ≤ NE ? 1.0f0 : -4.0f0, delay = steps(1), seed = seed)
    @test any(>(0), ei.weight)
    @test any(<(0), ei.weight)

    # `sources` restricts which presynaptic neurons emit edges (for E/I subpopulations)
    sub = Dewdrop.fixed_prob(arch, 200, 150, 0.2; weight = 1.0f0, delay = steps(1), seed = seed, sources = 1:50)
    @test Dewdrop.npre(sub) == 200                  # index space unchanged
    out_of_range = Ref(0)
    for pre in 51:200
        Dewdrop.for_each_post(sub, pre) do post, w, d
            out_of_range[] += 1
        end
    end
    @test out_of_range[] == 0                       # non-source neurons have no out-edges
    in_range = Ref(0)
    for pre in 1:50
        Dewdrop.for_each_post(sub, pre) do post, w, d
            in_range[] += 1
        end
    end
    @test in_range[] > 0                            # source neurons do

    # `targets` restricts which postsynaptic neurons receive edges (for E→I-only projections)
    tgt = Dewdrop.fixed_prob(arch, 200, 200, 0.2; weight = 1.0f0, delay = steps(1), seed = seed, targets = 101:200)
    @test Dewdrop.npost(tgt) == 200                 # index space unchanged
    posts = Int[]
    for pre in 1:200
        Dewdrop.for_each_post(tgt, pre) do post, w, d
            push!(posts, post)
        end
    end
    @test !isempty(posts)
    @test all(in(101:200), posts)                   # every edge lands in the target range

    # `sources` and `targets` compose: a block E(1:100) → I(101:200) projection
    ei_proj = Dewdrop.fixed_prob(arch, 200, 200, 0.2; weight = 1.0f0, delay = steps(1), seed = seed,
        sources = 1:100, targets = 101:200)
    pairs = Tuple{Int, Int}[]
    for pre in 1:200
        Dewdrop.for_each_post(ei_proj, pre) do post, w, d
            push!(pairs, (pre, post))
        end
    end
    @test !isempty(pairs)
    @test all(pr -> pr[1] in 1:100 && pr[2] in 101:200, pairs)
    # expected edge count ≈ p · |sources| · |targets|
    @test 0.85 * 0.2 * 100 * 100 < length(pairs) < 1.15 * 0.2 * 100 * 100
end
