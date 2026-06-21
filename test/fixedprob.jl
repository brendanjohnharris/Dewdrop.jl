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
    conn = Dewdrop.fixed_prob(arch, npre, npost, p; weight = 0.5f0, delay = 2, seed = seed)

    @test conn isa Dewdrop.SparseCSR
    @test Dewdrop.npre(conn) == npre
    @test Dewdrop.npost(conn) == npost
    # expected edge count ≈ p · npre · npost
    expected = p * npre * npost
    @test 0.9 * expected < Dewdrop.nedges(conn) < 1.1 * expected
    @test all(==(0.5f0), conn.weight)
    @test all(==(2), conn.delay)

    # reproducible: same seed → identical realised connectome
    conn2 = Dewdrop.fixed_prob(arch, npre, npost, p; weight = 0.5f0, delay = 2, seed = seed)
    @test conn.post == conn2.post
    @test conn.rowptr == conn2.rowptr
    # a different seed gives a different connectome
    conn3 = Dewdrop.fixed_prob(arch, npre, npost, p; weight = 0.5f0, delay = 2, seed = UInt64(43))
    @test conn3.post != conn2.post

    # no self-connections when disallowed
    rec = Dewdrop.fixed_prob(arch, 100, 100, 0.2; weight = 1.0f0, delay = 1, seed = seed, allow_self = false)
    self = 0
    for pre in 1:100
        Dewdrop.for_each_post(rec, pre) do post, w, d
            post == pre && (self += 1)
        end
    end
    @test self == 0

    # per-source signed weights (excitatory neurons +, inhibitory −)
    NE = 100
    ei = Dewdrop.fixed_prob(arch, 150, 150, 0.1; weight = pre -> pre ≤ NE ? 1.0f0 : -4.0f0, delay = 1, seed = seed)
    @test any(>(0), ei.weight)
    @test any(<(0), ei.weight)

    # `sources` restricts which presynaptic neurons emit edges (for E/I subpopulations)
    sub = Dewdrop.fixed_prob(arch, 200, 150, 0.2; weight = 1.0f0, delay = 1, seed = seed, sources = 1:50)
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
end
