using Dewdrop
using Test
using Adapt
using JLArrays

# The event-driven sparse scatter as a KernelAbstractions kernel: one thread per
# presynaptic neuron, spiking ones walk their CSR row and deposit each synapse's weight into
# the delay ring buffer at (now + per-synapse delay), accumulating atomically. The SAME
# kernel runs on CPU and device arrays via `get_backend`.
@testset "partitioned CSR scatter (KernelAbstractions)" begin
    arch = Dewdrop.CPU()
    edges = [(1, 2, 0.5f0, 1), (1, 3, 0.25f0, 2), (2, 1, 1.0f0, 3), (4, 3, 2.0f0, 1)]
    conn = Dewdrop.SparseCSR(arch, edges; npre = 4, npost = 3)
    buf = Dewdrop.DelayBuffer(arch, Float32, 3, 5)
    spiked = [true, false, false, true]          # neurons 1 and 4 spike; 2 and 3 silent
    L = buf.L
    Dewdrop.scatter!(buf, conn, spiked, 0)

    # the ring holds fixed-point counts, so read it in physical units through `slotvalues`
    sv = Dewdrop.slotvalues(buf)
    @test sv[2, mod(0 + 1, L) + 1] == 0.5f0     # 1→2, delay 1
    @test sv[3, mod(0 + 2, L) + 1] == 0.25f0    # 1→3, delay 2
    @test sv[3, mod(0 + 1, L) + 1] == 2.0f0     # 4→3, delay 1
    @test sv[1, mod(0 + 3, L) + 1] == 0.0f0     # 2→1 NOT deposited (neuron 2 silent)
    @test sum(sv) == 0.5f0 + 0.25f0 + 2.0f0

    # collisions: several presynaptic neurons → same (post, slot) accumulate atomically
    conn2 = Dewdrop.SparseCSR(arch, [(1, 1, 1.0f0, 1), (2, 1, 3.0f0, 1)]; npre = 2, npost = 1)
    buf2 = Dewdrop.DelayBuffer(arch, Float32, 1, 3)
    Dewdrop.scatter!(buf2, conn2, [true, true], 0)
    @test Dewdrop.slotvalues(buf2)[1, mod(0 + 1, buf2.L) + 1] == 4.0f0   # 1.0 + 3.0, atomic add

    # the identical kernel runs on a device array type (JLArray) via get_backend
    gbuf = adapt(JLArray, Dewdrop.DelayBuffer(arch, Float32, 3, 5))
    gconn = adapt(JLArray, conn)
    gspiked = adapt(JLArray, [true, false, false, true])
    Dewdrop.scatter!(gbuf, gconn, gspiked, 0)
    @test sum(Dewdrop.slotvalues(gbuf)) == 0.5f0 + 0.25f0 + 2.0f0
end

# The CPU scatter picks a serial or a threaded walk by how many edges this step deposits: threading
# costs a fixed `@threads` dispatch that only pays on enough work. The ring holds fixed-point counts,
# so integer addition is associative and both branches must leave exactly the same slots.
@testset "the scatter's work gate changes speed, not results" begin
    N, L = 300, 8
    conn = Dewdrop._resolve_delays(
        fixed_prob(Dewdrop.CPU(), N, N, 0.2; weight = 0.5f0, delay = steps(3), seed = UInt64(1)), 0.1f0
    )
    deg = Dewdrop.nedges(conn) ÷ N
    scale = Dewdrop.fixedpoint_scale(Int[], Float32[], 2)
    # the work estimate itself: no spikes is no work, all spiking is every edge
    @test Dewdrop._scatter_work(conn, falses(N)) == 0
    @test Dewdrop._scatter_work(conn, trues(N)) == N * deg
    # a sparse step and a fully spiking one, so both sides of the threshold are covered at any
    # thread count (the sparse case is below it even on one thread)
    for frac in (0.01, 1.0)
        spiked = falses(N)
        spiked[1:round(Int, frac * N)] .= true
        got = Dewdrop.DelayBuffer(Dewdrop.CPU(), Float32, N, L - 1; scale = scale)
        Dewdrop.scatter!(got, conn, spiked, 1)
        # the serial branch, forced, as the reference
        ref = Dewdrop.DelayBuffer(Dewdrop.CPU(), Float32, N, L - 1; scale = scale)
        @inbounds for pre in eachindex(spiked)
            spiked[pre] || continue
            for e in conn.rowptr[pre]:(conn.rowptr[pre + 1] - 1)
                ref.slots[conn.post[e], mod(1 + conn.delay[e], ref.L) + 1] +=
                    Dewdrop._fp_quantise(eltype(ref.slots), conn.weight[e], scale)
            end
        end
        @test got.slots == ref.slots                      # bit-identical either side of the gate
    end
end
