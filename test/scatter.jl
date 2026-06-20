using Dewdrop
using Test
using Adapt
using JLArrays

# M1c --- the event-driven sparse scatter as a KernelAbstractions kernel: one thread per
# presynaptic neuron, spiking ones walk their CSR row and deposit each synapse's weight into
# the delay ring buffer at (now + per-synapse delay), accumulating atomically. The SAME
# kernel runs on CPU and device arrays via `get_backend`.
@testset "partitioned CSR scatter (KernelAbstractions)" begin
    arch = Dewdrop.CPU()
    edges = [(1, 2, 0.5f0, 1), (1, 3, 0.25f0, 2), (2, 1, 1.0f0, 0), (4, 3, 2.0f0, 1)]
    conn = Dewdrop.SparseCSR(arch, edges; npre = 4, npost = 3)
    buf = Dewdrop.DelayBuffer(arch, Float32, 3, 5)
    spiked = [true, false, false, true]          # neurons 1 and 4 spike; 2 and 3 silent
    L = buf.L
    Dewdrop.scatter!(buf, conn, spiked, 0)

    @test buf.slots[2, mod(0 + 1, L) + 1] == 0.5f0     # 1→2, delay 1
    @test buf.slots[3, mod(0 + 2, L) + 1] == 0.25f0    # 1→3, delay 2
    @test buf.slots[3, mod(0 + 1, L) + 1] == 2.0f0     # 4→3, delay 1
    @test buf.slots[1, mod(0 + 0, L) + 1] == 0.0f0     # 2→1 NOT deposited (neuron 2 silent)
    @test sum(buf.slots) == 0.5f0 + 0.25f0 + 2.0f0

    # collisions: several presynaptic neurons → same (post, slot) accumulate atomically
    conn2 = Dewdrop.SparseCSR(arch, [(1, 1, 1.0f0, 1), (2, 1, 3.0f0, 1)]; npre = 2, npost = 1)
    buf2 = Dewdrop.DelayBuffer(arch, Float32, 1, 3)
    Dewdrop.scatter!(buf2, conn2, [true, true], 0)
    @test buf2.slots[1, mod(0 + 1, buf2.L) + 1] == 4.0f0   # 1.0 + 3.0, atomic add

    # the identical kernel runs on a device array type (JLArray) via get_backend
    gbuf = adapt(JLArray, Dewdrop.DelayBuffer(arch, Float32, 3, 5))
    gconn = adapt(JLArray, conn)
    gspiked = adapt(JLArray, [true, false, false, true])
    Dewdrop.scatter!(gbuf, gconn, gspiked, 0)
    @test sum(Array(gbuf.slots)) == 0.5f0 + 0.25f0 + 2.0f0
end
