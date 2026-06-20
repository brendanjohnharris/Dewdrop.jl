using Dewdrop
using Test
using Adapt
using JLArrays

# M1b --- per-synapse heterogeneous conduction delays via a NEST-style ring buffer of
# postsynaptic accumulators. A spike scattered at step `now` along a synapse of integer
# delay `d` lands in ring slot (now+d) mod L for its target, and is delivered when the
# clock reaches that step. Sized L = maxdelay + 1.
@testset "delay ring buffer (heterogeneous per-synapse delays)" begin
    arch = Dewdrop.CPU()
    N, maxd = 3, 5
    buf = Dewdrop.DelayBuffer(arch, Float64, N, maxd)
    @test size(buf.slots) == (N, maxd + 1)
    @test all(iszero, buf.slots)
    @test Dewdrop.maxdelay(buf) == maxd

    # at step 0, deposit distinct per-synapse delays onto distinct targets
    Dewdrop.deposit!(buf, 0, 1, 0.5, 2)       # neuron 1, value 0.5, delay 2 steps
    Dewdrop.deposit!(buf, 0, 2, 1.0, 5)       # neuron 2, value 1.0, delay 5 steps
    Dewdrop.deposit!(buf, 0, 1, 0.25, 2)      # accumulates onto neuron 1 at delay 2

    delivered = [zeros(N) for _ in 0:6]
    for t in 0:6
        delivered[t + 1] .= Dewdrop.collect_due!(buf, t)
    end
    @test delivered[2 + 1][1] == 0.75         # 0.5 + 0.25 arrive at step 0+2
    @test delivered[5 + 1][2] == 1.0          # arrives at step 0+5
    @test sum(sum, delivered) == 0.75 + 1.0   # nothing spurious or mistimed
    @test all(iszero, buf.slots)              # slots zeroed after collection (reusable)

    # delay 0 delivers at the same step
    Dewdrop.deposit!(buf, 10, 3, 2.0, 0)
    @test Dewdrop.collect_due!(buf, 10)[3] == 2.0

    # Adapt-movable (the scatter into the buffer becomes a device kernel later)
    gbuf = adapt(JLArray, buf)
    @test gbuf.slots isa JLArray
    @test Dewdrop.maxdelay(gbuf) == maxd
end
