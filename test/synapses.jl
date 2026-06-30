using Dewdrop
using Test

# Current-based (CUBA) exponential synapse: a delivered spike of weight `w` adds
# `w` to the postsynaptic synaptic current, which decays with time constant `τ` and feeds
# the neuron's input current directly.
@testset "current synapse (CUBA): decay + PSC" begin
    syn = CurrentSynapse(τ = 5.0)
    dt = 0.1
    decay = Dewdrop.synapse_decay(syn, dt)
    @test decay ≈ exp(-dt / 5.0)

    # single-synapse post-synaptic current (PSC): one presynaptic spike delivers weight w at
    # the delayed step into the synaptic current, which then decays exponentially. Exercises
    # the ring buffer + synapse decay together.
    arch = Dewdrop.CPU()
    buf = Dewdrop.DelayBuffer(arch, Float64, 1, 20)
    Dewdrop.deposit!(buf, 0, 1, 1.0, 10)         # weight 1.0, delay 10 steps
    Isyn = 0.0
    trace = Float64[]
    for t in 0:200
        Isyn *= decay
        Isyn += Dewdrop.collect_due!(buf, t)[1]
        push!(trace, Isyn)
    end
    @test all(iszero, trace[1:10])               # silent before the delayed arrival
    @test trace[11] ≈ 1.0                        # instantaneous jump on arrival (step 10)
    for k in 1:50
        @test trace[11 + k] ≈ exp(-k * dt / 5.0) # exact exponential decay afterwards
    end
end
