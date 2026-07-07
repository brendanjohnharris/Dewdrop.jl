using Dewdrop
using Test
using Adapt
using JLArrays

# A CONNECTED network: presynaptic spikes drive postsynaptic neurons through the
# synapse. Validated on a 2-neuron chain: neuron 1 is driven supra-threshold; neuron 2 has
# NO external input and fires only via synaptic transmission from neuron 1, after the
# conduction delay. This exercises the full chain scatter → ring buffer → deliver → synaptic
# current → integrate.
@testset "connected network: synaptic transmission" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    dt, tend = 0.1, 300.0
    Iext = [0.5, 0.0]                                  # neuron 1 driven; neuron 2 silent alone
    syn = CurrentSynapse(τ = 5.0)
    delay = 15                                                 # 15 steps = 1.5 ms at dt = 0.1 ms (raw edge = steps)
    conn = Dewdrop.SparseCSR(Dewdrop.CPU(), [(1, 2, 40.0, delay)]; npre = 2, npost = 2)
    proj = Dewdrop.Projection(syn, conn)

    prob = DewdropNetwork(m, 2; input = Iext, tspan = (0.0, tend), projection = proj)
    sol = solve(prob, FixedStep(dt))
    @test sol.spike_count[1] > 0                        # neuron 1 fires (externally driven)
    @test sol.spike_count[2] > 0                        # neuron 2 fires (synaptically driven)

    # control: with no projection, neuron 2 (no external input) is silent
    sol0 = solve(DewdropNetwork(m, 2; input = Iext, tspan = (0.0, tend)), FixedStep(dt))
    @test sol0.spike_count[1] > 0
    @test sol0.spike_count[2] == 0

    # the conduction delay: neuron 2's synaptic current is zero right after neuron 1's first
    # spike, and becomes positive only after the delay elapses.
    integ = init(prob, FixedStep(dt))
    while integ.spike_count[1] == 0
        step!(integ)
    end
    @test integ.syns[1].acc.Isyn[2] == 0.0                 # not yet delivered (delayed)
    for _ in 1:(delay + 5)
        step!(integ)
    end
    @test integ.syns[1].acc.Isyn[2] > 0.0                  # delivered after the conduction delay

    # GPU-readiness: the connected step runs under JLArray + allowscalar(false)
    gpu = adapt(JLArray, init(prob, FixedStep(dt)))
    @test gpu.syns[1].acc.Isyn isa JLArray
    cpu = init(prob, FixedStep(dt))
    for _ in 1:60
        step!(gpu); step!(cpu)
    end
    @test Array(gpu.spike_count) == cpu.spike_count
end
