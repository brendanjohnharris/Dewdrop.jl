using Dewdrop
using Test
using Adapt
using JLArrays

# M2 --- multiple projections per population + conductance-based (COBA) synapses. The
# integrate phase accumulates a per-neuron conductance (gtot) and current (itot) from every
# projection, then takes the COBA effective-τ / effective-V∞ exponential-Euler step, so a
# population can carry several synapse types at once (the prerequisite for E/I COBA networks).
@testset "multiple projections + COBA" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    conn = Dewdrop.SparseCSR(Dewdrop.CPU(), [(1, 2, 15.0, 1)]; npre = 2, npost = 2)
    base() = (input = [0.5, 0.0], tspan = (0.0, 300.0))

    # two CUBA projections (fast + slow) onto neuron 2 — contributions sum
    p_fast = Projection(CurrentSynapse(τ = 2.0), conn)
    p_slow = Projection(CurrentSynapse(τ = 10.0), conn)
    sol2 = solve(DewdropNetwork(m, 2; base()..., projections = (p_fast, p_slow)), FixedStep(0.1))
    sol1 = solve(DewdropNetwork(m, 2; base()..., projection = p_fast), FixedStep(0.1))
    @test sol1.spike_count[1] > 0                       # single-projection API still works
    @test sol2.spike_count[1] > 0
    @test sol2.spike_count[2] > sol1.spike_count[2]     # two projections drive neuron 2 more than one

    # COBA excitatory (Erev = 0, above V): conductance current g·(0−V) > 0 drives firing
    coba_e = Projection(ConductanceSynapse(τ = 5.0, Erev = 0.0), conn)
    sole = solve(DewdropNetwork(m, 2; base()..., projection = coba_e), FixedStep(0.1))
    @test sole.spike_count[2] > 0

    # COBA inhibitory (Erev = −80, below rest): hyperpolarizing, cannot drive firing
    coba_i = Projection(ConductanceSynapse(τ = 5.0, Erev = -80.0), conn)
    soli = solve(DewdropNetwork(m, 2; base()..., projection = coba_i), FixedStep(0.1))
    @test soli.spike_count[2] == 0

    # mixed E + I COBA projections on one population, GPU-movable
    prob = DewdropNetwork(m, 2; base()..., projections = (coba_e, coba_i))
    gpu = adapt(JLArray, init(prob, FixedStep(0.1)))
    cpu = init(prob, FixedStep(0.1))
    for _ in 1:200
        step!(gpu); step!(cpu)
    end
    @test Array(gpu.spike_count) == cpu.spike_count
end
