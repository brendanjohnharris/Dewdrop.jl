using Dewdrop
using Test
using Adapt
using JLArrays
using Statistics

# WRCircuit Phase 2 --- dual-exponential COBA synapse (`DualExpSynapse`; the WRCircuit
# `bp.dyn.DualExponV2` kinetic). Two per-target accumulators `g_rise`/`g_decay`, both kicked by a
# delivered weight, decay with `exp(-dt/τr)` / `exp(-dt/τd)`; the conductance is
# `g(t) = a·(g_decay − g_rise)` with `a = (τd/(τd−τr))·(τr/τd)^(τr/(τr−τd))` (peak normalised to the
# kick). COBA output `g·(Erev − V)`. A difference-of-exponentials PSG (rise then decay), vs the
# single-exponential `ConductanceSynapse`.

dt = 0.1

@testset "dual-exp normalisation coefficient" begin
    τr, τd = 1.0, 5.0
    a = Dewdrop._dualexp_a(τr, τd)
    @test a ≈ (τd / (τd - τr)) * (τr / τd)^(τr / (τr - τd))
    # the continuous peak of a·(e^{-t/τd} − e^{-t/τr}) is exactly 1 (so g_max is the peak conductance)
    tpeak = (τd * τr / (τd - τr)) * log(τd / τr)
    @test a * (exp(-tpeak / τd) - exp(-tpeak / τr)) ≈ 1.0
    @test_throws Exception DualExpSynapse(; τr = 3.0, τd = 3.0, Erev = 0.0)   # τr == τd is singular
end

@testset "dual-exp PSG kinetics vs analytic difference-of-exponentials" begin
    syn = DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    conn = fixed_prob(Dewdrop.CPU(), 1, 1, 1.0; weight = 1.0, delay = steps(1), seed = UInt64(1))
    st = Dewdrop._make_synstate(Dewdrop.CPU(), syn, conn, Float64, 1, dt)
    w = 2.0
    st.g_rise[1] += w; st.g_decay[1] += w           # one delivered spike of weight w
    a = st.a
    g(k) = a * (st.g_decay[1] - st.g_rise[1])
    gs = Float64[]
    for _ in 0:300
        push!(gs, a * (st.g_decay[1] - st.g_rise[1]))
        st.g_rise[1] *= st.decay_r
        st.g_decay[1] *= st.decay_d
    end
    analytic = [w * a * (exp(-k * dt / 5.0) - exp(-k * dt / 1.0)) for k in 0:300]
    @test gs ≈ analytic
    @test gs[1] == 0.0                              # zero at the delivery instant
    @test gs[5] > gs[1]                             # rises first (dual-exp shape)
    @test maximum(gs) ≈ w rtol = 1.0e-3             # peak normalised to the kick weight
    @test argmax(gs) > 1                            # peak is delayed (not at delivery)
end

@testset "dual-exp COBA transmission: post depolarised toward Erev" begin
    # neuron 1 driven supra-threshold (fires); neuron 2 receives an excitatory (Erev=0) dual-exp
    # conductance via a single 1→2 edge and depolarises above its rest.
    m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
    edge = fixed_prob(Dewdrop.CPU(), 2, 2, 1.0; weight = 5.0, delay = steps(1), seed = UInt64(1),
        sources = 1:1, targets = 2:2)
    prob = DewdropNetwork(m, 2; input = [20.0, 0.0], tspan = (0.0, 300.0),
        projection = Projection(DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0), edge))
    # record the dual-exp conductance from the SYNAPSE STATE (`g_decay`, the raw conductance) and V.
    sol = solve(prob, FixedStep(dt); record = (V = Trace(:V), g = Trace(:g_decay; projection = 1)))
    @test sum(sol.spike_count) > 0                  # neuron 1 fires, drives neuron 2
    @test maximum(sol.record.g.data[2, :]) > 0      # dual-exp conductance delivered to neuron 2
    @test mean(sol.record.V.data[2, :]) > m.EL      # neuron 2 depolarised above rest by excitation

    # the `:gtot` membrane accumulator is materialised on every SCALAR backend --- the fused step writes
    # it per-neuron in-kernel (Fused.jl), matching the Serial broadcast path --- so recording it works and
    # agrees across backends. (Only the (N,B) batched path leaves it kernel-local; rejected there, see batch.jl.)
    solF = solve(prob, FixedStep(dt); backend = Fused(), record = (g = Trace(:gtot),))
    solS = solve(prob, FixedStep(dt); backend = Serial(), record = (g = Trace(:gtot),))
    @test maximum(solF.record.g.data[2, :]) > 0     # :gtot materialised on the fused path (not zeros)
    @test solF.record.g.data ≈ solS.record.g.data   # fused ≡ Serial accumulator
end

@testset "dual-exp COBA: CPU broadcast ≡ JLArray fused" begin
    m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
    conn = fixed_prob(Dewdrop.CPU(), 64, 64, 0.1; weight = 1.5, delay = steps(2), seed = UInt64(3))
    prob = DewdropNetwork(m, 64; input = 18.0, tspan = (0.0, 200.0),
        projection = Projection(DualExpSynapse(; τr = 1.0, τd = 6.0, Erev = 0.0), conn))
    cpu = init(prob, FixedStep(dt))
    gpu = adapt(JLArray, init(prob, FixedStep(dt)))
    for _ in 1:2000
        step!(cpu); step!(gpu)
    end
    @test sum(cpu.spike_count) > 0
    @test Array(gpu.spike_count) == cpu.spike_count
    @test Array(gpu.state.state.V) ≈ cpu.state.state.V
end

@testset "dual-exp COBA: batched ≡ scalar oracle" begin
    m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
    conn = fixed_prob(Dewdrop.CPU(), 80, 80, 0.1; weight = 1.5, delay = steps(2), seed = UInt64(5))
    prob = DewdropNetwork(m, 80; input = 18.0, tspan = (0.0, 150.0),
        projection = Projection(DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0), conn))
    B = 4
    bsol = solve(prob, FixedStep(dt); batch = B)             # no drive/noise → columns identical
    ssol = solve(prob, FixedStep(dt))
    @test sum(ssol.spike_count) > 0
    for b in 1:B
        @test bsol.spike_count[:, b] == ssol.spike_count
    end
end
