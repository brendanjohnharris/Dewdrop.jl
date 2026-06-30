using Dewdrop
using Test
using Adapt
using JLArrays
using Statistics

# FrozenDualExpSynapse --- frozen-current variant of `DualExpSynapse`. IDENTICAL dual-exponential
# conductance kinetics (`g(t) = a·(g_decay − g_rise)`, same `a`/`decay_r`/`decay_d`), but the synaptic
# current `g·(Erev − V)` is injected with `V` FROZEN at its pre-update value and does NOT enter the
# effective leak (`gtot`) --- so it does not shunt the membrane time constant. Reproduces the BrainPy
# `sum_current_inputs`/`COBA` integration. Only the accumulate step differs from exact COBA, so the test
# focus is: (1) the accumulate semantics, (2) it diverges from exact, (3) every backend agrees.

dt = 0.1

@testset "frozen dual-exp: same kinetics, constructor guard" begin
    fr = FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    ex = DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    @test (fr.τr, fr.τd, fr.Erev) == (ex.τr, ex.τd, ex.Erev)
    conn = fixed_prob(Dewdrop.CPU(), 1, 1, 1.0; weight = 1.0, delay = steps(1), seed = UInt64(1))
    sf = Dewdrop._make_synstate(Dewdrop.CPU(), fr, conn, Float64, 1, dt)
    se = Dewdrop._make_synstate(Dewdrop.CPU(), ex, conn, Float64, 1, dt)
    @test (sf.a, sf.decay_r, sf.decay_d) == (se.a, se.decay_r, se.decay_d)   # identical PSG kinetics
    @test_throws Exception FrozenDualExpSynapse(; τr = 3.0, τd = 3.0, Erev = 0.0)   # τr == τd is singular
end

@testset "frozen dual-exp: accumulate is frozen current, no shunt" begin
    fr = FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    ex = DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    conn = fixed_prob(Dewdrop.CPU(), 1, 1, 1.0; weight = 1.0, delay = steps(1), seed = UInt64(1))
    sf = Dewdrop._make_synstate(Dewdrop.CPU(), fr, conn, Float64, 1, dt)
    se = Dewdrop._make_synstate(Dewdrop.CPU(), ex, conn, Float64, 1, dt)
    w = 2.0
    for s in (sf, se)                                 # one delivered spike, then let the PSG rise
        s.g_rise[1] += w; s.g_decay[1] += w
        for _ in 1:10; s.g_rise[1] *= s.decay_r; s.g_decay[1] *= s.decay_d; end
    end
    g = sf.a * (sf.g_decay[1] - sf.g_rise[1])
    @test g > 0 && g ≈ se.a * (se.g_decay[1] - se.g_rise[1])   # identical conductance from identical kinetics
    V = [-60.0]
    ge, ie = [0.0], [0.0]; Dewdrop._accumulate!(se, ge, ie, V)   # EXACT: conductance shunts
    @test ge[1] ≈ g && ie[1] ≈ g * se.Erev
    gf, iff = [0.0], [0.0]; Dewdrop._accumulate!(sf, gf, iff, V)  # FROZEN: current g·(Erev − V), gtot untouched
    @test gf[1] == 0.0
    @test iff[1] ≈ g * (fr.Erev - V[1])
end

@testset "frozen dual-exp ≠ exact: different post dynamics" begin
    m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
    mkedge() = fixed_prob(Dewdrop.CPU(), 2, 2, 1.0; weight = 5.0, delay = steps(1), seed = UInt64(1),
        sources = 1:1, targets = 2:2)
    solf = solve(DewdropNetwork(m, 2; input = [20.0, 0.0], tspan = (0.0, 300.0),
            projection = Projection(FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0), mkedge())),
        FixedStep(dt); record = (V = Trace(:V),))
    sole = solve(DewdropNetwork(m, 2; input = [20.0, 0.0], tspan = (0.0, 300.0),
            projection = Projection(DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0), mkedge())),
        FixedStep(dt); record = (V = Trace(:V),))
    @test mean(solf.record.V.data[2, :]) > m.EL      # frozen excitation (Erev=0) still depolarises post
    @test mean(sole.record.V.data[2, :]) > m.EL
    @test !(solf.record.V.data[2, :] ≈ sole.record.V.data[2, :])   # but the two schemes diverge
end

@testset "frozen dual-exp: CPU broadcast ≡ JLArray fused" begin
    m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
    conn = fixed_prob(Dewdrop.CPU(), 64, 64, 0.1; weight = 1.5, delay = steps(2), seed = UInt64(3))
    prob = DewdropNetwork(m, 64; input = 18.0, tspan = (0.0, 200.0),
        projection = Projection(FrozenDualExpSynapse(; τr = 1.0, τd = 6.0, Erev = 0.0), conn))
    cpu = init(prob, FixedStep(dt))
    gpu = adapt(JLArray, init(prob, FixedStep(dt)))
    for _ in 1:2000
        step!(cpu); step!(gpu)
    end
    @test sum(cpu.spike_count) > 0
    @test Array(gpu.spike_count) == cpu.spike_count
    @test Array(gpu.state.state.V) ≈ cpu.state.state.V
end

@testset "frozen dual-exp: batched ≡ scalar oracle" begin
    m = LIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
    conn = fixed_prob(Dewdrop.CPU(), 80, 80, 0.1; weight = 1.5, delay = steps(2), seed = UInt64(5))
    prob = DewdropNetwork(m, 80; input = 18.0, tspan = (0.0, 150.0),
        projection = Projection(FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0), conn))
    B = 4
    bsol = solve(prob, FixedStep(dt); batch = B)             # no drive/noise → columns identical
    ssol = solve(prob, FixedStep(dt))
    @test sum(ssol.spike_count) > 0
    for b in 1:B
        @test bsol.spike_count[:, b] == ssol.spike_count
    end
end
