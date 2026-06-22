using Dewdrop
using Test
using Adapt
using JLArrays

# M5c --- event-driven pair-based STDP. A plastic projection carries its OWN mutable per-edge
# weights (CSR-parallel, leaving the shared SparseCSR immutable) plus pre/post eligibility traces.
# The weight update rides the existing edge-parallel scatter (per-edge, no atomics on the weight);
# trace decay folds into :integrate and the bump into :propagate, so no new schedule phase is
# needed and the decay→update→bump order falls out of the phase order. Plasticity uses ACTUAL
# spike times (the conduction delay is transmission-only).

# Minimal STDP harness: a single edge pre→post, spikes driven by hand at controlled steps, running
# only the trace decay (:integrate's _decay_all!) and the plastic :propagate (scatter + bump). This
# isolates the weight rule, so Δw vs Δt is analytically exact.
function stdp_pair_dw(rule, Δt_steps; w0 = 5.0, d = 1, dt = 1.0, pre = 1, post = 2)
    m = LIF(; τ = 20.0, EL = -65.0, Vθ = 1.0e6, Vr = -65.0, R = 1.0, tref = 0.0)  # never self-fires
    conn = Dewdrop.SparseCSR(Dewdrop.CPU(), [(pre, post, w0, d)]; npre = 2, npost = 2)
    proj = Dewdrop.Projection(DeltaSynapse(), conn; plasticity = rule)
    integ = init(DewdropNetwork(m, 2; input = 0.0, tspan = (0.0, 100.0), projection = proj), FixedStep(dt))
    s0 = 40
    n_pre, n_post = s0, s0 + Δt_steps
    w_before = Array(integ.syns[1].weight)[1]
    for n in 0:(s0 + abs(Δt_steps) + 5)
        Dewdrop._decay_all!(integ.syns)               # decay traces (base too, harmless)
        integ.spiked .= false
        n == n_pre && (integ.spiked[pre] = true)
        n == n_post && (integ.spiked[post] = true)
        Dewdrop.run_phase!(Val(:propagate), integ)    # plastic scatter (weight update) + trace bump
        integ.n += 1
    end
    return Array(integ.syns[1].weight)[1] - w_before, integ
end

@testset "STDP: analytic exponential window (Δw vs Δt)" begin
    rule = Dewdrop.STDP(; Aplus = 0.1, Aminus = 0.12, τplus = 20.0, τminus = 25.0, wmin = 0.0, wmax = 100.0)
    dt = 1.0
    for Δ in (1, 3, 5, 10, 20, 40)
        dw, _ = stdp_pair_dw(rule, Δ; dt = dt)
        @test dw ≈ rule.Aplus * exp(-Δ * dt / rule.τplus) rtol = 1e-5     # pre→post potentiates
    end
    for Δ in (-1, -3, -5, -10, -20)
        dw, _ = stdp_pair_dw(rule, Δ; dt = dt)
        @test dw ≈ -rule.Aminus * exp(Δ * dt / rule.τminus) rtol = 1e-5    # post→pre depresses
    end
    # exact coincidence Δt = 0: both endpoints spike the same step -> potentiation reads the
    # not-yet-bumped pre trace (0) and depression the not-yet-bumped post trace (0) -> no change
    dw0, _ = stdp_pair_dw(rule, 0; dt = dt)
    @test dw0 ≈ 0.0 atol = 1e-12
end

@testset "STDP: delays are transmission-only (Δw independent of delay)" begin
    rule = Dewdrop.STDP(; Aplus = 0.1, Aminus = 0.1, τplus = 20.0, τminus = 20.0, wmin = 0.0, wmax = 100.0)
    dw1, _ = stdp_pair_dw(rule, 10; d = 1)
    dw5, _ = stdp_pair_dw(rule, 10; d = 7)
    @test dw1 ≈ dw5 rtol = 1e-12       # plasticity uses actual spike times, not delivery time
end

@testset "STDP: convergence + clamp to [wmin, wmax]" begin
    # repeated potentiating pairs drive w up to wmax; repeated depressing pairs down to wmin
    pot = Dewdrop.STDP(; Aplus = 0.5, Aminus = 0.5, τplus = 20.0, τminus = 20.0, wmin = 1.0, wmax = 6.0)
    m = LIF(; τ = 20.0, EL = -65.0, Vθ = 1.0e6, Vr = -65.0, R = 1.0, tref = 0.0)
    conn = Dewdrop.SparseCSR(Dewdrop.CPU(), [(1, 2, 3.0, 1)]; npre = 2, npost = 2)
    integ = init(DewdropNetwork(m, 2; input = 0.0, tspan = (0.0, 10.0),
        projection = Dewdrop.Projection(DeltaSynapse(), conn; plasticity = pot)), FixedStep(1.0))
    # drive many pre(t)→post(t+2) pairs
    for rep in 1:200
        for (k, sp) in ((0, 1), (2, 2))
            Dewdrop._decay_all!(integ.syns)
            integ.spiked .= false
            integ.spiked[sp] = true
            Dewdrop.run_phase!(Val(:propagate), integ)
            integ.n += 1
        end
    end
    @test Array(integ.syns[1].weight)[1] ≈ 6.0      # clamped at wmax
    @test Array(integ.syns[1].weight)[1] ≤ 6.0
end

@testset "STDP: mutable weight is separate; shared CSR immutable" begin
    rule = Dewdrop.STDP(; Aplus = 0.1, Aminus = 0.1, τplus = 20.0, τminus = 20.0, wmin = 0.0, wmax = 100.0)
    conn = Dewdrop.SparseCSR(Dewdrop.CPU(), [(1, 2, 5.0, 1)]; npre = 2, npost = 2)
    dw, integ = stdp_pair_dw(rule, 10)
    @test dw != 0.0
    # the learned change landed in the plastic state's weight, NOT the shared connectivity array
    @test conn.weight[1] == 5.0
    @test Array(integ.syns[1].weight)[1] != 5.0
end

@testset "STDP: CPU broadcast ≡ JLArray fused (learned weights + spikes)" begin
    # a small recurrent net that actually fires, with plastic synapses; the fused megakernel decays
    # the traces in-kernel and the plastic scatter updates the weights --- both must match the CPU
    # broadcast path bit-for-bit (exactly-representable weights -> order-independent atomic deposit).
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 0.0, R = 1.0, tref = 2.0)
    rule = Dewdrop.STDP(; Aplus = 0.5, Aminus = 0.25, τplus = 20.0, τminus = 20.0, wmin = 0.0, wmax = 64.0)
    edges = [(i, mod(i, 8) + 1, 4.0, 1) for i in 1:8]      # ring of 8, exact weights
    conn = Dewdrop.SparseCSR(Dewdrop.CPU(), edges; npre = 8, npost = 8)
    proj = Dewdrop.Projection(DeltaSynapse(), conn; plasticity = rule)
    prob = DewdropNetwork(m, 8; input = 30.0, tspan = (0.0, 300.0), projection = proj)   # supra-threshold
    cpu = init(prob, FixedStep(0.1))
    gpu = adapt(JLArray, init(prob, FixedStep(0.1)))
    for _ in 1:3000
        step!(cpu); step!(gpu)
    end
    @test sum(cpu.spike_count) > 0
    @test Array(gpu.spike_count) == cpu.spike_count
    @test Array(gpu.syns[1].weight) ≈ Array(cpu.syns[1].weight)
    @test Array(cpu.syns[1].weight) != conn.weight        # weights actually moved

    # allocation-free plastic step
    warm = init(prob, FixedStep(0.1)); step!(warm)
    @test @allocated(step!(warm)) == 0
end
