using Dewdrop
using Test
using Adapt
using JLArrays

# `FNSNeuron`, a conductance-adaptation LIF (Treves-style).
# `C dV/dt = -gL(V-VL) - gK(V-VK) + I`, `τK dgK/dt = -gK`; spike (`V ≥ Vθ`) → `V ← Vr`,
# `gK += ΔgK`. The adaptation is a CONDUCTANCE `gK` with reversal `VK` (not a current like AdaptLIF's
# `w`), so it folds onto the same exact COBA propagator: `gK` adds to the total conductance and a
# reversal drive `gK·VK`. E neurons adapt (`ΔgK > 0`); I neurons have `ΔgK = 0` (a plain
# conductance-LIF). Reuses the multi-state seam (the generic aux column `:w` holds `gK`).

# exact engine recursion for one FNS neuron, no synapses/drive (matches the broadcast phases:
# w-first split, refractory V-clamp, threshold after refrac decrement, reset + gK increment).
_coba(V, EL, R, τ, gtot, itot, dt) = (denom = 1 + R * gtot; V∞ = (EL + R * itot) / denom; V∞ + (V - V∞) * exp(-dt * denom / τ))

@testset "FNS: hooks + metadata" begin
    m = FNSNeuron(;
        C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0,
        tref = 4.0, τK = 80.0, ΔgK = 0.01
    )
    @test Dewdrop.statevars(m) == (:V, :refrac, :w)         # aux column :w holds gK
    @test Dewdrop.float_type(m) == Float64
    @test Dewdrop._resting(m) == -70.0                     # rests at the leak reversal VL
    @test Dewdrop.reset_value(m) == -60.0
    @test Dewdrop.refractory(m) == 4.0
    @test Dewdrop.threshold(m, -49.0) == true && Dewdrop.threshold(m, -51.0) == false
    @test Dewdrop._has_w(typeof(m)) == true
    # gK decays toward 0 with τK; the membrane folds gK into the COBA propagator
    @test Dewdrop._step_w(m, -65.0, 0.02, 0.1) ≈ 0.02 * exp(-0.1 / m.τK)
    @test Dewdrop._step_V(m, -65.0, 0.02, 0.0, 0.3, 0.1) ≈
        _coba(-65.0, m.VL, inv(m.gL), m.C / m.gL, 0.02, 0.3 + 0.02 * m.VK, 0.1)
end

@testset "FNS: subthreshold fixed point (gK decays to 0 → conductance-LIF)" begin
    # no spikes (I below rheobase gL·(Vθ−VL)); gK starts and stays 0 → V* = VL + I/gL
    m = FNSNeuron(;
        C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0,
        tref = 4.0, τK = 80.0, ΔgK = 0.01
    )
    I = 0.2                                                # I_rh = gL·(Vθ−VL) = 0.334 > 0.2
    sol = solve(DewdropNetwork(m, 1; input = I, tspan = (0.0, 600.0)), FixedStep(0.1))
    @test only(sol.spike_count) == 0
    @test only(sol.state.state.V) ≈ m.VL + I / m.gL rtol = 1.0e-3
    @test only(sol.state.state.w) ≈ 0.0 atol = 1.0e-9       # gK relaxes to 0
end

@testset "FNS: exact (V, gK) recursion vs the engine (machine precision, spiking)" begin
    m = FNSNeuron(;
        C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0,
        tref = 4.0, τK = 80.0, ΔgK = 0.01
    )
    I = 0.6; dt = 0.1; tend = 250.0                       # supra-threshold (V∞|gK=0 = -34 > Vθ)
    sol = solve(
        DewdropNetwork(m, 1; input = I, tspan = (0.0, tend)), FixedStep(dt);
        record = (V = Trace(:V), w = Trace(:w))
    )
    V, w, refr = m.VL, 0.0, 0.0
    nsteps = size(sol.record.V.data, 2)
    nspk = 0
    for n in 1:nsteps
        w *= exp(-dt / m.τK)                              # gK decays (w-first)
        V = refr > 0 ? m.Vr : _coba(V, m.VL, inv(m.gL), m.C / m.gL, w, I + w * m.VK, dt)
        refr = max(refr - dt, 0.0)
        spiked = (refr ≤ 0) && (V ≥ m.Vθ)
        if spiked
            V = m.Vr; refr = m.tref; w += m.ΔgK; nspk += 1
        end
        @test sol.record.V.data[1, n] ≈ V
        @test sol.record.w.data[1, n] ≈ w
    end
    @test nspk ≥ 4                                        # actually spiking (so gK is exercised)
    @test only(sol.spike_count) == nspk
end

@testset "FNS: conductance adaptation (E adapts; ΔgK=0 I does not)" begin
    base = (; C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0, tref = 4.0, τK = 80.0)
    dt, tend, I = 0.1, 1500.0, 0.6
    # E: ΔgK > 0 → gK accumulates, hyperpolarising toward VK → ISIs lengthen (adaptation)
    mE = FNSNeuron(; base..., ΔgK = 0.02)
    sE = solve(DewdropNetwork(mE, 1; input = I, tspan = (0.0, tend)), FixedStep(dt); record = (spk = Spikes(),))
    tE, _ = raster(sE; name = :spk)
    @test length(tE) ≥ 5
    isis = diff(tE)
    @test isis[end] > isis[1]                             # later intervals longer
    # I: ΔgK = 0 → no adaptation → more spikes than the adapting E neuron at the same drive
    mI = FNSNeuron(; base..., ΔgK = 0.0)
    sI = solve(DewdropNetwork(mI, 1; input = I, tspan = (0.0, tend)), FixedStep(dt))
    @test only(sI.spike_count) > only(sE.spike_count)
end

@testset "FNS + Heterogeneous: E adapts, I doesn't (the E/I pattern)" begin
    # one concatenated FNS population, per-neuron ΔgK: E (1:NE) adapts, I (NE+1:N) does not
    base = FNSNeuron(;
        C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0,
        tref = 4.0, τK = 80.0, ΔgK = 0.0
    )
    NE, NI = 40, 40
    ΔgK = vcat(fill(0.02, NE), fill(0.0, NI))             # E adapts, I plain conductance-LIF
    hm = Heterogeneous(base; ΔgK = ΔgK)
    sol = solve(DewdropNetwork(hm, NE + NI; input = 0.6, tspan = (0.0, 1500.0)), FixedStep(0.1))
    @test sum(sol.spike_count[1:NE]) > 0
    @test sum(sol.spike_count[1:NE]) < sum(sol.spike_count[(NE + 1):end])   # E adaptation suppresses E
end

@testset "FNS: CPU broadcast ≡ JLArray fused; allocation-free" begin
    m = FNSNeuron(;
        C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0,
        tref = 4.0, τK = 80.0, ΔgK = 0.015
    )
    dt = 0.1
    prob = DewdropNetwork(m, 48; input = 0.7, tspan = (0.0, 300.0))
    cpu = init(prob, FixedStep(dt))
    gpu = adapt(JLArray, init(prob, FixedStep(dt)))
    for _ in 1:3000
        step!(cpu); step!(gpu)
    end
    @test sum(cpu.spike_count) > 0
    @test Array(gpu.spike_count) == cpu.spike_count
    @test Array(gpu.state.state.V) ≈ cpu.state.state.V
    @test Array(gpu.state.state.w) ≈ cpu.state.state.w
    warm = init(prob, FixedStep(dt)); step!(warm)
    @test @allocated(step!(warm)) == 0
end
