using Dewdrop
using Test
using Adapt
using JLArrays

# Adaptation neurons. `AdaptLIF` (linear adaptation current `w`) and `AdEx` (nonlinear
# exponential spike initiation) both add a `w` state variable advanced WITH `V` (the "w-first"
# split: w from the old V, then V from the new w) plus a spike-triggered increment `w += b`. LIF
# and every `@neuron` model keep the V-only fast path bit-identical (the empty-aux dispatch).

# --- AdEx fine-dt reference (forward Euler on the AdEx ODE) for an independent dynamics check ---
function adex_reference_spikes(m, I, dt, tend)
    V = m.EL; w = 0.0; refr = 0.0; spikes = Float64[]
    for k in 1:round(Int, tend / dt)
        if refr > 0
            refr -= dt
        else
            V += dt * (-m.gL * (V - m.EL) + m.gL * m.ΔT * exp((V - m.VT) / m.ΔT) + I - w) / m.C
        end
        w += dt * (m.a * (V - m.EL) - w) / m.τw
        if V ≥ m.Vpeak
            push!(spikes, k * dt); V = m.Vr; w += m.b; refr = m.tref
        end
    end
    return spikes
end

@testset "AdaptLIF: hooks + exact (V,w) recursion (plumbing)" begin
    m = AdaptLIF(; τ = 20.0, EL = -65.0, Vθ = 1.0e6, Vr = -65.0, R = 1.0, tref = 0.0,
        a = 0.05, b = 0.0, τw = 100.0)
    @test Dewdrop.statevars(m) == (:V, :refrac, :w)
    @test Dewdrop.float_type(m) == Float64
    @test Dewdrop.reset_value(m) == -65.0
    @test Dewdrop.threshold(m, 0.0) == false   # Vθ huge

    I = 10.0; dt = 0.1; tend = 60.0
    prob = DewdropNetwork(m, 1; input = I, tspan = (0.0, tend))
    sol = solve(prob, FixedStep(dt); record = (V = Trace(:V), w = Trace(:w)))
    # hand-rolled w-first exact-propagator recursion must match the engine to machine precision
    Vr, wr = m.EL, 0.0
    nsteps = size(sol.record.V.data, 2)
    for n in 1:nsteps
        w∞ = m.a * (Vr - m.EL); wr = w∞ + (wr - w∞) * exp(-dt / m.τw)   # w from old V
        V∞ = m.EL + m.R * (I - wr); Vr = V∞ + (Vr - V∞) * exp(-dt / m.τ)  # V from old V, new w
        @test sol.record.V.data[1, n] ≈ Vr
        @test sol.record.w.data[1, n] ≈ wr
    end
end

@testset "AdaptLIF: subthreshold fixed point (analytic)" begin
    # subthreshold (Vθ → ∞), constant input I: the (V,w) system relaxes to
    #   V* = EL + R·I/(1 + R·a),   w* = a·R·I/(1 + R·a)
    m = AdaptLIF(; τ = 20.0, EL = -65.0, Vθ = 1.0e6, Vr = -65.0, R = 1.0, tref = 0.0,
        a = 0.05, b = 0.0, τw = 100.0)
    I = 10.0
    sol = solve(DewdropNetwork(m, 1; input = I, tspan = (0.0, 800.0)), FixedStep(0.1))
    Vstar = m.EL + m.R * I / (1 + m.R * m.a)
    wstar = m.a * m.R * I / (1 + m.R * m.a)
    @test only(sol.state.state.V) ≈ Vstar rtol = 1e-3
    @test only(sol.state.state.w) ≈ wstar rtol = 1e-3
end

@testset "AdaptLIF: spike-frequency adaptation; b=0,a=0 ≡ LIF" begin
    base = (; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0)
    dt, tend = 0.1, 1000.0
    I = 20.0   # supra-threshold (V∞ = -45 > Vθ = -50)

    # adapting model: spike-frequency adaptation -> ISIs grow over the train
    madapt = AdaptLIF(; base..., a = 0.0, b = 1.0, τw = 150.0)
    sa = solve(DewdropNetwork(madapt, 1; input = I, tspan = (0.0, tend)), FixedStep(dt);
        record = (spk = Spikes(),))
    times, _ = raster(sa; name = :spk)
    @test length(times) ≥ 4
    isis = diff(times)
    @test isis[end] > isis[1]                    # later intervals longer (adaptation)
    h = length(isis) ÷ 2                          # second half slower on average (accumulating w)
    @test sum(@view isis[(h + 1):end]) / (length(isis) - h) > sum(@view isis[1:h]) / h

    # a=0,b=0 -> w stays 0 -> identical to the plain LIF f-I point
    mnoadapt = AdaptLIF(; base..., a = 0.0, b = 0.0, τw = 150.0)
    lif = LIF(; base...)
    rate_adapt = only(firing_rate(solve(DewdropNetwork(mnoadapt, 1; input = I, tspan = (0.0, tend)), FixedStep(dt))))
    rate_lif = only(firing_rate(solve(DewdropNetwork(lif, 1; input = I, tspan = (0.0, tend)), FixedStep(dt))))
    @test rate_adapt ≈ rate_lif
end

@testset "AdEx: dynamics vs fine-dt reference; no NaN; sub/supra rheobase" begin
    # canonical adapting AdEx (Brette & Gerstner 2005-ish, canonical units)
    m = AdEx(; C = 200.0, gL = 10.0, EL = -70.0, VT = -50.0, ΔT = 2.0, Vr = -58.0,
        Vpeak = 0.0, a = 2.0, b = 60.0, τw = 120.0, tref = 2.0)
    dt = 0.1

    # sub-rheobase: settles, no spikes, finite
    Isub = 100.0
    ssub = solve(DewdropNetwork(m, 1; input = Isub, tspan = (0.0, 400.0)), FixedStep(dt))
    @test only(ssub.spike_count) == 0
    @test isfinite(only(ssub.state.state.V))

    # supra-rheobase: spikes, finite (Vpeak cutoff fires, no NaN), adapts
    Isup = 500.0
    ssup = solve(DewdropNetwork(m, 1; input = Isup, tspan = (0.0, 1000.0)), FixedStep(dt);
        record = (spk = Spikes(),))
    @test ssup.spike_count[1] ≥ 4
    @test all(isfinite, ssup.state.state.V)
    t_eng, _ = raster(ssup; name = :spk)
    isis = diff(t_eng)
    @test isis[end] ≥ isis[1]                    # adaptation (b large)

    # first-spike time matches a fine-dt forward-Euler reference within a few percent
    t_ref = adex_reference_spikes(m, Isup, 1.0e-3, 1000.0)
    @test !isempty(t_ref)
    @test isapprox(t_eng[1], t_ref[1]; rtol = 0.05)
end

@testset "adaptation: CPU broadcast ≡ JLArray fused; batched reference" begin
    m = AdaptLIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0,
        a = 0.02, b = 0.5, τw = 100.0)
    dt = 0.1
    prob = DewdropNetwork(m, 48; input = 22.0, tspan = (0.0, 200.0))
    cpu = init(prob, FixedStep(dt))
    gpu = adapt(JLArray, init(prob, FixedStep(dt)))
    for _ in 1:2000
        step!(cpu); step!(gpu)
    end
    @test Array(gpu.spike_count) == cpu.spike_count
    @test Array(gpu.state.state.V) ≈ cpu.state.state.V
    @test Array(gpu.state.state.w) ≈ cpu.state.state.w
    @test sum(cpu.spike_count) > 0

    # allocation-free hot loop for a multi-state model
    warm = init(prob, FixedStep(dt)); step!(warm)
    @test @allocated(step!(warm)) == 0

    # batched (streams irrelevant: no drive/noise) -> every column equals the scalar solve
    B = 4
    bsol = solve(prob, FixedStep(dt); batch = B)
    ssol = solve(prob, FixedStep(dt))
    for b in 1:B
        @test bsol.spike_count[:, b] == ssol.spike_count
    end
end

@testset "LIF V-only fast path is preserved (empty-aux dispatch)" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    # the generic advance for a V-only model is exactly membrane_step (no w plumbing)
    v = -65.0
    @test Dewdrop._advance_unit(m, v, nothing, 0.0, 0.1, 0.1) ===
        (Dewdrop.membrane_step(m, v, 0.0, 0.1, 0.1), nothing)
    @test Dewdrop._has_w(typeof(m)) == false
    @test Dewdrop._has_w(typeof(AdaptLIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0,
        R = 1.0, tref = 2.0, a = 0.02, b = 0.5, τw = 100.0))) == true
end
