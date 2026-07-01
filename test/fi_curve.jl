using Dewdrop
using Test

# Validate the full threshold/reset/refractory machinery against the
# analytic LIF f-I curve:  ISI = tref + τ·ln((V∞ - Vr)/(V∞ - Vθ))  for V∞ = EL + R·I > Vθ,
# else the neuron is silent. The fixed-step engine reproduces this up to dt-quantisation
# of the threshold crossing (sub-dt interpolation is a deferred refinement).
@testset "LIF f-I curve" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    dt = 0.01
    tend = 2000.0      # long window so finite-count granularity is << tolerance

    function analytic_rate(I)
        V∞ = m.EL + m.R * I
        V∞ ≤ m.Vθ && return 0.0
        return 1.0 / (m.tref + m.τ * log((V∞ - m.Vr) / (V∞ - m.Vθ)))   # spikes per ms
    end

    # sub-rheobase and exactly-rheobase: silent
    for I in (0.0, 0.1, 0.2)
        sol = solve(DewdropNetwork(m, 1; input = I, tspan = (0.0, tend)), FixedStep(dt))
        @test only(sol.spike_count) == 0
    end

    # supra-rheobase: simulated rate matches the analytic f-I curve
    for I in (0.3, 0.5, 0.8, 1.2)
        sol = solve(DewdropNetwork(m, 1; input = I, tspan = (0.0, tend)), FixedStep(dt))
        sim = only(firing_rate(sol))
        @test isapprox(sim, analytic_rate(I); rtol = 0.02)
    end

    # refractory floor: rate can never exceed 1/tref no matter how strong the drive
    sol = solve(DewdropNetwork(m, 1; input = 100.0, tspan = (0.0, tend)), FixedStep(dt))
    @test only(firing_rate(sol)) ≤ 1.0 / m.tref + 1.0e-9

    # a homogeneous population fires identically across units (determinism + no cross-talk)
    solN = solve(DewdropNetwork(m, 16; input = 0.5, tspan = (0.0, tend)), FixedStep(dt))
    @test all(==(solN.spike_count[1]), solN.spike_count)
end
