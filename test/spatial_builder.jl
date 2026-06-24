using Dewdrop
using Test
using Statistics

# A WRCircuit / spatial-FNS "working-regime" circuit built ENTIRELY from first-class Dewdrop components ---
# no bespoke `spatial_fns`/`wrcircuit` constructor, just the builder over generic primitives:
#   • FNSNeuron E/I populations on a periodic sheet (`grid_positions(…; centered=true)` + `random_positions`)
#   • distance-fixed-count recurrent connectivity (`project!` with `kernel`/`count`/`period`)
#   • in-degree-normalised weights (`adjust = correlate_weights(J; seed)`)
#   • exact-COBA `DualExpSynapse` (excitatory Erev=0 from E, inhibitory Erev=−80 from I)
#   • streaming external Poisson drive (`drive!`, one PoissonSource per population)
# This is the consolidation target: a WRCircuit-class circuit is a thin composition of
# `network`/`population!`/`project!`/`drive!`/`build`, not a custom model patching Dewdrop. (Exact COBA +
# independent per-population drive → a statistically-equivalent WRCircuit, not a bit-for-bit BrainPy match.)

@testset "WRCircuit from first-class builder components" begin
    ne, dx, gamma = 12, 0.5, 4
    NE = ne^2                                   # 144 excitatory
    NI = NE ÷ gamma                             # 36 inhibitory
    period = (dx, dx)
    posE = grid_positions(ne, ne; spacing = dx / ne, centered = true)
    posI = random_positions(NI, (dx, dx); seed = UInt64(1))

    E = FNSNeuron(; C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -70.0,
        tref = 4.0, τK = 40.0, ΔgK = 0.002)     # E adapts (conductance-based spike-frequency adaptation)
    I = FNSNeuron(; C = 0.25, gL = 0.025, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -70.0,
        tref = 4.0, τK = 40.0, ΔgK = 0.0)       # I does not adapt

    exc() = DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)     # excitatory (from E)
    inh() = DualExpSynapse(; τr = 2.0, τd = 4.5, Erev = -80.0)  # inhibitory (from I)
    J_ee, J_ei, δ = 0.00105, 0.00145, 4.0
    J_ie, J_ii = J_ee * δ, J_ei * δ             # I weights δ-amplified (sign carried by Erev)
    cw(J, s) = correlate_weights(J; seed = UInt64(s))

    nb = network(; tspan = (0.0, 300.0))
    population!(nb, :E, E, NE; positions = posE)
    population!(nb, :I, I, NI; positions = posI)
    # four recurrent paths: distance-dependent fixed-count connectomes, in-degree-normalised weights, ms delays
    project!(nb, :E => :E, exc(); kernel = exponential_kernel(0.06), count = 30NE, weight = 1.0,
        delay = 1.5, seed = UInt64(2), allow_self = true,  period = period, adjust = cw(J_ee, 6))
    project!(nb, :E => :I, exc(); kernel = exponential_kernel(0.07), count = 30NI, weight = 1.0,
        delay = 1.5, seed = UInt64(3), allow_self = false, period = period, adjust = cw(J_ei, 7))
    project!(nb, :I => :E, inh(); kernel = exponential_kernel(0.14), count = 15NE, weight = 1.0,
        delay = 1.5, seed = UInt64(4), allow_self = false, period = period, adjust = cw(J_ie, 8))
    project!(nb, :I => :I, inh(); kernel = exponential_kernel(0.14), count = 15NI, weight = 1.0,
        delay = 1.5, seed = UInt64(5), allow_self = true,  period = period, adjust = cw(J_ii, 9))
    # streaming external Poisson drive to each population (excitatory, exact-COBA)
    drive!(nb, :E, exc(); rate = 40.0, n_ext = 100, p = 0.3, weight = 1.0, delay = 1.0, seed = UInt64(10), adjust = cw(J_ee, 12))
    drive!(nb, :I, exc(); rate = 40.0, n_ext = 100, p = 0.3, weight = 1.0, delay = 1.0, seed = UInt64(11), adjust = cw(J_ei, 13))

    net = build(nb)
    @test haskey(net.subpops, :E) && haskey(net.subpops, :I)              # named subpops registered
    @test length(net.subpops[:E]) == NE && length(net.subpops[:I]) == NI
    @test length(net.projections) == 6                                   # 4 recurrent + 2 drive

    sol = solve(net, FixedStep(0.1); progress = false)
    rE = mean(sol.spike_count[net.subpops[:E]]) / 0.3                     # mean firing rate (Hz), T = 0.3 s
    rI = mean(sol.spike_count[net.subpops[:I]]) / 0.3
    @test 1 < rE < 100 && 1 < rI < 100                                   # both populations active at plausible rates
    @test maximum(sol.spike_count) < 70                                  # no runaway (refractory cap ≈ 75 over 3000 steps)
    @test solve(build(nb), FixedStep(0.1); progress = false).spike_count == sol.spike_count   # deterministic
end
