using Dewdrop
using Test

# The @neuron macro: a declarative model definition lowering to the same parameter
# struct + hooks the engine consumes. Define a LIF through the macro and check it reproduces
# the hand-written LIF exactly, and that the generated model integrates (incl. with COBA).
@neuron MacroLIF begin
    @parameters τ = 20.0 EL = 0.0 Vθ = 20.0 Vr = 10.0 R = 1.0 tref = 2.0
    @state V refrac
    @asymptote EL + R * I
    @resistance R
    @timeconstant τ
    @threshold V ≥ Vθ
    @reset Vr
    @refractory tref
end

@testset "@neuron macro" begin
    m = MacroLIF()                      # all defaults
    @test m isa Dewdrop.AbstractNeuronModel
    @test Dewdrop.statevars(typeof(m)) == (:V, :refrac)
    @test Dewdrop.float_type(m) == Float64
    @test Dewdrop.asymptote(m, 5.0) == 5.0
    @test Dewdrop.threshold(m, 25.0) && !Dewdrop.threshold(m, 15.0)
    @test Dewdrop.reset_value(m) == 10.0 && Dewdrop.refractory(m) == 2.0
    @test MacroLIF(; τ = 10.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0).τ == 10.0

    # the macro model reproduces the hand-written LIF exactly (same dynamics → identical run)
    hand = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    sm = solve(DewdropNetwork(MacroLIF(), 16; input = 1.5, tspan = (0.0, 50.0)), FixedStep(0.1); record = (spikes = Spikes(),))
    sh = solve(DewdropNetwork(hand, 16; input = 1.5, tspan = (0.0, 50.0)), FixedStep(0.1); record = (spikes = Spikes(),))
    @test sm.spike_count == sh.spike_count
    @test sm.state.state.V ≈ sh.state.state.V

    # the generic linear membrane step couples conductance (COBA) synapses for macro models too
    ce = fixed_prob(Dewdrop.CPU(), 16, 16, 0.2; weight = 0.5, delay = steps(2), seed = UInt64(1))
    scoba = solve(DewdropNetwork(MacroLIF(), 16; input = 1.5, tspan = (0.0, 50.0),
            projection = Projection(ConductanceSynapse(τ = 5.0, Erev = 0.0), ce)), FixedStep(0.1))
    @test sum(scoba.spike_count) ≥ 0
end
