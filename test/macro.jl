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
    scoba = solve(
        DewdropNetwork(
            MacroLIF(), 16; input = 1.5, tspan = (0.0, 50.0),
            projection = Projection(ConductanceSynapse(τ = 5.0, Erev = 0.0), ce)
        ), FixedStep(0.1)
    )
    # subthreshold drive (asymptote EL + R·I = 1.5 < Vθ), so no spikes and no synaptic conductance:
    # V must sit on the analytic LIF trajectory, which is what checks the generated model integrates.
    @test sum(scoba.spike_count) == 0
    @test all(v -> isapprox(v, 1.5 * (1 - exp(-50 / 20)); atol = 1.0e-8), scoba.state.state.V)
end

# A `@neuron` model must not require integer-free defaults, nor a leak reversal literally named `EL`:
# integer defaults gave a `{Int}` model (integer state columns → InexactError on the first store), and
# the generic `_resting` read `m.EL`, so any other name failed at init with a raw FieldError.
@neuron IntDefaultLIF begin
    @parameters τ = 20 VL = 0 Vθ = 20 Vr = 10 R = 1 tref = 2
    @state V refrac
    @asymptote VL + R * I
    @resistance R
    @timeconstant τ
    @threshold V >= Vθ
    @reset Vr
    @refractory tref
end

@testset "@neuron: integer defaults and a non-EL leak reversal" begin
    m = IntDefaultLIF()
    @test Dewdrop.float_type(m) <: AbstractFloat            # not Int
    @test Dewdrop._resting(m) == 0                          # the asymptote at zero input, i.e. VL
    sol = solve(DewdropNetwork(m, 8; input = 1.5, tspan = (0.0, 50.0)), FixedStep(0.1); progress = false)
    @test eltype(sol.state.state.V) <: AbstractFloat
    @test all(v -> isapprox(v, 1.5 * (1 - exp(-50 / 20)); atol = 1.0e-8), sol.state.state.V)

    # a non-zero leak reversal is what the default v0 uses, per neuron
    m2 = IntDefaultLIF(; VL = -55)
    @test init(DewdropNetwork(m2, 4; input = 0.0, tspan = (0.0, 1.0)), FixedStep(0.1)).state.state.V ==
        fill(-55.0, 4)
end

# The macro is imported as often as it is `using`-ed, so the generated code must not assume the name
# `Dewdrop` is bound in the caller.
module MacroHygiene
    import Dewdrop: @neuron
    @neuron HygieneLIF begin
        @parameters τ = 20.0 EL = 0.0 Vθ = 20.0 Vr = 10.0 R = 1.0 tref = 2.0
        @state V refrac
        @asymptote EL + R * I
        @resistance R
        @timeconstant τ
        @threshold V ≥ Vθ
        @reset Vr
        @refractory tref
    end
end

@testset "@neuron expands without `Dewdrop` bound in the caller" begin
    @test !isdefined(MacroHygiene, :Dewdrop)
    mm = MacroHygiene.HygieneLIF()
    @test Dewdrop.threshold(mm, 25.0) && !Dewdrop.threshold(mm, 15.0)
    sol = solve(DewdropNetwork(mm, 4; input = 30.0, tspan = (0.0, 100.0)), FixedStep(0.1))
    @test sum(sol.spike_count) > 0
end

# `_subparams` rewrites every parameter symbol to a field access, so a parameter named `V` or `I`
# would silently replace the membrane or the input current in the hook expressions.
@testset "@neuron refuses parameters that shadow the reserved names" begin
    @test_throws Exception @eval @neuron ShadowsV begin
        @parameters V = 1.0 τ = 20.0
        @state V refrac
        @asymptote V + I
        @resistance τ
        @timeconstant τ
        @threshold V ≥ 0
        @reset 0.0
        @refractory 0.0
    end
end
