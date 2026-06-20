using Dewdrop
using Test

# M1a cycle 1 --- the LIF neuron model and its exact subthreshold propagator.
# Units (plain floats, consistent): time ms, V mV, R MΩ, I nA (so R·I is mV).
@testset "LIF model + exact subthreshold propagator" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)

    @test Dewdrop.statevars(m) == (:V, :refrac)
    @test Dewdrop.float_type(m) === Float64
    @test Dewdrop.threshold(m, -49.0) == true
    @test Dewdrop.threshold(m, -51.0) == false
    @test Dewdrop.reset_value(m) == -60.0
    @test Dewdrop.refractory(m) == 2.0

    # asymptote V∞ = EL + R·I
    @test Dewdrop.asymptote(m, 0.1) ≈ -60.0      # -70 + 100*0.1
    @test Dewdrop.asymptote(m, 0.0) ≈ -70.0

    # THE correctness anchor: iterating the exact linear propagator reproduces the
    # analytic exponential relaxation V(t) = V∞ + (V0 - V∞)·exp(-t/τ) to ~machine
    # precision at every step (constant sub-rheobase input, so no threshold crossing).
    I = 0.1                                       # R·I = 10 < (Vθ-EL) = 20  → sub-rheobase
    V∞ = Dewdrop.asymptote(m, I)
    dt = 0.1
    decay = Dewdrop.propagator_decay(m, dt)
    @test decay ≈ exp(-dt / m.τ)

    V0 = -70.0
    V = V0
    for n in 1:1000
        V = Dewdrop.subthreshold_step(V, V∞, decay)
        analytic = V∞ + (V0 - V∞) * exp(-(n * dt) / m.τ)
        @test V ≈ analytic
    end
    # after 1000 steps = 100 ms = 5τ the residual is (V0-V∞)·e⁻⁵ ≈ 0.067 mV, and V
    # approaches V∞ monotonically from below (V0 < V∞).
    @test V < V∞
    @test V∞ - V ≈ (V∞ - V0) * exp(-(1000 * dt) / m.τ)

    # parametric float type is preserved through the constructor
    m32 = LIF(; τ = 20.0f0, EL = -70.0f0, Vθ = -50.0f0, Vr = -60.0f0, R = 100.0f0, tref = 2.0f0)
    @test Dewdrop.float_type(m32) === Float32
    @test Dewdrop.asymptote(m32, 0.1f0) isa Float32
end
