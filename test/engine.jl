using Dewdrop
using Test

# M1a cycle 2 --- the fixed-step engine behind the CommonSolve verbs (init/step!/solve!/
# solve), driving a single LIF population. Validated in the SUBTHRESHOLD regime: with
# constant sub-rheobase input the engine must reproduce the exact analytic trajectory
# and never spike. (Spiking + f-I is cycle 3.)
@testset "CommonSolve engine (subthreshold)" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    N = 5
    I = 0.1            # sub-rheobase: V∞ = -60 < Vθ = -50  → no spikes
    dt = 0.1
    tend = 50.0
    V∞ = Dewdrop.asymptote(m, I)

    prob = DewdropNetwork(m, N; input = I, tspan = (0.0, tend))
    integ = init(prob, FixedStep(dt))

    # initial conditions: V = EL, refrac = 0, t = 0, n = 0
    @test all(==(-70.0), integ.state.state.V)
    @test all(==(0.0), integ.state.state.refrac)
    @test integ.t == 0.0
    @test integ.n == 0

    # one step advances (n, t) and integrates V via the exact propagator
    step!(integ)
    @test integ.n == 1
    @test integ.t ≈ dt
    @test all(v -> v ≈ V∞ + (-70.0 - V∞) * exp(-dt / m.τ), integ.state.state.V)

    # `solve` runs a fresh integrator to tend; below rheobase → no spikes; the final V
    # matches the exact analytic value.
    sol = solve(prob, FixedStep(dt))
    @test sol.nsteps == round(Int, tend / dt)
    @test all(iszero, sol.spike_count)
    Vfinal = V∞ + (-70.0 - V∞) * exp(-tend / m.τ)
    @test all(v -> isapprox(v, Vfinal; atol = 1e-6), sol.state.state.V)

    # the schedule is the public, inspectable contract the engine executes
    @test Dewdrop.phases(prob.schedule) == (:deliver, :integrate, :threshold, :reset, :propagate, :record)
end

# The fixed-step loop runs millions of times, so the step must be allocation-free ---
# this also pins that the Val-dispatched schedule unrolls and the phases stay type-stable.
@testset "step! is allocation-free" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    integ = init(DewdropNetwork(m, 64; input = 0.5, tspan = (0.0, 10.0)), FixedStep(0.1))
    step!(integ)            # warm
    step!(integ)
    @test @allocated(step!(integ)) == 0
end
