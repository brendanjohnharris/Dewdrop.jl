using Dewdrop
using Test

# Engine behaviours not covered by the subthreshold / f-I tests: the CommonSolve verb
# interface, solution metadata, the exact refractory branch semantics, custom schedules,
# float-type propagation.
@testset "engine behaviour" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)

    @testset "CommonSolve verb interface" begin
        prob = DewdropNetwork(m, 4; input = 0.5, tspan = (0.0, 100.0))
        # step! returns the integrator (CommonSolve interface)
        integ0 = init(prob, FixedStep(0.1))
        @test step!(integ0) === integ0
        # solve! consumes a pre-built integrator and equals the solve convenience
        sol_a = solve!(init(prob, FixedStep(0.1)))
        sol_b = solve(prob, FixedStep(0.1))
        @test sol_a isa DewdropSolution
        @test sol_a.nsteps == sol_b.nsteps
        @test sol_a.spike_count == sol_b.spike_count
    end

    @testset "solution metadata" begin
        dt, tend = 0.1, 50.0
        sol = solve(DewdropNetwork(m, 3; input = 0.5, tspan = (0.0, tend)), FixedStep(dt))
        @test sol.nsteps == round(Int, tend / dt)
        @test Dewdrop.duration(sol) ≈ sol.nsteps * dt
        @test sol.tspan[2] ≈ tend
        @test sol.tspan[1] ≈ 0.0 atol = 1.0e-9
    end

    @testset "refractory branch semantics (direct)" begin
        dt = 0.1
        # very strong drive → spikes on the first step
        integ = init(DewdropNetwork(m, 1; input = 50.0, tspan = (0.0, 100.0)), FixedStep(dt))
        step!(integ)
        @test integ.spike_count[1] == 1
        @test integ.state.state.V[1] == m.Vr            # reset to Vr
        @test integ.state.state.refrac[1] == m.tref     # refractory armed
        # next step: clamped at Vr, refrac decremented by dt, cannot spike again
        step!(integ)
        @test integ.state.state.V[1] == m.Vr
        @test integ.state.state.refrac[1] ≈ m.tref - dt
        @test integ.spike_count[1] == 1
    end

    @testset "custom schedule (arbitrary order/subset honoured)" begin
        dt, tend = 0.1, 500.0
        # omitting the no-op synapse phases reproduces the default schedule exactly
        minimal = Schedule(:integrate, :threshold, :reset, :record)
        sol_min = solve(DewdropNetwork(m, 1; input = 0.5, tspan = (0.0, tend), schedule = minimal), FixedStep(dt))
        sol_def = solve(DewdropNetwork(m, 1; input = 0.5, tspan = (0.0, tend)), FixedStep(dt))
        @test only(sol_min.spike_count) == only(sol_def.spike_count)
        @test only(sol_def.spike_count) > 0
        # dropping :reset → a supra-threshold unit never resets and fires (nearly) every
        # step after crossing → far more spikes. Pins that the unroll honours the subset.
        noreset = Schedule(:integrate, :threshold, :record)
        sol_nr = solve(DewdropNetwork(m, 1; input = 0.5, tspan = (0.0, tend), schedule = noreset), FixedStep(dt))
        @test only(sol_nr.spike_count) > 10 * only(sol_def.spike_count)
    end

    @testset "float-type propagation (Float32 end-to-end)" begin
        m32 = LIF(; τ = 20.0f0, EL = -70.0f0, Vθ = -50.0f0, Vr = -60.0f0, R = 100.0f0, tref = 2.0f0)
        # integer tspan is coerced to the model's float type
        prob = DewdropNetwork(m32, 4; input = 0.1f0, tspan = (0, 50))
        @test prob.tspan isa Tuple{Float32, Float32}
        sol = solve(prob, FixedStep(0.1f0))
        @test eltype(sol.state.state.V) === Float32
        V∞ = Dewdrop.asymptote(m32, 0.1f0)
        @test V∞ isa Float32
        Vfinal = V∞ + (-70.0f0 - V∞) * exp(-50.0f0 / m32.τ)
        @test all(v -> isapprox(v, Vfinal; atol = 1.0f-2), sol.state.state.V)
    end
end
