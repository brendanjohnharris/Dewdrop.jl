using Dewdrop
using LoopVectorization        # activates DewdropLoopVectorizationExt → registers the Turbo kernels
using Test

# The `Turbo` backend (SIMD via LoopVectorization): the dense membrane/threshold/reset is replaced by
# a `@turbo`-vectorised kernel chosen per model (`turbo_kernel`), the rest of the step stays scalar.
# Spike-IDENTICAL to `Serial` (V differs only at the SIMD-exp ULP level), CPU-only, opt-in. NOTE: this
# file `using LoopVectorization`, so it must come AFTER test/backends.jl (which assumes no ext).

@testset "Turbo backend (LoopVectorization extension)" begin
    @test Dewdrop._turbo_available()                 # the extension loaded
    @test Dewdrop.supports_turbo(AdEx) && Dewdrop.supports_turbo(LIF)
    @test Dewdrop.turbo_kernel(AdEx) !== nothing && Dewdrop.turbo_kernel(LIF) !== nothing

    # spike-identical to Serial, for the built-in supported families
    function check(model, name; T = 300.0, dt = 0.1f0, N = 2000)
        conn = fixed_prob(Dewdrop.CPU(), N, N, 0.05f0; weight = 0.05f0, delay = 2, seed = UInt64(1))
        prob = DewdropNetwork(model, N; input = 400.0f0, tspan = (0.0f0, Float32(T)),
            projection = Projection(CurrentSynapse(τ = 5.0f0), conn))
        ss = solve(prob, FixedStep(dt); backend = Serial(), v0 = (-70.0f0, -50.0f0))
        st = solve(prob, FixedStep(dt); backend = Turbo(), v0 = (-70.0f0, -50.0f0))
        @testset "$name: Turbo ≡ Serial (spike-identical)" begin
            @test sum(ss.spike_count) > 0
            @test st.spike_count == ss.spike_count                       # identical spike trains
            @test maximum(abs.(st.state.state.V .- ss.state.state.V)) < 0.5f0   # V close (SIMD-exp ULP)
        end
    end
    check(AdEx(; C = 200.0f0, gL = 10.0f0, EL = -70.0f0, VT = -50.0f0, ΔT = 2.0f0, Vr = -58.0f0,
            Vpeak = -40.0f0, a = 0.0f0, b = 40.0f0, τw = 100.0f0, tref = 0.0f0), "AdEx")
    check(LIF(; τ = 20.0f0, EL = -70.0f0, Vθ = -50.0f0, Vr = -60.0f0, R = 0.1f0, tref = 2.0f0), "LIF")

    # COBA is handled too (the kernel reads gtot)
    @testset "Turbo with COBA conductance synapses" begin
        m = LIF(; τ = 20.0f0, EL = -60.0f0, Vθ = -50.0f0, Vr = -60.0f0, R = 1.0f0, tref = 5.0f0)
        conn = fixed_prob(Dewdrop.CPU(), 1000, 1000, 0.05f0; weight = 0.6f0, delay = 1, seed = UInt64(2))
        prob = DewdropNetwork(m, 1000; input = 0.0f0, tspan = (0.0f0, 150.0f0),
            projection = Projection(ConductanceSynapse(τ = 5.0f0, Erev = 0.0f0), conn))
        ss = solve(prob, FixedStep(0.1f0); backend = Serial(), v0 = (-60.0f0, -50.0f0))
        st = solve(prob, FixedStep(0.1f0); backend = Turbo(), v0 = (-60.0f0, -50.0f0))
        @test st.spike_count == ss.spike_count
    end

    @testset "with-extension error paths" begin
        m = AdEx(; C = 200.0f0, gL = 10.0f0, EL = -70.0f0, VT = -50.0f0, ΔT = 2.0f0, Vr = -58.0f0,
            Vpeak = -40.0f0, a = 0.0f0, b = 40.0f0, τw = 100.0f0, tref = 0.0f0)
        # unsupported model → clear error
        fns = FNSNeuron()
        pf = DewdropNetwork(fns, 32; input = 0.5, tspan = (0.0, 10.0))
        @test_throws Exception init(pf, FixedStep(0.1); backend = Turbo())
        # WhiteNoise → unsupported under Turbo
        pn = DewdropNetwork(m, 32; input = 400.0f0, tspan = (0.0f0, 10.0f0), noise = WhiteNoise(1.0f0))
        @test_throws Exception init(pn, FixedStep(0.1f0); backend = Turbo())
        # deprecated step alias also reaches Turbo
        @test sum(solve(DewdropNetwork(m, 64; input = 400.0f0, tspan = (0.0f0, 30.0f0)),
            FixedStep(0.1f0); step = :turbo, v0 = (-70.0f0, -50.0f0)).spike_count) ≥ 0
    end
end
