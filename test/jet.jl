using JET
using Dewdrop
using Test

# Static analysis (JET), two layers:
#   1. whole-package ERROR analysis (test_package) --- catches undefined methods etc.
#   2. OptAnalyzer assertions that PIN hot-path type-stability / dispatch-freedom.
#
# JET tracks Julia's inference internals and is version-sensitive, so this whole block is
# version-gated and ISOLATED in its own file: a JET/Julia upgrade can never turn the
# version-stable @allocated/@inferred backbone (engine.jl/rng.jl/gpu_readiness.jl) red.
# Every assertion is scoped with target_modules=(Dewdrop,) so only dispatch failures
# attributable to Dewdrop's OWN methods count --- this filters Adapt / StructArrays /
# Random123 / Base.Broadcast inference noise while keeping a genuine non-concrete result
# INSIDE a Dewdrop method visible (it is attributed to that method).
if VERSION >= v"1.10"
    @testset "code quality (JET)" begin
        @testset "error analysis (concrete entry points)" begin
            # Concrete-call error analysis over the real entry points --- preferred here over
            # test_package, whose ABSTRACT analysis of the backend-generic `scatter!` kernel
            # launcher reports a GPU-branch false positive (an abstract GPU `Kernel` has no
            # matching `kwcall`) that occurs for NO concrete CPU/GPU call, as the connected
            # `solve` below (which drives `scatter!` with concrete types) confirms.
            m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
            prob = DewdropNetwork(m, 8; input = 0.5, tspan = (0.0, 5.0))
            JET.@test_call target_modules = (Dewdrop,) solve(prob, FixedStep(0.1))
            JET.@test_call target_modules = (Dewdrop,) Dewdrop.draw_uniform(Float64, UInt64(1), 1, 1)

            conn = SparseCSR(Dewdrop.CPU(), [(1, 2, 40.0, 15)]; npre = 2, npost = 2)
            cprob = DewdropNetwork(m, 2; input = [0.5, 0.0], tspan = (0.0, 5.0),
                projection = Projection(CurrentSynapse(τ = 5.0), conn))
            JET.@test_call target_modules = (Dewdrop,) solve(cprob, FixedStep(0.1))
            buf = Dewdrop.DelayBuffer(Dewdrop.CPU(), Float64, 2, 5)
            JET.@test_call target_modules = (Dewdrop,) Dewdrop.scatter!(buf, conn, [true, false], 0)
        end

        @testset "hot-path optimization (no runtime dispatch)" begin
            m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
            prob = DewdropNetwork(m, 64; input = 0.5, tspan = (0.0, 10.0))
            integ = init(prob, FixedStep(0.1))
            step!(integ)   # warm
            step!(integ)

            # THE load-bearing assertion: the whole fixed-step is dispatch-free. Pins that
            # the @generated schedule unroll fully resolves and the fused SoA broadcasts
            # stay concrete. Complements @allocated(step!)==0 (boxing) in engine.jl --- this
            # catches runtime dispatch that need not allocate (e.g. a Nothing-returning one).
            JET.@test_opt target_modules = (Dewdrop,) step!(integ)

            # per-phase localisation so a regression points at the offending @. broadcast
            JET.@test_opt target_modules = (Dewdrop,) Dewdrop.run_phase!(Val(:integrate), integ)
            JET.@test_opt target_modules = (Dewdrop,) Dewdrop.run_phase!(Val(:threshold), integ)
            JET.@test_opt target_modules = (Dewdrop,) Dewdrop.run_phase!(Val(:reset), integ)
            JET.@test_opt target_modules = (Dewdrop,) Dewdrop.run_phase!(Val(:record), integ)

            # the generated schedule unroll itself
            JET.@test_opt target_modules = (Dewdrop,) Dewdrop.run_phases!(integ.schedule, integ)

            # RNG hot path (complements the @allocated==0 SROA guard in rng.jl)
            JET.@test_opt target_modules = (Dewdrop,) Dewdrop.draw_uniform(Float64, UInt64(1), 1, 1)
            JET.@test_opt target_modules = (Dewdrop,) Dewdrop.draw_uniform(Float32, UInt64(1), 1, 1)

            # construction + the solve verb surface at call granularity
            JET.@test_opt target_modules = (Dewdrop,) init(prob, FixedStep(0.1))
            JET.@test_call target_modules = (Dewdrop,) solve(prob, FixedStep(0.1))

            # the CONNECTED (synapse-coupled) hot path must also be dispatch-free: the phase
            # dispatch on synaptic-state type, per-unit input, and the KA scatter launch all
            # resolve at compile time.
            conn = SparseCSR(Dewdrop.CPU(), [(1, 2, 40.0, 15)]; npre = 2, npost = 2)
            cprob = DewdropNetwork(m, 2; input = [0.5, 0.0], tspan = (0.0, 10.0),
                projection = Projection(CurrentSynapse(τ = 5.0), conn))
            cinteg = init(cprob, FixedStep(0.1))
            step!(cinteg)
            JET.@test_opt target_modules = (Dewdrop,) step!(cinteg)
            JET.@test_opt target_modules = (Dewdrop,) init(cprob, FixedStep(0.1))

            # the Brunel path: delta synapses + Poisson drive must also be dispatch-free
            dconn = SparseCSR(Dewdrop.CPU(), [(1, 2, 0.2, 15), (2, 1, -1.0, 15)]; npre = 2, npost = 2)
            dprob = DewdropNetwork(m, 2; input = 0.0, tspan = (0.0, 5.0),
                projection = Projection(DeltaSynapse(), dconn),
                drive = PoissonDrive(; rate = 6.0, weight = 0.2, seed = UInt64(2)))
            dinteg = init(dprob, FixedStep(0.1))
            step!(dinteg)
            JET.@test_opt target_modules = (Dewdrop,) step!(dinteg)
            JET.@test_opt target_modules = (Dewdrop,) Dewdrop.draw_poisson(2.0, UInt64(1), 1, 1)
        end
    end
end
