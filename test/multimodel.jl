using Dewdrop
using Test
using Adapt
using JLArrays

# Multi-type populations --- `MultiModel` holds an ordered tuple of (model, range) groups
# over one flat concatenated SoA, so a network can mix neuron model TYPES (e.g. AdEx excitatory +
# LIF inhibitory) in one engine. State is a UNION SoA (columns = union of the groups' statevars; a
# group ignores columns its model does not declare). A heterogeneous model routes through the fused
# megakernel, launched once per group over its range with the group's concrete model --- each launch
# is monomorphic, so it specialises exactly like the single-model kernel. One group spanning the
# whole population reduces to the single-model run.

mE() = AdEx(; C = 200.0, gL = 10.0, EL = -70.0, VT = -50.0, ΔT = 2.0, Vr = -58.0, Vpeak = 0.0,
    a = 2.0, b = 60.0, τw = 120.0, tref = 2.0)
mI() = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 2.0)

@testset "MultiModel: union SoA + metadata" begin
    mm = Dewdrop.MultiModel([mE(), mI()], [30, 30])
    @test Dewdrop.statevars(typeof(mm)) == (:V, :refrac, :w)   # union: AdEx adds :w, LIF rides along
    @test Dewdrop.float_type(mm) == Float64
    @test Dewdrop._is_hetero(mm)                               # routes through the fused per-group path
    st = Dewdrop.Population(Dewdrop.CPU(), mm, 60).state
    @test :w in propertynames(st)                              # the union allocates the w column
    @test length(st.V) == 60
end

@testset "MultiModel: distinct types each evolve (AdEx-E + LIF-I)" begin
    mm = Dewdrop.MultiModel([mE(), mI()], [30, 30])
    inp = vcat(fill(700.0, 30), fill(400.0, 30))              # E (AdEx) and I (LIF) both supra-threshold
    sol = solve(DewdropNetwork(mm, 60; input = inp, tspan = (0.0, 500.0)), FixedStep(0.05))
    @test sum(sol.spike_count[1:30]) > 0                       # AdEx group fires
    @test sum(sol.spike_count[31:60]) > 0                      # LIF group fires
    @test all(isfinite, sol.state.state.V)                    # no NaN from the AdEx exp term
end

@testset "MultiModel: CPU ≡ JLArray fused (bit-identical spikes)" begin
    mm = Dewdrop.MultiModel([mE(), mI()], [30, 30])
    inp = vcat(fill(700.0, 30), fill(400.0, 30))
    prob = DewdropNetwork(mm, 60; input = inp, tspan = (0.0, 200.0))
    cpu = init(prob, FixedStep(0.05))
    gpu = adapt(JLArray, init(prob, FixedStep(0.05)))
    for _ in 1:4000
        step!(cpu); step!(gpu)
    end
    @test sum(cpu.spike_count) > 0
    @test Array(gpu.spike_count) == cpu.spike_count
    @test Array(gpu.state.state.V) ≈ cpu.state.state.V
    @test Array(gpu.state.state.w) ≈ cpu.state.state.w
end

@testset "MultiModel: one group spanning 1:N ≡ the bare model" begin
    m = mI()
    inp = fill(400.0, 50)
    bare = solve(DewdropNetwork(m, 50; input = inp, tspan = (0.0, 200.0)), FixedStep(0.05))
    mm1 = solve(DewdropNetwork(Dewdrop.MultiModel([m], [50]), 50; input = inp, tspan = (0.0, 200.0)), FixedStep(0.05))
    @test mm1.spike_count == bare.spike_count                 # same dynamics, fused vs broadcast on CPU
    @test mm1.state.state.V ≈ bare.state.state.V
end

@testset "MultiModel: batched run errors clearly (scope)" begin
    mm = Dewdrop.MultiModel([mE(), mI()], [30, 30])
    prob = DewdropNetwork(mm, 60; input = vcat(fill(700.0, 30), fill(400.0, 30)), tspan = (0.0, 50.0))
    @test_throws Exception init(prob, FixedStep(0.05); batch = 4)
end
