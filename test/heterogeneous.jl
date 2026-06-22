using Dewdrop
using Test
using Adapt
using JLArrays

# Phase 3 --- per-neuron heterogeneous parameters. `Heterogeneous(base; field = array, …)` wraps a
# scalar neuron model and overrides chosen parameters with per-neuron arrays (the storage decision:
# frozen arrays, computed once, read many; fill reproducibly via the counter RNG). The engine
# resolves a per-neuron scalar model in the hot loop, so a homogeneous run is unchanged and a
# heterogeneous one (e.g. E adapts / I doesn't, the WRCircuit E/I pattern) works on every path.

@testset "no-op heterogeneity ≡ scalar model (bit-identical)" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    N = 64
    hm = Heterogeneous(m; Vθ = fill(m.Vθ, N), R = fill(m.R, N))   # overrides equal the scalar values
    @test Dewdrop.statevars(hm) == Dewdrop.statevars(m)
    @test Dewdrop.float_type(hm) == Float64
    prob_s = DewdropNetwork(m, N; input = 30.0, tspan = (0.0, 200.0))
    prob_h = DewdropNetwork(hm, N; input = 30.0, tspan = (0.0, 200.0))
    s = solve(prob_s, FixedStep(0.1))
    h = solve(prob_h, FixedStep(0.1))
    @test sum(s.spike_count) > 0
    @test h.spike_count == s.spike_count                          # identical: overrides match the scalar
    @test h.state.state.V == s.state.state.V
end

@testset "per-neuron parameter takes effect (heterogeneous threshold)" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    N = 100
    Vθ = [i <= N ÷ 2 ? 20.0 : 1.0e6 for i in 1:N]                 # second half: unreachable threshold
    hm = Heterogeneous(m; Vθ = Vθ)
    sol = solve(DewdropNetwork(hm, N; input = 30.0, tspan = (0.0, 300.0)), FixedStep(0.1))
    @test all(>(0), sol.spike_count[1:(N ÷ 2)])                   # low-threshold half fires
    @test all(==(0), sol.spike_count[(N ÷ 2 + 1):end])           # high-threshold half never fires
end

@testset "block E/I heterogeneity: only E adapts (WRCircuit pattern)" begin
    base = AdaptLIF(; τ = 20.0, EL = -65.0, Vθ = -50.0, Vr = -65.0, R = 1.0, tref = 2.0,
        a = 0.0, b = 0.0, τw = 150.0)
    N = 100
    b = [i <= N ÷ 2 ? 1.0 : 0.0 for i in 1:N]                     # E (1:50) adapt, I (51:100) do not
    hm = Heterogeneous(base; b = b)
    sol = solve(DewdropNetwork(hm, N; input = 20.0, tspan = (0.0, 1000.0)), FixedStep(0.1))
    @test sum(sol.spike_count[1:(N ÷ 2)]) > 0
    # spike-frequency adaptation suppresses the adapting half relative to the non-adapting half
    @test sum(sol.spike_count[1:(N ÷ 2)]) < sum(sol.spike_count[(N ÷ 2 + 1):end])
end

@testset "reproducible per-neuron fill helper" begin
    @test Dewdrop.per_neuron(i -> 2.0 * i, 5) == [2.0, 4.0, 6.0, 8.0, 10.0]
    # counter-RNG fill is reproducible (same seed → same draws), the endorsed distribution path
    f = i -> Dewdrop.draw_uniform(Float64, UInt64(7), 0, i)
    @test Dewdrop.per_neuron(f, 50) == Dewdrop.per_neuron(f, 50)
end

@testset "heterogeneous: CPU broadcast ≡ JLArray fused; allocation-free" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    N = 64
    hm = Heterogeneous(m; Vθ = [15.0 + 0.1 * i for i in 1:N], tref = fill(2.0, N))
    prob = DewdropNetwork(hm, N; input = 25.0, tspan = (0.0, 50.0))
    cpu = init(prob, FixedStep(0.1))
    gpu = adapt(JLArray, init(prob, FixedStep(0.1)))
    for _ in 1:500
        step!(cpu); step!(gpu)
    end
    @test sum(cpu.spike_count) > 0
    @test Array(gpu.spike_count) == cpu.spike_count
    @test Array(gpu.state.state.V) ≈ cpu.state.state.V

    # the per-neuron model resolution is allocation-free (the hot-loop concern) --- the reconstructed
    # isbits model lives on the stack. (The fused step itself allocates a fixed KA launch, by design.)
    Dewdrop._resolve(hm, 1)
    @test @allocated(Dewdrop._resolve(hm, 1)) == 0
    @test Dewdrop._resolve(hm, 3) isa typeof(m)            # resolves to the base (scalar) model type
end
