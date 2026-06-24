using Dewdrop
using Test
using Statistics

# Native spatial FNS "working-regime" constructor (`spatial_fns`) --- the BrainPy `WRCircuit.jl` `Spatial`
# model re-expressed over Dewdrop primitives. These checks pin the construction invariants (geometry,
# exact fixed-count connectomes, in-degree-scaled weights, broad-inhibition kernel ordering, the external
# drive, E-adapts/I-does-not, determinism) WITHOUT BrainPy. The structural distribution-match against a
# seeded BrainPy export (in-degree / weight / connection-distance means) lives in
# test/simulator_comparisons/wrcircuit; the engine's bit-for-bit reproduction is validated there too.

# mean periodic connection distance of a projection's edges (tests the kernel length scale)
function _mean_conndist(conn, pos, period)
    src = Int.(conn.src); post = Int.(conn.post)
    return mean(Dewdrop.distance(pos[src[e]], pos[post[e]], period) for e in eachindex(src))
end

@testset "spatial_fns: geometry, exact connectomes, in-degree-scaled weights" begin
    # rho=400, dx=0.6 → ne = round(√400·0.6) = 12 → NE=144, NI=round(144/4)=36, N=180
    prob = spatial_fns(; rho = 400.0, dx = 0.6, gamma = 4,
        K_ee = 20, K_ei = 25, K_ie = 20, K_ii = 25, nu = 15.0, n_ext = 40,
        delta = 4.0, J_ee = 0.00105, J_ei = 0.00145, Delta_g_K = 0.005,
        tspan = (0.0, 300.0), dt = 0.1, seed = 20)
    NE, NI, N = 144, 36, 180
    @test prob.n == N
    @test prob.subpops.E == 1:NE && prob.subpops.I == (NE + 1):N
    @test prob.model isa Dewdrop.Heterogeneous                 # E adapts (ΔgK>0), I does not (ΔgK=0)
    @test length(prob.positions) == N

    # four recurrent projections (FrozenDualExpSynapse) + the external drive (streaming by default)
    @test length(prob.projections) == 5
    @test prob.projections[5].synapse isa PoissonSource
    cee, cei, cie, cii = (prob.projections[k].conn for k in 1:4)
    # EXACT fixed edge counts = K_xx · N_post (mean in-degree = K_xx)
    @test Dewdrop.nedges(cee) == 20 * NE
    @test Dewdrop.nedges(cei) == 25 * NI
    @test Dewdrop.nedges(cie) == 20 * NE
    @test Dewdrop.nedges(cii) == 25 * NI

    # in-degree-scaled weights: E2E mean ≈ J_ee; I weights are δ-amplified (J_ie = J_ee·δ, J_ii = J_ei·δ);
    # sign of inhibition is carried by the reversal potential, so every weight stays positive.
    @test all(>(0), cee.weight) && all(>(0), cie.weight)
    @test isapprox(mean(cee.weight), 0.00105; rtol = 0.05)
    @test isapprox(mean(cei.weight), 0.00145; rtol = 0.05)
    @test isapprox(mean(cie.weight), 0.00105 * 4.0; rtol = 0.05)   # J_ie = J_ee·δ
    @test isapprox(mean(cii.weight), 0.00145 * 4.0; rtol = 0.05)   # J_ii = J_ei·δ

    # broad inhibition: I projections (σ=0.14) reach farther than E projections (σ=0.06/0.07)
    period = (0.6, 0.6)
    dE = _mean_conndist(cee, prob.positions, period)
    dI = _mean_conndist(cie, prob.positions, period)
    @test dI > dE
end

@testset "spatial_fns: dynamics --- adaptation, activity, determinism" begin
    mk(seed) = spatial_fns(; rho = 400.0, dx = 0.6, K_ee = 20, K_ei = 25, K_ie = 20, K_ii = 25,
        nu = 15.0, n_ext = 40, Delta_g_K = 0.005, tspan = (0.0, 300.0), dt = 0.1, seed = seed)
    NE, NI, N = 144, 36, 180
    prob = mk(20)
    sol = solve(prob, FixedStep(0.1); v0 = (-70.0, -50.0), record = (spikes = Spikes(),))
    sol2 = solve(prob, FixedStep(0.1); v0 = (-70.0, -50.0), record = (spikes = Spikes(),))

    @test all(isfinite, sol.state.state.V)                      # conductance dynamics stay finite
    @test sol.spike_count == sol2.spike_count                   # deterministic given (seed, v0)
    @test sum(sol.spike_count[1:NE]) > 0                        # E active
    @test sum(sol.spike_count[(NE + 1):N]) > 0                  # I active (broad inhibition still leaves activity)
    @test maximum(sol.state.state.w[1:NE]) > 0                  # E adaptation conductance accumulates
    @test all(iszero, sol.state.state.w[(NE + 1):N])           # I never adapts (ΔgK = 0)

    # a different seed → a different connectome/drive realisation → different spikes
    solb = solve(mk(21), FixedStep(0.1); v0 = (-70.0, -50.0))
    @test solb.spike_count != sol.spike_count
end

@testset "spatial_fns: synapse scheme + argument validation" begin
    # the exact-propagator COBA variant also assembles and runs finite
    prob = spatial_fns(; rho = 400.0, dx = 0.6, nu = 15.0, n_ext = 40, K_ee = 20, K_ei = 25,
        K_ie = 20, K_ii = 25, tspan = (0.0, 200.0), dt = 0.1, seed = 7, synapse = :exact)
    @test prob.projections[1].synapse isa DualExpSynapse
    sol = solve(prob, FixedStep(0.1); v0 = (-70.0, -50.0))
    @test all(isfinite, sol.state.state.V)

    @test_throws ArgumentError spatial_fns(; tspan = (0.0, 100.0), dt = 0.1, synapse = :bogus)
    @test_throws ArgumentError spatial_fns(; rho = 1.0, dx = 0.01, tspan = (0.0, 100.0), dt = 0.1)  # ne < 1
end

@testset "spatial_fns: streaming external drive ≡ prescribed gext" begin
    # the streaming Poisson→dual-exp drive (default, O(N) memory) and the dense replayed `gext` realise
    # the SAME external connectome + Poisson raster for a given seed → numerically identical spikes. This
    # transitively validates the streaming drive against BrainPy (gext is the BrainPy-validated path).
    P = (; rho = 400.0, dx = 0.6, K_ee = 20, K_ei = 25, K_ie = 20, K_ii = 25,
        nu = 15.0, n_ext = 40, Delta_g_K = 0.005, tspan = (0.0, 300.0), dt = 0.1, seed = 20)
    prob_s = spatial_fns(; P..., external = :streaming)
    prob_p = spatial_fns(; P..., external = :prescribed)
    @test prob_s.projections[5].synapse isa PoissonSource
    @test prob_p.projections[5].synapse isa PrescribedCOBA

    sol_s = solve(prob_s, FixedStep(0.1); v0 = (-70.0, -50.0))
    sol_p = solve(prob_p, FixedStep(0.1); v0 = (-70.0, -50.0))
    @test sol_s.spike_count == sol_p.spike_count            # streaming ≡ prescribed, bit-for-bit
    @test sum(sol_s.spike_count) > 0

    @test_throws ArgumentError spatial_fns(; tspan = (0.0, 100.0), dt = 0.1, external = :bogus)
end
