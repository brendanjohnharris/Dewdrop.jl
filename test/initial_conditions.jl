using Dewdrop
using Test
using Statistics

# Randomized / explicit initial membrane potential (`v0` kwarg to `init`/`solve`).
# `nothing` keeps the synchronous EL default; a `(lo, hi)` tuple draws per-neuron uniform
# initial conditions from the counter-based RNG (deterministic, reproducible) which breaks
# the initial synchrony that otherwise traps strongly-driven balanced networks.
@testset "randomized initial conditions (v0)" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    prob = DewdropNetwork(m, 200; input = 0.0, tspan = (0.0, 1.0))

    # default: synchronous EL
    @test all(==(0.0), init(prob, FixedStep(0.1)).state.state.V)

    # scalar clamp
    @test all(==(5.0), init(prob, FixedStep(0.1); v0 = 5.0).state.state.V)

    # uniform-random in [lo, hi): in-range and genuinely spread
    V = init(prob, FixedStep(0.1); v0 = (10.0, 20.0)).state.state.V
    @test all(v -> 10.0 ≤ v < 20.0, V)
    @test std(V) > 1.0

    # reproducible (same seed) and seed-sensitive (different seed)
    @test V == init(prob, FixedStep(0.1); v0 = (10.0, 20.0)).state.state.V
    @test V != init(prob, FixedStep(0.1); v0 = (10.0, 20.0), v0_seed = UInt64(99)).state.state.V

    # explicit vector copied verbatim
    vv = collect(range(0.0, 19.0; length = 200))
    @test init(prob, FixedStep(0.1); v0 = vv).state.state.V == vv

    # forwards through `solve` and the run still completes
    sol = solve(prob, FixedStep(0.1); v0 = (10.0, 20.0))
    @test sol.nsteps == 10
end
