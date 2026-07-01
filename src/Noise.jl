# * SDE noise: additive-voltage white noise on the membrane, integrated by the EXACT
# Ornstein--Uhlenbeck discretization (the stochastic analogue of the engine's exact drift
# propagator). A `WhiteNoise` is attached to the network like a `PoissonDrive`; it is optional and
# compiles away when absent (the `Nothing` strong-zero idiom shared with `drive`/`compaction`).
# The per-step increment is `s · ξ`, ξ ~ N(0, 1) from the counter-based `draw_normal` keyed by
# (seed, step, neuron), with `s` the exact-OU scale below; so the discretized membrane carries
# the analytically correct stationary variance σ²τ/2 at any dt, not merely as dt → 0.

"""
    WhiteNoise(σ; seed = 0)

An additive white-noise source on the membrane potential: each step every neuron's `V` gains an
independent Gaussian increment realising the Ornstein--Uhlenbeck diffusion of intensity `σ` (the
subthreshold stationary variance is `σ²τ/2`). Drawn reproducibly from the counter-based RNG keyed
by `(seed, step, neuron)`: identical across runs, threads and devices. Use a `seed` distinct
from any [`PoissonDrive`](@ref).
"""
struct WhiteNoise{T}
    σ::T
    seed::UInt64
end
WhiteNoise(σ::Real; seed = 0) = WhiteNoise(float(σ), UInt64(seed))
export WhiteNoise

# The membrane time constant the model relaxes with: the OU rate that sets the noise scaling.
# Defined per model; the `@neuron` macro and the adaptation models add their own methods.
@inline _tau(m::LIF) = m.τ

# Exact-OU per-step noise scale. For a linear membrane advanced by the exact propagator
# `V ← V∞ + (V−V∞)e^{−dt/τ}`, the increment `s·ξ` with `s = σ·√((τ/2)(1−e^{−2dt/τ}))` gives the
# discrete process the EXACT stationary variance σ²τ/2 at any dt (the literal Euler--Maruyama
# scale `σ√dt` is correct only as dt → 0). `-expm1(-2dt/τ) = 1 − e^{−2dt/τ}`, accurate at small dt.
@inline function _noise_scale(noise::WhiteNoise, m, dt)
    τ = _tau(m)
    σ = oftype(dt, noise.σ)
    return σ * sqrt((τ / 2) * (-expm1(-2 * dt / τ)))
end
