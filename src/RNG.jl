# * Counter-based RNG (M0 contract 4)
# A pure, stateless, counter-based generator: a draw keyed by (seed, step, entity)
# is a function of those values alone, so results are identical regardless of thread
# count or iteration order. This is the numerics contract that makes seeded runs
# reproducible and CPU/GPU-comparable; it must be fixed before any golden-seed test.
# A counter-based RNG (Philox) is also the only RNG that parallelises cleanly across
# GPU threads (no shared mutable state, no per-thread stream bookkeeping).

import Random123: Philox2x, set_counter!

# Golden-ratio odd constant; spreads the entity index across the RNG key space so
# that adjacent entities draw from well-separated streams.
const _RNG_MIX = 0x9e3779b97f4a7c15

"""
    draw_uniform(T, seed, step, entity) -> T

A uniform draw in `[0, 1)` of float type `T`, a *pure* function of the global `seed`,
the time `step` (used as the counter), and the `entity` index (mixed into the key).
Identical for identical arguments regardless of thread or iteration order.
"""
# NOTE: `Philox2x` is intentionally *mutable*; per-call construction is allocation-free
# only because escape analysis (SROA) keeps the non-escaping generator in registers (the
# zero-allocation guard in test/rng.jl pins this). Do NOT "optimise" this into a reused
# stateful generator with `set_counter!`: that introduces persistent mutable per-thread
# state which, on GPU, lands in local memory and hurts occupancy --- for no benefit.
@inline function draw_uniform(
        ::Type{T}, seed::Unsigned, step::Integer, entity::Integer
    ) where {T <: AbstractFloat}
    key = (seed % UInt64) ⊻ ((entity % UInt64) * _RNG_MIX)
    rng = Philox2x(UInt64, key)
    set_counter!(rng, step % UInt64)
    return _uniform(T, rand(rng, UInt64))
end

# Uniform draw in [0, 1) constructed directly from 64 random bits: set the mantissa to
# form a float in [1, 2) then subtract 1. This is dispatch-free (it avoids the Sampler
# machinery that `rand(rng, Float32)` routes through --- a runtime dispatch that is also
# hostile to GPU kernels) and range-correct at any width.
@inline _uniform(::Type{Float64}, u::UInt64) =
    reinterpret(Float64, (u >> 12) | 0x3ff0000000000000) - 1.0
@inline _uniform(::Type{Float32}, u::UInt64) =
    reinterpret(Float32, ((u >> 41) % UInt32) | 0x3f800000) - 1.0f0

"""
    poisson_count(λ, u) -> Int

Number of events of a Poisson(λ) variate, by inverse-CDF inversion of a single uniform
`u ∈ [0, 1)`. One uniform per sample (no rejection loop over multiple draws), so it is pure
and GPU-kernel-safe. A guard caps the search for pathological inputs.
"""
@inline function poisson_count(λ::Real, u::Real)
    p = exp(-λ)            # P(0)
    cdf = p
    k = 0
    while u > cdf && k < 1000
        k += 1
        p *= λ / k          # P(k) = P(k-1)·λ/k
        cdf += p
    end
    return k
end

"""
    draw_poisson(λ, seed, step, entity) -> Int

A Poisson(λ) draw keyed by `(seed, step, entity)` --- a pure function (one counter-based
uniform), identical across threads and iteration order. For per-neuron external drive.
"""
@inline draw_poisson(λ::Real, seed::Unsigned, step::Integer, entity::Integer) =
    poisson_count(λ, draw_uniform(Float64, seed, step, entity))
