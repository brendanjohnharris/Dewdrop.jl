# * Counter-based RNG
# A pure, stateless, counter-based generator: a draw keyed by (seed, step, entity)
# is a function of those values alone, so results are identical regardless of thread
# count or iteration order. This is the numerical rule that makes seeded runs
# reproducible and CPU/GPU-comparable; it must be fixed before any fixed-seed test.
# A counter-based RNG (Philox) is also the only RNG that parallelises cleanly across
# GPU threads (no shared mutable state, no per-thread stream bookkeeping).

import Random123: philox

# Golden-ratio odd constant; spreads the entity index across the RNG key space so
# that adjacent entities draw from well-separated streams.
const _RNG_MIX = 0x9e3779b97f4a7c15

# The Philox key for a draw: fold the entity index into the seed via the golden-ratio mix. Single
# source of truth so the uniform and normal streams can never drift apart.
@inline _rng_key(seed, entity) = (seed % UInt64) ⊻ ((entity % UInt64) * _RNG_MIX)

"""
    draw_uniform(T, seed, step, entity) -> T

A uniform draw in `[0, 1)` of float type `T`, a *pure* function of the global `seed`,
the time `step` (used as the counter), and the `entity` index (mixed into the key).
Identical for identical arguments regardless of thread or iteration order.
"""
# Call the FUNCTIONAL Philox directly rather than constructing a `Philox2x` generator: a
# freshly-seeded generator does a wasted extra round at counter (0,0) on construction, then
# `set_counter!` recomputes and the buffered `rand` adds dispatch: ~170× slower for the
# same bits. `philox((key,), (step, 0), Val(10))[1]` is bit-identical to that generator's
# first `UInt64` draw (verified over 10^5 inputs) and is a pure, allocation-free, side-effect-
# free function of (seed, step, entity): strictly better on GPU than the mutable generator.
@inline function draw_uniform(
        ::Type{T}, seed::Unsigned, step::Integer, entity::Integer
    ) where {T <: AbstractFloat}
    return draw_uniform(T, seed, step, entity, 0)
end

"""
    draw_uniform(T, seed, step, entity, batch) -> T

The ensemble-batched draw: an independent, bit-reproducible stream per `batch`. The Philox
counter is `(step, batch)`; the high counter word is unused by the 4-arg form (it is a
hard zero there), so folding `batch` into it yields B collision-free independent streams keyed
by `(seed, step, entity, batch)` on CPU and GPU. `batch = 0` reproduces the 4-arg bits exactly
(so the scalar B=1 path is bit-for-bit unchanged); `batch` must NOT be mixed into `entity`
(the golden-ratio key mix aliases there).
"""
@inline function draw_uniform(
        ::Type{T}, seed::Unsigned, step::Integer, entity::Integer, batch::Integer
    ) where {T <: AbstractFloat}
    key = _rng_key(seed, entity)
    x1, _ = philox((key,), (step % UInt64, batch % UInt64), Val(10))
    return _uniform(T, x1)
end

# Uniform draw in [0, 1) constructed directly from 64 random bits: set the mantissa to
# form a float in [1, 2) then subtract 1. This is dispatch-free (it avoids the Sampler
# machinery that `rand(rng, Float32)` routes through: a runtime dispatch that is also
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

A Poisson(λ) draw keyed by `(seed, step, entity)`: a pure function (one counter-based
uniform), identical across threads and iteration order. For per-neuron external drive.
"""
@inline draw_poisson(λ::Real, seed::Unsigned, step::Integer, entity::Integer) =
    poisson_count(λ, draw_uniform(Float64, seed, step, entity, 0))

"""
    draw_poisson(λ, seed, step, entity, batch) -> Int

The ensemble-batched Poisson draw: an independent reproducible stream per `batch` (see the
5-arg [`draw_uniform`](@ref)). `batch = 0` reproduces the 4-arg bits.
"""
@inline draw_poisson(λ::Real, seed::Unsigned, step::Integer, entity::Integer, batch::Integer) =
    poisson_count(λ, draw_uniform(Float64, seed, step, entity, batch))

"""
    draw_normal(T, seed, step, entity) -> T

A standard-normal `N(0, 1)` draw of float type `T`, a *pure* function of `(seed, step, entity)`
via the Box--Muller transform. It consumes BOTH 64-bit words of a single Philox evaluation (the
same call whose second word [`draw_uniform`](@ref) discards), so it costs one Philox eval per
draw and is allocation-free and GPU-kernel-safe (no Sampler dispatch, same discipline as
`_uniform`). Keyed identically to `draw_uniform`, so a distinct `seed` yields an independent
stream; for the SDE noise term, use a `seed` distinct from any Poisson drive. The `cos` branch
only (one normal per call), matching the one-draw-per-neuron-per-step shape of the drive.
"""
@inline function draw_normal(
        ::Type{T}, seed::Unsigned, step::Integer, entity::Integer
    ) where {T <: AbstractFloat}
    return draw_normal(T, seed, step, entity, 0)
end

"""
    draw_normal(T, seed, step, entity, batch) -> T

The ensemble-batched Gaussian: an independent, bit-reproducible stream per `batch` (the high
Philox counter word, as in the 5-arg [`draw_uniform`](@ref)). `batch = 0` reproduces the 4-arg
bits exactly.
"""
@inline function draw_normal(
        ::Type{T}, seed::Unsigned, step::Integer, entity::Integer, batch::Integer
    ) where {T <: AbstractFloat}
    key = _rng_key(seed, entity)
    x1, x2 = philox((key,), (step % UInt64, batch % UInt64), Val(10))   # BOTH words (x2 unused by draw_uniform)
    u1 = _uniform(T, x1)
    u2 = _uniform(T, x2)
    u1 = ifelse(iszero(u1), eps(T), u1)                # guard log(0): _uniform ∈ [0, 1) can be exactly 0
    return sqrt(T(-2) * log(u1)) * cos(T(2π) * u2)
end
