# * Unified stimulus seam. Every external input (constant current, Poisson voltage drive, OU membrane
# noise, and the new time-varying / functional / inhomogeneous families) is an `AbstractStimulus` applied
# at ONE point in the per-neuron step: `:current`/`:conductance` fold into `itot`/`gtot` at accumulate,
# `:kick` into `v` at deliver, `:noise` into `v` at the membrane step. A compile-time `stim_point` trait plus
# point-filtered tuple unrolls generate the Serial / fused / GPU / batched paths from ONE source (mirroring
# the synapse-descriptor collapse), and the fold of the three existing inputs stays BYTE-IDENTICAL. The
# `input=` / `drive=` / `noise=` kwargs remain the public sugar, lowering to `ConstantCurrent` / `PoissonDrive`
# / `WhiteNoise`. See .claude/docs/2026-07-08-unified-stimulus-design.md.

abstract type AbstractStimulus end
Base.Broadcast.broadcastable(s::AbstractStimulus) = Ref(s)

# The application point, a TYPE trait (isbits + JET-stable; never a runtime field).
stim_point(::Type{S}) where {S <: AbstractStimulus} = error("$(nameof(S)) must define stim_point (one of :current/:conductance/:kick/:noise)")

# Per-(neuron, step) context, isbits, rebuilt inside every kernel. The ONLY scalar-vs-batched delta is
# `b`/`stream` (scalar: b=1, stream=0 --- bit-identical to the 4-arg RNG; batched: b, streams[b]). `m` is the
# resolved per-neuron model (`:noise` needs `_tau`); `v` the membrane; `t = muladd(n,dt,t0)` the current time.
@inline stim_ctx(m, v, i, b, n, t, dt, stream) = (; m, v, i, b, n, t, dt, stream)

# The current step's simulation time, drift-free (`t0 = tend - nsteps*dt`, so no accumulated-float drift under
# Float32). Generic over the integrator (scalar and batched both carry n/dt/tend/nsteps).
@inline _step_time(integ) = muladd(integ.n, integ.dt, integ.tend - integ.nsteps * integ.dt)

# Per-neuron input reader: a shared scalar, a per-neuron vector, or a per-(neuron,member) matrix (b ignored by
# the first two). Shared by `ConstantCurrent` and the batched megakernel.
@inline _binputval(input::Number, i, b) = input
@inline _binputval(input::AbstractVector, i, b) = @inbounds input[i]
@inline _binputval(input::AbstractMatrix, i, b) = @inbounds input[i, b]

# ─────────────────────────── point-filtered tuple unrolls (compile-time, Base.tail) ───────────────────────────
# Each point sums only its own kind; non-matching stimuli compile to a no-op (so a wrong-kind stimulus never
# consumes an RNG counter at the wrong point). The strong-zero `false` is the additive identity: an all-absent
# unroll vanishes byte-identically (`v + false === v`, type-preserving), reproducing `_drive_kick(::Nothing)`.

# :kick → added to v at deliver
@inline _stim_kick(::Tuple{}, x) = false
@inline _stim_kick(t::Tuple, x) = _kick1(stim_point(typeof(first(t))), first(t), Base.tail(t), x)
@inline _kick1(::Val, s, r, x) = _stim_kick(r, x)
@inline _kick1(::Val{:kick}, s, r, x) = stim_kick(s, x) + _stim_kick(r, x)

# :noise → added to v_adv under the refractory gate
@inline _stim_noise(::Tuple{}, x) = false
@inline _stim_noise(t::Tuple, x) = _noise1(stim_point(typeof(first(t))), first(t), Base.tail(t), x)
@inline _noise1(::Val, s, r, x) = _stim_noise(r, x)
@inline _noise1(::Val{:noise}, s, r, x) = stim_noise(s, x) + _stim_noise(r, x)

# :current → the itot base. The FIRST :current stimulus ASSIGNS (byte-preserving, incl. signed zero), the rest
# ADD; with no :current stimulus the base is `z` (the gtot-typed zero == today's untouched accumulator).
@inline _stim_itot(::Tuple{}, z, x) = z
@inline _stim_itot(t::Tuple, z, x) = _cur0(stim_point(typeof(first(t))), first(t), Base.tail(t), z, x)
@inline _cur0(::Val, s, r, z, x) = _stim_itot(r, z, x)
@inline _cur0(::Val{:current}, s, r, z, x) = _curΣ(stim_current(s, x), r, x)
@inline _curΣ(acc, ::Tuple{}, x) = acc
@inline _curΣ(acc, r::Tuple, x) = _cur1(stim_point(typeof(first(r))), acc, first(r), Base.tail(r), x)
@inline _cur1(::Val, acc, s, r, x) = _curΣ(acc, r, x)
@inline _cur1(::Val{:current}, acc, s, r, x) = _curΣ(acc + stim_current(s, x), r, x)

# Fold a conductance (Δg or Δi) into an accumulator. The strong-zero `false` (an all-non-conductance unroll)
# is the IDENTITY: returns the accumulator UNTOUCHED, preserving a signed zero; unlike `x + false`, which
# flips -0.0 → +0.0. Keeps the conductance append a byte-identical no-op when no :conductance stimulus exists.
@inline _addcond(x, ::Bool) = x
@inline _addcond(x, Δ) = x + Δ

# :conductance → (Δg, Δi), COBA-shaped, appended after synapse accumulation (prescribed g(t) only; Poisson
# conductance is a synapse/projection). Summed with the `(false, false)` strong-zero identity.
@inline _stim_gtot(::Tuple{}, x) = (false, false)
@inline _stim_gtot(t::Tuple, x) = _cond1(stim_point(typeof(first(t))), first(t), Base.tail(t), x)
@inline _cond1(::Val, s, r, x) = _stim_gtot(r, x)
@inline function _cond1(::Val{:conductance}, s, r, x)
    (g, i) = stim_conductance(s, x)
    (gr, ir) = _stim_gtot(r, x)
    return (g + gr, i + ir)
end

# ─────────────────────────── serial (broadcast) drivers ───────────────────────────
# The Serial backend is CPU-only (GPU always uses the megakernel). Each driver reproduces the exact prior
# broadcast for the single-stimulus case; multiple same-point stimuli sum left-to-right. `m` is the scalar
# model (heterogeneous forces the fused path). Per-neuron contribution via a ctx built per broadcast element
# (isbits → allocation-free).
@inline _cur_bc(s, m, v, i, n, t, dt) = stim_current(s, stim_ctx(m, v, i, 1, n, t, dt, 0))
@inline _kick_bc(s, m, v, i, n, t, dt) = stim_kick(s, stim_ctx(m, v, i, 1, n, t, dt, 0))
@inline _noise_bc(s, m, v, i, n, t, dt) = stim_noise(s, stim_ctx(m, v, i, 1, n, t, dt, 0))

# itot base: first :current assigns, rest add; no :current → itot .= 0 (never happens: init always adds a
# ConstantCurrent, so the assign path always runs and reproduces `itot .= input`).
function _stim_itot!(itot, stimuli, m, n, t, dt)
    _sitot!(itot, stimuli, m, n, t, dt) || fill!(itot, zero(eltype(itot)))
    return nothing
end
@inline _sitot!(itot, ::Tuple{}, m, n, t, dt) = false
@inline function _sitot!(itot, stims::Tuple, m, n, t, dt)
    _sit1(stim_point(typeof(first(stims))), itot, first(stims), Base.tail(stims), m, n, t, dt)
end
@inline _sit1(::Val, itot, s, r, m, n, t, dt) = _sitot!(itot, r, m, n, t, dt)   # skip non-current
@inline function _sit1(::Val{:current}, itot, s, r, m, n, t, dt)
    idx = eachindex(itot)
    itot .= _cur_bc.(Ref(s), Ref(m), itot, idx, n, t, dt)                      # first :current ASSIGNS
    _sitot_add!(itot, r, m, n, t, dt)                                          # subsequent :current add
    return true
end
@inline _sitot_add!(itot, ::Tuple{}, m, n, t, dt) = nothing
@inline function _sitot_add!(itot, stims::Tuple, m, n, t, dt)
    if stim_point(typeof(first(stims))) === Val(:current)
        idx = eachindex(itot)
        itot .+= _cur_bc.(Ref(first(stims)), Ref(m), itot, idx, n, t, dt)
    end
    return _sitot_add!(itot, Base.tail(stims), m, n, t, dt)
end

# conductance stims: append (Δg, Δi) into gtot/itot after synapse accumulation (no-op unless :conductance).
function _stim_gtot!(gtot, itot, stimuli, V, m, n, t, dt)
    _sgtot!(gtot, itot, stimuli, V, m, n, t, dt)
    return nothing
end
@inline _sgtot!(gtot, itot, ::Tuple{}, V, m, n, t, dt) = nothing
@inline function _sgtot!(gtot, itot, stims::Tuple, V, m, n, t, dt)
    s = first(stims)
    if stim_point(typeof(s)) === Val(:conductance)
        idx = eachindex(gtot)
        gi = _cond_bc.(Ref(s), Ref(m), V, idx, n, t, dt)                       # (Δg, Δi) per neuron
        gtot .+= first.(gi)
        itot .+= last.(gi)
    end
    return _sgtot!(gtot, itot, Base.tail(stims), V, m, n, t, dt)
end
@inline _cond_bc(s, m, v, i, n, t, dt) = stim_conductance(s, stim_ctx(m, v, i, 1, n, t, dt, 0))

# :kick broadcast into V at deliver (per :kick stimulus). Reproduces `_apply_drive!` for a single PoissonDrive.
function _apply_kicks!(stimuli, V, m, n, t, dt)
    _akicks!(stimuli, V, m, n, t, dt)
    return nothing
end
@inline _akicks!(::Tuple{}, V, m, n, t, dt) = nothing
@inline function _akicks!(stims::Tuple, V, m, n, t, dt)
    s = first(stims)
    if stim_point(typeof(s)) === Val(:kick)
        idx = eachindex(V)
        V .+= _kick_bc.(Ref(s), Ref(m), V, idx, n, t, dt)
    end
    return _akicks!(Base.tail(stims), V, m, n, t, dt)
end

# :noise broadcast into V at the membrane step (refractory-gated), per :noise stimulus. Reproduces
# `_apply_noise!`: the increment is added only to non-refractory neurons; the draw itself is UNCONDITIONAL
# (evaluated for every neuron every step) so the counter-RNG never desyncs.
function _apply_noises!(stimuli, V, refrac, z, m, n, t, dt)
    _anoises!(stimuli, V, refrac, z, m, n, t, dt)
    return nothing
end
@inline _anoises!(::Tuple{}, V, refrac, z, m, n, t, dt) = nothing
@inline function _anoises!(stims::Tuple, V, refrac, z, m, n, t, dt)
    s = first(stims)
    if stim_point(typeof(s)) === Val(:noise)
        idx = eachindex(V)                                                     # Ref-wrap the (non-broadcast) stimulus + model
        V .= ifelse.(refrac .> z, V, V .+ _noise_bc.(Ref(s), Ref(m), V, idx, n, t, dt))
    end
    return _anoises!(Base.tail(stims), V, refrac, z, m, n, t, dt)
end

# ─────────────────────────── ConstantCurrent (the `input=` default) ───────────────────────────
"""
    ConstantCurrent(input)

A time-invariant external input current: a shared scalar or a per-neuron vector, applied to `itot`. This is
what `input =` on a [`DewdropNetwork`](@ref) lowers to; use it (in `stimuli =`) to combine a constant input
with other stimuli.
"""
struct ConstantCurrent{In} <: AbstractStimulus
    input::In
end
Adapt.@adapt_structure ConstantCurrent
stim_point(::Type{<:ConstantCurrent}) = Val(:current)
@inline stim_current(c::ConstantCurrent, x) = _binputval(c.input, x.i, x.b)
export ConstantCurrent

# ─────────────────────────── WhiteNoise gains the :noise point (struct in Noise.jl) ───────────────────────────
stim_point(::Type{<:WhiteNoise}) = Val(:noise)
@inline function stim_noise(w::WhiteNoise, x)
    s = _noise_scale(w, x.m, x.dt)                       # exact-OU scale (Noise.jl), unchanged
    return s * draw_normal(typeof(s), w.seed, x.n, x.i, x.stream)
end

# ─────────────────────────── init-time hooks: device upload + shape validation ───────────────────────────
# Two OPTIONAL per-stimulus hooks the engine applies once (upload at network construction, validate at init),
# defaulting to no-ops (the three legacy inputs and any isbits stimulus need neither): `stim_upload` moves a
# stimulus's backing arrays to the run architecture (mirrors `on_architecture(arch, input)`); `stim_validate`
# checks its shapes against the run's `(N, nsteps, dt)`. Applied across the extras tuple by the `_*_stimuli`.
stim_upload(s::AbstractStimulus, arch) = s
stim_validate(::AbstractStimulus, N, nsteps, dt) = nothing
@inline _upload_stimuli(t::Tuple, arch) = map(s -> stim_upload(s, arch), t)
_validate_stimuli(t::Tuple, N, nsteps, dt) = foreach(s -> stim_validate(s, N, nsteps, dt), t)

# ─────────────────────────── FunctionalCurrent / FunctionalKick / FunctionalConductance (live f) ───────────
# Normalise a user input function to the canonical (i, t) call form: a 2-arg `f(i, t)` passes through, a 1-arg
# `f(t)` is wrapped uniform over neurons. Arity is resolved ONCE (host-side, at construction), so the per-neuron
# call is a plain static dispatch; GPU-safe when the wrapped `f` is isbits (a bare function or a closure over
# isbits data). `_Uniform` is a struct (not a closure) so `Adapt` moves it cleanly onto the device.
struct _Uniform{F}
    f::F
end
@inline (u::_Uniform)(i, t) = u.f(t)
Adapt.@adapt_structure _Uniform
@inline _lift_it(f) = (!applicable(f, 1, oneunit(Float64)) && applicable(f, oneunit(Float64))) ? _Uniform(f) : f

"""
    FunctionalCurrent(f)

An external input current evaluated LIVE at every step from `f(t)` (space-uniform) or `f(i, t)` (per neuron
`i` at simulation time `t`), folded into `itot`. Pure and stateless; GPU-safe when `f` is isbits (a bare
function or a closure over isbits data). For recorded / tabulated signals use [`TimedArray`](@ref); for common
analytic shapes see [`ramp`](@ref) / [`step_input`](@ref) / [`sinusoid`](@ref) / [`pulses`](@ref).
"""
struct FunctionalCurrent{F} <: AbstractStimulus
    f::F
    FunctionalCurrent(f) = (g = _lift_it(f); new{typeof(g)}(g))   # lift f→(i,t) at construction (idempotent)
end
Adapt.@adapt_structure FunctionalCurrent
stim_point(::Type{<:FunctionalCurrent}) = Val(:current)
@inline stim_current(c::FunctionalCurrent, x) = c.f(x.i, x.t)
export FunctionalCurrent

"""
    FunctionalKick(f)

A live voltage kick `f(t)` / `f(i, t)` added straight to `v` each step (the deterministic analogue of
[`PoissonDrive`](@ref)); same call convention and GPU-safety as [`FunctionalCurrent`](@ref).
"""
struct FunctionalKick{F} <: AbstractStimulus
    f::F
    FunctionalKick(f) = (g = _lift_it(f); new{typeof(g)}(g))      # lift f→(i,t) at construction (idempotent)
end
Adapt.@adapt_structure FunctionalKick
stim_point(::Type{<:FunctionalKick}) = Val(:kick)
@inline stim_kick(c::FunctionalKick, x) = c.f(x.i, x.t)
export FunctionalKick

"""
    FunctionalConductance(f; Erev)

A live prescribed conductance `g = f(t)` / `f(i, t)` with reversal potential `Erev`, folded COBA-style into
the membrane (`g` into the effective leak, `g·Erev` into `itot`); an external, stateless conductance input
(distinct from a conductance SYNAPSE, which is a projection). GPU-safety as [`FunctionalCurrent`](@ref).
"""
struct FunctionalConductance{F, T} <: AbstractStimulus
    f::F
    Erev::T
end
FunctionalConductance(f; Erev) = (g = _lift_it(f); FunctionalConductance{typeof(g), typeof(Erev)}(g, Erev))
Adapt.@adapt_structure FunctionalConductance
stim_point(::Type{<:FunctionalConductance}) = Val(:conductance)
@inline stim_conductance(c::FunctionalConductance, x) = (g = c.f(x.i, x.t); (g, g * c.Erev))
export FunctionalConductance

# ─────────────────────────── TimedArray (tabulated, indexed by step) ───────────────────────────
"""
    TimedArray(data; as = :current)

A precomputed, time-varying input read live by step index: `data` is a length-`nsteps` vector (space-uniform)
or an `N × nsteps` matrix (per neuron), applied at the `as` point (`:current` → `itot`, `:kick` → `v`). Use
for recorded / tabulated signals; for closed-form signals prefer [`FunctionalCurrent`](@ref). A TimeseriesBase
`RegularTimeseries` (with `samplingperiod == dt`) can be passed directly, via the TimeseriesBase extension.
"""
struct TimedArray{P, A} <: AbstractStimulus
    data::A
end
TimedArray(data; as::Symbol = :current) = TimedArray{as, typeof(data)}(data)
Adapt.adapt_structure(to, ta::TimedArray{P}) where {P} = (d = adapt(to, ta.data); TimedArray{P, typeof(d)}(d))
stim_point(::Type{<:TimedArray{P}}) where {P} = Val(P)
@inline _timed_read(data::AbstractVector, i, n) = @inbounds data[n + 1]     # step n (0-based) → 1-based slot
@inline _timed_read(data::AbstractMatrix, i, n) = @inbounds data[i, n + 1]
@inline stim_current(ta::TimedArray{:current}, x) = _timed_read(ta.data, x.i, x.n)
@inline stim_kick(ta::TimedArray{:kick}, x) = _timed_read(ta.data, x.i, x.n)
stim_upload(ta::TimedArray{P}, arch) where {P} = (d = on_architecture(arch, ta.data); TimedArray{P, typeof(d)}(d))
function stim_validate(ta::TimedArray, N, nsteps, dt)
    L = size(ta.data)[end]
    L ≥ nsteps || throw(ArgumentError("TimedArray has $L time points < nsteps = $nsteps (need ≥ one value per step)"))
    ta.data isa AbstractMatrix && size(ta.data, 1) != N &&
        throw(ArgumentError("TimedArray is $(size(ta.data)) but N = $N (a matrix must be N × nsteps)"))
    return nothing
end
export TimedArray

# ─────────────────────────── InhomogeneousPoisson (:kick, per-neuron / time-varying rate) ───────────────────────────
# Resolve the instantaneous rate for neuron `i` at step `n` / time `t`: a shared scalar, a per-neuron vector
# `rate[i]`, an `N × nsteps` matrix `rate[i, n]`, or a live function `rate(t)` / `rate(i, t)` (lifted to (i,t)
# at construction). A per-neuron vector with zeros outside a subpopulation is the targeting mechanism.
@inline _rate_at(r::Number, i, n, t) = r
@inline _rate_at(r::AbstractVector, i, n, t) = @inbounds r[i]
@inline _rate_at(r::AbstractMatrix, i, n, t) = @inbounds r[i, n + 1]
@inline _rate_at(r, i, n, t) = r(i, t)                           # a lifted rate function

"""
    InhomogeneousPoisson(rate; weight, seed = 0)

A Poisson voltage drive whose rate varies in space and/or time (the generalisation of [`PoissonDrive`](@ref)).
`rate` (in Hz) is a shared scalar, a per-neuron vector `rate[i]`, an `N × nsteps` matrix `rate[i, n]`, or a
live function `rate(t)` / `rate(i, t)`; each step every neuron draws `weight · Poisson(rate · dt)` from the
counter RNG keyed by `(seed, step, neuron[, stream])` (independent per batch column). A per-neuron vector
that is zero outside a subpopulation targets the drive to that subpopulation.
"""
struct InhomogeneousPoisson{R, W} <: AbstractStimulus
    rate::R
    weight::W
    seed::UInt64
end
function InhomogeneousPoisson(rate; weight, seed = 0)
    r = rate isa Union{Number, AbstractArray} ? rate : _lift_it(rate)   # lift a function rate to (i,t)
    return InhomogeneousPoisson{typeof(r), typeof(weight)}(r, weight, UInt64(seed))
end
Adapt.@adapt_structure InhomogeneousPoisson
stim_point(::Type{<:InhomogeneousPoisson}) = Val(:kick)
@inline stim_kick(d::InhomogeneousPoisson, x) =
    d.weight * draw_poisson(_rate_at(d.rate, x.i, x.n, x.t) * x.dt, d.seed, x.n, x.i, x.stream)
stim_upload(d::InhomogeneousPoisson, arch) =
    d.rate isa AbstractArray ? InhomogeneousPoisson{typeof(on_architecture(arch, d.rate)), typeof(d.weight)}(on_architecture(arch, d.rate), d.weight, d.seed) : d
function stim_validate(d::InhomogeneousPoisson, N, nsteps, dt)
    d.rate isa AbstractVector && length(d.rate) != N &&
        throw(ArgumentError("InhomogeneousPoisson rate vector length $(length(d.rate)) ≠ N = $N"))
    d.rate isa AbstractMatrix && (size(d.rate, 1) != N || size(d.rate, 2) < nsteps) &&
        throw(ArgumentError("InhomogeneousPoisson rate matrix $(size(d.rate)) must be N × ≥nsteps = $N × $nsteps"))
    return nothing
end
export InhomogeneousPoisson

# ─────────────────────────── analytic input shapes (live, → FunctionalCurrent) ───────────────────────────
"""
    ramp(; t1, to, t0 = 0, from = 0) -> FunctionalCurrent

A linear ramp from `from` (at `t0`) to `to` (at `t1`), flat outside `[t0, t1]`.
"""
function ramp(; t1, to, t0 = 0.0, from = 0.0)
    t0, t1, from, to = promote(float(t0), float(t1), float(from), float(to))
    return FunctionalCurrent(t -> from + (to - from) * clamp((t - t0) / (t1 - t0), zero(t), oneunit(t)))
end

"""
    step_input(; amplitude, t0 = 0, base = 0) -> FunctionalCurrent

A step from `base` to `amplitude` at time `t0`.
"""
function step_input(; amplitude, t0 = 0.0, base = 0.0)
    amplitude, t0, base = promote(float(amplitude), float(t0), float(base))
    return FunctionalCurrent(t -> ifelse(t < t0, base, amplitude))
end

"""
    sinusoid(; amplitude, freq, phase = 0, offset = 0) -> FunctionalCurrent

A sinusoid `offset + amplitude·sin(2π·freq·t + phase)` (`freq` in Hz when `t` is in seconds).
"""
function sinusoid(; amplitude, freq, phase = 0.0, offset = 0.0)
    amplitude, freq, phase, offset = promote(float(amplitude), float(freq), float(phase), float(offset))
    return FunctionalCurrent(t -> offset + amplitude * sin(2 * oftype(t, π) * freq * t + phase))
end

"""
    pulses(; amplitude, period, width, t0 = 0, base = 0) -> FunctionalCurrent

A periodic train of rectangular pulses of height `amplitude` and duration `width`, one per `period`, starting
at `t0` (value `base` between pulses).
"""
function pulses(; amplitude, period, width, t0 = 0.0, base = 0.0)
    amplitude, period, width, t0, base = promote(float(amplitude), float(period), float(width), float(t0), float(base))
    return FunctionalCurrent(t -> ifelse(t ≥ t0 && mod(t - t0, period) < width, amplitude, base))
end
export ramp, step_input, sinusoid, pulses

# Assemble the integrator's ordered stimulus tuple from the legacy `input`/`drive`/`noise` fields plus any
# `stimuli =` extras. The `ConstantCurrent` base comes first (the itot seed), so it ASSIGNS; drive/noise and
# extras follow. Absent drive/noise contribute nothing (`nothing → ()`).
@inline _opt_stim(::Nothing) = ()
@inline _opt_stim(s) = (s,)
@inline _assemble_stimuli(input, drive, noise, extra = ()) =
    (ConstantCurrent(input), _opt_stim(drive)..., _opt_stim(noise)..., _stimtuple(extra)...)
@inline _stimtuple(::Nothing) = ()
@inline _stimtuple(s::AbstractStimulus) = (s,)
@inline _stimtuple(t) = Tuple(t)

export AbstractStimulus, stim_point
