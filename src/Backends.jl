# * Execution backends --- HOW the per-step engine runs, orthogonal to the CPU/GPU architecture
# (`arch`). The architecture chooses WHERE the arrays live; the backend chooses how the dense
# per-neuron phases of a step are executed. Pick one with `solve(prob, alg; backend = …)`; the
# default `Auto()` lets the advisor choose. All backends are numerically equivalent EXCEPT `Turbo`,
# which trades the exp ULP for SIMD speed (spike-identical, not bit-identical).

abstract type SimBackend end

"""
    Auto()

The default backend: the advisor picks the best available for the architecture, network size, and
loaded extensions --- `Fused` on a GPU, `Fused` for a large multithreaded CPU network or a
heterogeneous/multi-type model, otherwise `Serial`. Override explicitly with `Serial`/`Fused`/`Turbo`.
"""
struct Auto <: SimBackend end

"""
    Serial()

The per-phase broadcast engine: single-threaded dense phases, **bit-reproducible and allocation-free**.
The predictable baseline --- best for small networks, exact reproducibility, and debugging. (The
sparse scatter is still threaded when `Threads.nthreads() > 1`, as on every backend.)
"""
struct Serial <: SimBackend end

"""
    Fused()

One fused pass per step: on the CPU a tight, threaded plain-Julia loop over neurons; on a GPU the
KernelAbstractions megakernel. **Bit-identical** to `Serial` (same hooks, same order, same `libm`),
and faster at scale (it collapses the ~10 dense phases into a single pass and threads it). The
recommended performance backend; required for heterogeneous (`Heterogeneous`) and multi-type
(`MultiModel`) models.
"""
struct Fused <: SimBackend end

"""
    Turbo()

A SIMD-vectorised fused loop (via LoopVectorization). The **fastest CPU backend** --- it vectorises
the membrane `exp`, reaching compiled-C++ throughput. CPU-only; requires `using LoopVectorization`
and a model that provides a Turbo specialization (see the model-support table in the docs). **Not
bit-identical**: SIMD `exp` differs from scalar `libm` at the ULP level, so results are
spike-identical but not bit-reproducible (hence opt-in, never an `Auto` default).
"""
struct Turbo <: SimBackend end

"""
    Differentiable(; β = 10)

A surrogate-gradient execution backend that makes a CPU run **automatically differentiable**: it
replaces the discontinuous spike (hard threshold + reset + integer count) with a smooth fast-sigmoid
surrogate `s = 1 / (1 + exp(-β·(V - Vθ)))`, a soft reset `V ← V - s·(V - Vr)`, and a real-valued spike
accumulation. Everything else — the exact subthreshold propagator, the counter-based RNG — differentiates
as-is, so gradients of a scalar loss flow back to the model parameters through the whole time loop. Pair
it with any AD tool: `ForwardDiff` for a few parameters, `Enzyme` reverse-mode for many (e.g. weights).

`β` is the surrogate steepness (larger → closer to the true Heaviside, but stiffer gradients). The
gradients are *approximate* gradients of the true discrete dynamics (standard surrogate-gradient
training), so this is a distinct numerical path you opt into — every other backend is unaffected and
stays bit-identical. CPU-only and single-population for now; synaptic-weight training (a
surrogate-weighted scatter) and a GPU path are the documented next steps.
"""
struct Differentiable{T <: Real} <: SimBackend
    β::T
end
Differentiable(; β::Real = 10) = Differentiable(float(β))

export SimBackend, Auto, Serial, Fused, Turbo, Differentiable

# The differentiable backend accumulates a REAL-valued surrogate spike, so `spiked` / `spike_count`
# take the state float type instead of `Bool` / `Int`. Every other backend keeps `Bool` / `Int`, so the
# default allocation — and therefore every existing run — is byte-for-byte unchanged.
@inline _spiked_eltype(::SimBackend, ::Type{T}) where {T} = Bool
@inline _spiked_eltype(::Differentiable, ::Type{T}) where {T} = T
@inline _count_eltype(::SimBackend, ::Type{T}) where {T} = Int
@inline _count_eltype(::Differentiable, ::Type{T}) where {T} = T

# --- resolution: Auto → a concrete backend for this problem; explicit backends pass through ---
@inline _resolve_backend(b::SimBackend, prob) = b
function _resolve_backend(::Auto, prob)
    prob.arch isa GPU && return Fused()                       # the GPU path is the megakernel
    _is_hetero(prob.model) && return Fused()                  # Heterogeneous/MultiModel need the fused per-neuron path
    # Canonical CPU → the single-pass fused tight loop: bit-identical to Serial but ~2× the multi-pass
    # broadcast, AND work-aware --- it gates its own threading on neurons-per-thread, so it runs
    # single-threaded for small nets (no `@threads`-dispatch floor) and threads only once the work pays
    # for it. So it is the right pick at every size; Serial stays available for non-canonical schedules
    # (and explicitly, as the allocation-free reproducible baseline).
    prob.schedule == default_schedule() && return Fused()
    return Serial()
end

# back-compat: the old `step::Symbol` kwarg maps onto a backend.
_step_to_backend(::Nothing) = nothing
function _step_to_backend(step::Symbol)
    step === :auto && return Auto()
    step === :fused && return Fused()
    step === :serial && return Serial()
    step === :turbo && return Turbo()
    throw(ArgumentError("step must be :auto, :serial, :fused or :turbo (got :$step); prefer the `backend = …` keyword"))
end

# --- Turbo availability + per-model specialisation (populated by the LoopVectorization extension) ---
# `turbo_kernel(M)` returns a SIMD step function for model type `M`, or `nothing`. A model gains a
# Turbo specialization when the extension (or a user) defines a method; `supports_turbo` reports it.
turbo_kernel(::Type) = nothing
@inline supports_turbo(::Type{M}) where {M} = turbo_kernel(M) !== nothing
# the extension is loaded iff `Base.get_extension` resolves it (returns `nothing` when unloaded).
_turbo_available() = Base.get_extension(@__MODULE__, :DewdropLoopVectorizationExt) !== nothing

# validation of a resolved backend against the problem (clear errors over a late MethodError).
function _check_backend(b::SimBackend, prob)
    if b isa Serial && _is_hetero(prob.model)
        throw(ArgumentError("Heterogeneous / MultiModel models require `backend = Fused()` (or Auto); the per-phase Serial engine cannot resolve a per-neuron model"))
    end
    if b isa Turbo
        _turbo_available() ||
            throw(ArgumentError("backend = Turbo() requires `using LoopVectorization` (loads the DewdropLoopVectorizationExt extension)"))
        prob.arch isa GPU && throw(ArgumentError("backend = Turbo() is CPU-only; use Fused() on the GPU"))
        prob.schedule == default_schedule() ||
            throw(ArgumentError("backend = Turbo() requires the canonical schedule"))
        prob.noise === nothing ||
            throw(ArgumentError("backend = Turbo() does not support WhiteNoise (the SIMD kernel is deterministic); use Fused()"))
        supports_turbo(typeof(prob.model)) ||
            throw(ArgumentError("model $(typeof(prob.model)) has no Turbo specialization (define `Dewdrop.turbo_kernel(::Type{$(nameof(typeof(prob.model)))})`, or use Fused())"))
    end
    return nothing
end

# The dense membrane accumulators `:gtot`/`:itot` are materialised into `integ.gtot`/`integ.itot` ONLY
# by the Serial broadcast and Turbo paths (`_accum_all!`). The fused tight loop / GPU megakernel compute
# them per-neuron IN-KERNEL and never write them back, so a `Trace(:gtot)`/`Trace(:itot)` there would
# silently record zeros (the batched path errors on this too, see Batch.jl). Reject it with a clear
# pointer rather than return wrong data. Synaptic state (`Trace(:g_decay; projection = i)`) is
# materialised on every backend, so it is the portable way to record a conductance.
@inline _populates_accum(b::SimBackend) = (b isa Serial) || (b isa Turbo)
function _check_accum_record(bk::SimBackend, record)
    (record === nothing || _populates_accum(bk)) && return nothing
    for spec in values(record)
        if spec isa Trace && spec.projection === nothing && spec.var in (:gtot, :itot)
            throw(ArgumentError(
                "Trace(:$(spec.var)) requires `backend = Serial()` (or Turbo): the fused/GPU step " *
                "computes the $(spec.var) accumulator per-neuron in-kernel and never materialises it, " *
                "so recording it here would silently return zeros. Pass `backend = Serial()`, or record " *
                "a synaptic variable instead, e.g. `Trace(:g_decay; projection = i)`."))
        end
    end
    return nothing
end

function _check_backend(b::Differentiable, prob)
    prob.arch isa GPU &&
        throw(ArgumentError("backend = Differentiable() is CPU-only for now (a GPU surrogate-AD path is the documented next step); run the forward pass on the GPU with Fused()"))
    prob.schedule == default_schedule() ||
        throw(ArgumentError("backend = Differentiable() requires the canonical schedule"))
    isempty(prob.projections) ||
        throw(ArgumentError("backend = Differentiable() currently supports unconnected populations only; synaptic-weight training needs a surrogate-weighted scatter (the next step)"))
    _is_hetero(prob.model) &&
        throw(ArgumentError("backend = Differentiable() does not yet support Heterogeneous / MultiModel models"))
    return nothing
end
