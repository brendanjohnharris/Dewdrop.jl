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

export SimBackend, Auto, Serial, Fused, Turbo

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
