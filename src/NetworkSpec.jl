# * Deferred network spec --- an immutable, run-parameter-free description of a network that
# materialises into a `DewdropNetwork` only at solve time, when `dt`/`tspan` are known. This lets a
# network be SPECIFIED without building its (expensive) connectome up front, supports constructors whose
# assembly genuinely needs `dt`/`tspan` (the spatial-FNS streaming drive), makes parameter sweeps cheap +
# serialisable ("define once, solve over many durations"), and is the future front-end for vmap-style
# batching (a batch is "B materialisations of B spec-variations"; `materialize` is the pure seam each mode
# consumes; the structured `FrozenBuilder` retains the per-projection recipes those modes need).
#
# Two representations under one interface:
#   FrozenBuilder   --- an immutable freeze of a `NetworkBuilder` (structured; retains projection recipes;
#                       carries a default `tspan`). Built via `freeze(nb)`; materialises via `_build_network`.
#   DeferredNetwork --- an opaque thunk: a captured build function + its kwargs (model params, seed, arch;
#                       NOT tspan/dt). Built via `defer(f; kw...)`; materialises by calling `f(; kw…, tspan, dt)`.
#                       Use for `dt`-dependent constructors (the deferred `f` must accept `tspan` + `dt`).
#
# The CommonSolve seam dispatches on the spec (no clash with `DewdropNetwork`): `init`/`solve` materialise,
# then delegate to the existing network `init`/`solve` --- so every kwarg (record/v0/backend/BATCH/progress/…)
# and the shared-CSR ensemble (`batch=B`) work through a spec on day one.

"""
    AbstractNetworkSpec

A deferred, run-parameter-free description of a network that materialises into a [`DewdropNetwork`](@ref)
only at solve time (when `dt`/`tspan` are known). Lets a network be specified without building its
connectome up front --- making parameter sweeps cheap and serialisable, and supporting constructors whose
assembly genuinely needs `dt`/`tspan`. Two concrete forms: a structured [`freeze`](@ref) of a
`NetworkBuilder`, or an opaque [`defer`](@ref)red constructor thunk; both are consumed by
[`materialize`](@ref) (and work directly with `solve`/`init`).
"""
abstract type AbstractNetworkSpec end
export AbstractNetworkSpec

# --- structured spec: a frozen builder -------------------------------------------------------------
struct FrozenBuilder{A, T, MV, IV, PV, PSV, DR} <: AbstractNetworkSpec
    arch::A
    tspan::Tuple{T, T}      # the default run window (overridable at solve)
    names::Vector{Symbol}
    models::MV
    sizes::Vector{Int}
    inputs::IV
    positions::PV
    projspecs::PSV          # Vector{_ProjSpec} --- the retained projection recipes
    drive::DR
end

"""
    freeze(nb::NetworkBuilder) -> FrozenBuilder

Freeze a [`NetworkBuilder`](@ref) into an immutable, reusable [`AbstractNetworkSpec`](@ref): a structured
network description that materialises into a `DewdropNetwork` at solve time (carrying the builder's `tspan`
as the default, overridable per solve). Retains the projection recipes, so it renders as a full tree and is
a clean source for future batching. A snapshot --- later mutation of `nb` does not affect the frozen spec.
"""
freeze(nb::NetworkBuilder) = FrozenBuilder(
    nb.arch, nb.tspan, copy(nb.names), copy(nb.models),
    copy(nb.sizes), copy(nb.inputs), copy(nb.positions), copy(nb.projspecs), nb.drive
)
export freeze

# --- thunk spec: a captured constructor + its kwargs ----------------------------------------------
struct DeferredNetwork{F, KW <: NamedTuple} <: AbstractNetworkSpec
    build::F        # build(; kw..., tspan, dt) -> DewdropNetwork
    kw::KW          # captured construction kwargs (NOT tspan/dt)
    label::Symbol   # for `show` (the constructor's name)
end

"""
    defer(f; kw...) -> DeferredNetwork

Capture a network constructor `f` and its construction `kw...` as a deferred [`AbstractNetworkSpec`](@ref),
materialised at solve time by calling `f(; kw..., tspan = …, dt = …)`. Makes any constructor (e.g. a
parametric network builder) a reusable, `tspan`-free spec with no refactor. `f` must accept `tspan` and `dt` keywords
(it may ignore `dt`); do NOT put `tspan`/`dt` in `kw` (they are injected at materialise time). `seed`/`arch`
belong in `kw` (they are part of the model's identity --- vary them by constructing a new spec).
"""
defer(f; kw...) = DeferredNetwork(f, NamedTuple(kw), nameof(f))
export defer

# --- materialise: the pure (spec, run-params) -> DewdropNetwork seam --------------------------------
"""
    materialize(spec::AbstractNetworkSpec, alg::FixedStep; tspan) -> DewdropNetwork

Build the concrete `DewdropNetwork` from a spec, given the run parameters (`dt` from `alg`, `tspan` as
shown). Pure: same spec + run-params → same network. A `FrozenBuilder` defaults `tspan` to the frozen
builder's; a `DeferredNetwork` has no default and requires it.
"""
function materialize end
export materialize

materialize(spec::FrozenBuilder, alg::FixedStep; tspan = nothing) =
    _build_network(
    spec.arch, tspan === nothing ? spec.tspan : tspan, spec.names, spec.models, spec.sizes,
    spec.inputs, spec.positions, spec.projspecs, spec.drive
)

function materialize(spec::DeferredNetwork, alg::FixedStep; tspan = nothing)
    tspan === nothing && throw(
        ArgumentError(
            "solving a deferred spec needs `tspan` (it captures no default); call " *
                "`solve(spec, FixedStep(dt); tspan = (t0, t1))`"
        )
    )
    return spec.build(; spec.kw..., tspan = tspan, dt = alg.dt)
end

# --- CommonSolve seam (dispatch on the spec; delegate to the materialised network) -----------------
CommonSolve.init(spec::AbstractNetworkSpec, alg::FixedStep; tspan = nothing, kwargs...) =
    init(materialize(spec, alg; tspan = tspan), alg; kwargs...)

# reuse the advisor-wrapped network `solve` (Advisor.jl) on the materialised network. The connectome build
# (`materialize`) can be slow and runs BEFORE the solve loop's progress bar, so announce it (see below).
function CommonSolve.solve(
        spec::AbstractNetworkSpec, alg::FixedStep; tspan = nothing, advise::Bool = true,
        progress = :auto, kwargs...
    )
    net = _materialize_announced(spec, alg; tspan = tspan, progress = progress)
    return solve(net, alg; advise = advise, progress = progress, kwargs...)
end

# Materialise the spec, announcing "Building network" while the (possibly slow) connectome build runs. It
# happens BEFORE the solve loop's bar, so without this the UI sits dead during the build. Emits an
# indeterminate ProgressLogging record (`progress = nothing`) on the SAME level/convention as the solve bar,
# under its own id, and clears it (`progress = "done"`) when the build finishes --- so a terminal/VSCode
# progress logger renders a "Building network…" spinner that gives way to the solve bar. `progress = false`
# suppresses it (matching the solve bar).
function _materialize_announced(spec::AbstractNetworkSpec, alg::FixedStep; tspan, progress)
    progress === false && return materialize(spec, alg; tspan = tspan)
    id = UUIDs.uuid4()
    @logmsg _PROGRESS_LEVEL "Building network" progress = nothing _id = id
    net = materialize(spec, alg; tspan = tspan)
    @logmsg _PROGRESS_LEVEL "Building network" progress = "done" _id = id
    return net
end

"""
    build(spec::AbstractNetworkSpec; dt, tspan = nothing) -> DewdropNetwork

Explicitly materialise a spec into a `DewdropNetwork` (the escape hatch for inspecting/reusing the built
network without solving). `dt` is required (it parameterises delay conversion for `dt`-dependent specs).
"""
build(spec::AbstractNetworkSpec; dt, tspan = nothing) = materialize(spec, FixedStep(dt); tspan = tspan)
