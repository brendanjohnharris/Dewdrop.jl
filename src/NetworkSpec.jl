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
freeze(nb::NetworkBuilder) = FrozenBuilder(nb.arch, nb.tspan, copy(nb.names), copy(nb.models),
    copy(nb.sizes), copy(nb.inputs), copy(nb.positions), copy(nb.projspecs), nb.drive)
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
materialised at solve time by calling `f(; kw..., tspan = …, dt = …)`. Makes any constructor (e.g.
`spatial_fns`) a reusable, `tspan`-free spec with no refactor. `f` must accept `tspan` and `dt` keywords
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
    _build_network(spec.arch, tspan === nothing ? spec.tspan : tspan, spec.names, spec.models, spec.sizes,
        spec.inputs, spec.positions, spec.projspecs, spec.drive)

function materialize(spec::DeferredNetwork, alg::FixedStep; tspan = nothing)
    tspan === nothing && throw(ArgumentError(
        "solving a deferred spec needs `tspan` (it captures no default); call " *
        "`solve(spec, FixedStep(dt); tspan = (t0, t1))`"))
    return spec.build(; spec.kw..., tspan = tspan, dt = alg.dt)
end

# --- CommonSolve seam (dispatch on the spec; delegate to the materialised network) -----------------
CommonSolve.init(spec::AbstractNetworkSpec, alg::FixedStep; tspan = nothing, kwargs...) =
    init(materialize(spec, alg; tspan = tspan), alg; kwargs...)

# reuse the advisor-wrapped network `solve` (Advisor.jl) on the materialised network.
CommonSolve.solve(spec::AbstractNetworkSpec, alg::FixedStep; tspan = nothing, advise::Bool = true, kwargs...) =
    solve(materialize(spec, alg; tspan = tspan), alg; advise = advise, kwargs...)

"""
    build(spec::AbstractNetworkSpec; dt, tspan = nothing) -> DewdropNetwork

Explicitly materialise a spec into a `DewdropNetwork` (the escape hatch for inspecting/reusing the built
network without solving). `dt` is required (it parameterises delay conversion for `dt`-dependent specs).
"""
build(spec::AbstractNetworkSpec; dt, tspan = nothing) = materialize(spec, FixedStep(dt); tspan = tspan)
