# * Multi-type populations: `MultiModel` holds an ordered tuple of (model, range)
# groups over one flat concatenated SoA, so a network can mix neuron model TYPES (e.g. AdEx
# excitatory + LIF inhibitory) in one engine. The groups partition `1:N` contiguously in
# declaration order.
#
# State is a UNION SoA: the SoA carries the union of every group's `statevars` (length `N` each), and
# a group's kernel touches only the columns its own model declares (e.g. `w` is allocated for all `N`
# but read only by the adaptation groups; the `_aux_col` seam returns `nothing` for a V-only group,
# so it keeps its byte-identical fast path). The waste is bounded; groups are few and overlap
# heavily on `:V`/`:refrac`.
#
# A `MultiModel` is `_is_hetero` (like `Heterogeneous`), so `init` routes it through the fused
# megakernel. The launch (Fused.jl) loops the groups, launching the SAME per-neuron kernel once per
# group over its range with that group's CONCRETE model and an index `offset`. Each launch is
# monomorphic, so it specialises exactly like the single-model kernel; and the homogeneous path
# (a bare model, or one group spanning `1:N`) is unchanged.

"""
    MultiModel(models, sizes)

A heterogeneous population: `models[g]` governs the `g`-th group of `sizes[g]` neurons, laid out
contiguously over `1:sum(sizes)` in order. Groups may use different neuron model TYPES (sharing one
float type). Built automatically by the [`network`](@ref) builder when populations of distinct
model types are added; addressable via the subpop registry (`sol[:E]`).
"""
struct MultiModel{MS <: Tuple, R <: Tuple} <: AbstractNeuronModel
    models::MS      # (modelE, modelI, …): a heterogeneous tuple of AbstractNeuronModels
    ranges::R       # (1:NE, NE+1:N, …): contiguous, covering 1:N, in declaration order
end
function MultiModel(models::AbstractVector, sizes::AbstractVector)
    isempty(models) && error("MultiModel needs at least one group")
    length(models) == length(sizes) || error("MultiModel: $(length(models)) models but $(length(sizes)) sizes")
    T = float_type(first(models))
    all(m -> float_type(m) === T, models) ||
        error("MultiModel: all groups must share one float type (got $(unique(float_type.(models))))")
    ranges = UnitRange{Int}[]
    off = 0
    for s in sizes
        push!(ranges, (off + 1):(off + Int(s)))
        off += Int(s)
    end
    return MultiModel(Tuple(models), Tuple(ranges))
end
export MultiModel
Adapt.@adapt_structure MultiModel                              # moves each group model (isbits) + the ranges
Base.Broadcast.broadcastable(m::MultiModel) = Ref(m)

# union of the groups' statevars, first-seen order (so :V, :refrac lead). Not in the inner loop:
# the SoA columns are built by the type-stable `Population(arch, ::MultiModel, N)` below.
function statevars(::Type{MM}) where {MM <: MultiModel}
    MS = MM.parameters[1]
    syms = Symbol[]
    for M in MS.parameters, s in statevars(M)
        s in syms || push!(syms, s)
    end
    return Tuple(syms)
end
float_type(mm::MultiModel) = float_type(first(mm.models))
@inline _is_hetero(::MultiModel) = true                       # → init routes it through the fused per-group launch
@inline _resting(mm::MultiModel) = _resting(first(mm.models)) # generic fallback; init fills V per group (see below)

# validate the group ranges cover 1:N contiguously (called from init, via the hetero check seam).
function _check_hetero(mm::MultiModel, N::Integer)
    expected = 1
    for r in mm.ranges
        first(r) == expected || error("MultiModel ranges must be contiguous from 1 (gap/overlap before $(first(r)))")
        expected = last(r) + 1
    end
    expected - 1 == N || error("MultiModel ranges cover $(expected - 1) neurons but N = $N")
    return nothing
end

# Union SoA construction: merge each group's zero-initialised statevar columns (later groups' shared
# columns overwrite, same zero value: a negligible init-time allocation). Each `_group_columns`
# call sees its model's `statevars` via constant propagation (`Val(statevars(typeof(m)))`), like the
# single-model `Population`, so the merged NamedTuple type is inferred and the StructArray is
# concretely typed (no @generated, so @neuron group models work too).
function Population(arch::AbstractArchitecture, mm::MultiModel, N::Integer)
    return Population(StructArray(_union_columns(arch, float_type(mm), Int(N), mm.models)))
end
@inline _union_columns(arch, ::Type{T}, N::Int, ::Tuple{}) where {T} = NamedTuple()
@inline function _union_columns(arch, ::Type{T}, N::Int, models::Tuple) where {T}
    this = _group_columns(arch, T, N, Val(statevars(typeof(first(models)))))
    return merge(this, _union_columns(arch, T, N, Base.tail(models)))
end
@inline function _group_columns(arch, ::Type{T}, N::Int, ::Val{names}) where {T, names}
    cols = ntuple(_ -> fill!(allocate(arch, T, N), zero(T)), Val(length(names)))
    return NamedTuple{names}(cols)
end

# per-group initial voltage: with no explicit v0, fill each group's V range with its own resting
# potential (groups may differ in EL); an explicit v0 (scalar / (lo,hi) / vector) applies over the
# whole flat population via the single-model logic. The scalar-model method is the identity wrapper,
# so the homogeneous path is byte-identical.
@inline _init_voltage_model!(V, model::AbstractNeuronModel, v0, ::Type{T}, seed) where {T} =
    _init_voltage!(V, T(_resting(model)), v0, T, seed)
function _init_voltage_model!(V, mm::MultiModel, ::Nothing, ::Type{T}, seed) where {T}
    for (m, r) in zip(mm.models, mm.ranges)
        fill!(view(V, r), T(_resting(m)))
    end
    return V
end
# An explicit v0 (scalar / (lo,hi) / vector) applies over the whole flat population via the generic
# `_init_voltage_model!(::AbstractNeuronModel, ...)` method above; `_resting(::MultiModel)` already
# returns `_resting(first(models))`, so no MultiModel-specific method is needed here.
