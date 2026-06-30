# * SoA population state
# Per-unit state is a struct-of-arrays of an `isbits` element type, allocated
# through the architecture seam, so a population is contiguous and coalescing on
# CPU and GPU alike and is movable to a device via `Adapt`.

"""
    Population(arch, T, names, N)

A population of `N` units whose per-unit state is a struct-of-arrays
([`StructArray`](@ref)) with one column per symbol in `names`, element type `T`,
allocated through `arch` and zero-initialised. The SoA element type is `isbits`, so
the population is movable to a device via `Adapt`.

The number of units is `length(pop)`; columns are accessed as `pop.state.<name>`.
"""
struct Population{S}
    state::S
end
Adapt.@adapt_structure Population
export Population

function _soa_columns(
        arch::AbstractArchitecture, ::Type{T}, names::NTuple{K, Symbol}, N::Integer
    ) where {T, K}
    cols = ntuple(_ -> fill!(allocate(arch, T, N), zero(T)), K)
    return NamedTuple{names}(cols)
end

function Population(
        arch::AbstractArchitecture, ::Type{T}, names::NTuple{K, Symbol}, N::Integer
    ) where {T, K}
    return Population(StructArray(_soa_columns(arch, T, names, N)))
end

Base.length(pop::Population) = length(pop.state)
