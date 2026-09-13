# * Per-neuron heterogeneous parameters: `Heterogeneous(base; field = array, …)` wraps a
# scalar neuron model and overrides chosen parameters with per-neuron arrays. Storage is a frozen
# array (computed once at construction, read many times), not a procedural in-kernel draw: a per-neuron
# parameter is time-constant, so recomputing it every step would be pure waste; an array is general
# (block E/I, parametric distributions, or loaded data), trivially reproducible, and Adapt-movable.
# Reproducible distributions are obtained by filling the array via the counter-based RNG (`per_neuron`).
#
# Mechanism: the engine resolves a per-neuron scalar base model in the step loop; `_resolve(h, i)`
# rebuilds `base` with the i-th value of each overridden field (ConstructionBase.setproperties, isbits
# in/out → allocation-free + GPU-safe). The existing model hooks then run unchanged on the resolved
# model. A heterogeneous model routes through the fused megakernel (which has the per-neuron index),
# so the broadcast/threshold/reset phases are untouched and every homogeneous path stays bit-identical
# (`_resolve(m, i) = m` for a scalar model).

"""
    Heterogeneous(base; field = array, …)

Wrap a scalar neuron `base` model, overriding the named parameter `field`s with per-neuron arrays
(each length `N`, the population size). Fields not overridden keep the scalar value. Use for E/I
populations with different parameters, or any per-neuron heterogeneity. Fill arrays reproducibly with
[`per_neuron`](@ref) + the counter RNG. Requires the canonical schedule (runs via the fused megakernel).

As one group of a [`MultiModel`](@ref) the arrays still span the whole population (length `N`, indexed
by the flat neuron index), not just the group's range; entries outside the group are never read.
"""
struct Heterogeneous{M <: AbstractNeuronModel, NT <: NamedTuple} <: AbstractNeuronModel
    base::M
    params::NT      # per-neuron override arrays, keyed by base field name (e.g. (Vθ = […], b = […]))
end
function Heterogeneous(base::AbstractNeuronModel; kw...)
    nt = NamedTuple(kw)
    isempty(nt) && error("Heterogeneous: pass at least one per-neuron parameter array")
    for k in keys(nt)
        hasfield(typeof(base), k) || error("Heterogeneous: `$k` is not a field of $(typeof(base))")
        nt[k] isa AbstractVector || error("Heterogeneous: override `$k` must be a per-neuron vector")
    end
    return Heterogeneous(base, nt)
end
export Heterogeneous
Adapt.@adapt_structure Heterogeneous                          # moves the override arrays to the device
Base.Broadcast.broadcastable(h::Heterogeneous) = Ref(h)

statevars(::Type{Heterogeneous{M, NT}}) where {M, NT} = statevars(M)
float_type(h::Heterogeneous) = float_type(h.base)
@inline _is_hetero(::Heterogeneous) = true                    # → init routes it through the fused kernel
@inline _resting(h::Heterogeneous) = _resting(h.base)         # default initial V uses the base's EL

# the per-neuron scalar model: rebuild `base` with the i-th value of each overridden field, via the
# base's positional constructor (`@generated` so it specialises per (base type, override keys) and
# inlines to a plain isbits constructor call: allocation-free + GPU-kernel-safe). Generic
# `_resolve(m, i) = m` (in Adaptation.jl) keeps scalar models bit-identical and zero-cost.
@generated function _resolve(h::Heterogeneous{M, NT}, i) where {M, NT}
    overkeys = NT.parameters[1]                              # the overridden field names
    args = map(fieldnames(M)) do f
        f in overkeys ? :(@inbounds(h.params.$f[i])) : :(getfield(h.base, $(QuoteNode(f))))
    end
    return :($(M)($(args...)))
end

# Default initial V reads each neuron's resolved model, so an overridden `EL` (or `VL`) sets that
# neuron's resting potential instead of every neuron starting at the base scalar. One fused broadcast
# (`_resolve` is the same GPU-safe generated function the fused kernel uses); an explicit `v0` takes
# the generic path.
@inline _resting_of(h::Heterogeneous, i) = _resting(_resolve(h, i))
# `offset` shifts the read into the flat population index. As a `MultiModel` group this model's `V`
# slice is a view over the group's range while its override arrays span the whole population (the
# indexing the fused kernel uses), so the slice must be read at its global position.
function _init_voltage_model!(V, h::Heterogeneous, ::Nothing, ::Type{T}, seed, offset::Int = 0) where {T}
    idx = reshape((1 + offset):(size(V, 1) + offset), :, ntuple(_ -> 1, ndims(V) - 1)...)
    V .= T.(_resting_of.(h, idx))
    return V
end
# A plain scalar model carries no override arrays to validate; the seam exists so a `MultiModel` can
# check its groups uniformly.
_check_hetero(::AbstractNeuronModel, N::Integer) = nothing
# validate the override arrays against the population size (called from init)
function _check_hetero(h::Heterogeneous, N::Integer)
    for (k, a) in pairs(h.params)
        length(a) == N || throw(ArgumentError("Heterogeneous override `$k` has length $(length(a)) but N = $N"))
    end
    return nothing
end

"""
    per_neuron(f, N) -> Vector

Materialise a per-neuron parameter array `[f(i) for i in 1:N]`. For a reproducible distribution, draw
from the counter RNG, e.g. `per_neuron(i -> μ + σ*draw_normal(Float64, seed, 0, i), N)`; for block E/I,
`vcat(fill(xE, NE), fill(xI, NI))`.
"""
per_neuron(f, N::Integer) = [f(i) for i in 1:Int(N)]
export per_neuron
