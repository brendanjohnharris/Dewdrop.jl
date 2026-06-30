# * Network builder (M2 + Phase A) --- a small, fluent, mutating helper for constructing networks
# with named subpopulations. The builder accumulates an ordered list of populations
# (`population!`), projections (`project!`), and an external drive (`drive!`), then `build`
# assembles them into a flat `DewdropNetwork`: the populations are concatenated into one SoA in
# declaration order, their ranges recorded in a subpop registry (`:E`, `:I`, …), and the per-group
# models merged. Same-type groups that differ only in parameter values collapse to one
# `Heterogeneous` model (block per-neuron arrays); a single shared model stays a bare model (the
# homogeneous fast path, byte-identical to a hand-built `DewdropNetwork`). Different model TYPES per
# group need the MultiModel engine (Phase B) and currently error.
#
# The accumulated populations/projections are heterogeneous (any model, any CUBA/COBA/delta
# synapse), so `build` is the single dynamic boundary --- it materialises everything into concretely
# typed tuples via a function barrier, keeping the simulation hot loop fully type-stable.

# a deferred projection: `src => dst` over a synapse, with the connectivity built at `build` time
# (when the final registry + positions are known). Either `fixed_prob` (`p`), `distance_prob`
# (`kernel` + positions), or a prebuilt `connectivity`.
struct _ProjSpec
    src::Symbol
    dst::Symbol
    synapse::Any
    kw::Any            # NamedTuple of connectivity kwargs (p/weight/delay/seed/allow_self/kernel/period/connectivity)
    plasticity::Any
end

"""
    NetworkBuilder

A mutable accumulator for a network with named subpopulations (see [`network`](@ref),
[`population!`](@ref), [`project!`](@ref), [`drive!`](@ref), [`build`](@ref)).
"""
mutable struct NetworkBuilder{A <: AbstractArchitecture, T}
    const arch::A
    const tspan::Tuple{T, T}
    const names::Vector{Symbol}
    const models::Vector{Any}
    const sizes::Vector{Int}
    const inputs::Vector{Any}
    const positions::Vector{Any}      # per-population positions vector or `nothing`
    const projspecs::Vector{_ProjSpec}
    drive::Any
end

"""
    network(; arch=CPU(), tspan) -> NetworkBuilder
    network(model, NE, NI; arch=CPU(), tspan) -> NetworkBuilder

Begin building a network. The keyword form starts empty --- add named populations with
[`population!`](@ref). The 3-argument form is sugar for the common E/I case: it adds an `:E`
population of `NE` and an `:I` population of `NI`, both of `model` (one shared model → the
homogeneous fast path).
"""
function network(; arch::AbstractArchitecture = CPU(), tspan)
    T = float_type_of_tspan(tspan)
    return NetworkBuilder(arch, (T(to_time(tspan[1])), T(to_time(tspan[2]))),
        Symbol[], Any[], Int[], Any[], Any[], _ProjSpec[], nothing)
end
function network(model::AbstractNeuronModel, NE::Integer, NI::Integer;
        arch::AbstractArchitecture = CPU(), tspan)
    nb = network(; arch = arch, tspan = tspan)
    population!(nb, :E, model, NE)
    population!(nb, :I, model, NI)
    return nb
end
export network

# the builder's float type from `tspan` (units stripped); falls back to Float64 for plain numbers.
float_type_of_tspan(tspan) = typeof(to_time(tspan[1]) + to_time(tspan[2]) + 0.0)

"""
    population!(nb, name, model, N; input=0.0, positions=nothing) -> nb

Add a named population of `N` `model` neurons. Populations are concatenated in declaration order;
`name` (e.g. `:E`) addresses the population in `project!` and on the solution (`sol[name]`).
`input` is a per-population constant current (scalar or length-`N` vector); `positions` (optional)
are consumed by distance-kernel projections.
"""
function population!(nb::NetworkBuilder, name::Symbol, model::AbstractNeuronModel, N::Integer;
        input = 0.0, positions = nothing)
    name === :all && error("population name :all is reserved (it always denotes the whole network)")
    name in nb.names && error("population :$name already defined")
    push!(nb.names, name)
    push!(nb.models, model)
    push!(nb.sizes, Int(N))
    push!(nb.inputs, input)
    push!(nb.positions, positions)
    return nb
end
export population!

# the registry of subpop ranges from population names + sizes (declaration order).
function _registry(names, sizes)
    ranges = Pair{Symbol, UnitRange{Int}}[]
    off = 0
    for (nm, sz) in zip(names, sizes)
        push!(ranges, nm => (off + 1):(off + sz))
        off += sz
    end
    return NamedTuple(ranges)
end
_registry(nb::NetworkBuilder) = _registry(nb.names, nb.sizes)

"""
    project!(nb, src => dst, synapse; p, weight, delay, seed, allow_self=false, plasticity=nothing) -> nb
    project!(nb, src, synapse; …) -> nb

Add a projection from subpopulation `src` onto `dst` (a `src => dst` pair of subpop names; the
single-symbol form targets the whole network, `src => :all`). Connectivity is `fixed_prob` with
probability `p` by default, `distance_prob` if a `kernel` is given (using the populations'
`positions`), or a prebuilt `connectivity`. `weight`/`delay` may be scalars or per-source functions
of the presynaptic index; `delay` is a physical time in **milliseconds** (resolved to integer steps at
the solve `dt`), or use `steps(n)` for an explicit step count. An optional `adjust = conn -> …` hook
runs over the materialised connectivity (e.g. to rescale/correlate weights). (Named `project!` rather
than `connect!` to avoid clashing with `Observables.connect!`, which `Makie` re-exports.)
"""
function project!(nb::NetworkBuilder, pair::Pair{Symbol, Symbol}, synapse::AbstractSynapseModel;
        plasticity = nothing, kw...)
    push!(nb.projspecs, _ProjSpec(pair.first, pair.second, synapse, NamedTuple(kw), plasticity))
    return nb
end
project!(nb::NetworkBuilder, src::Symbol, synapse::AbstractSynapseModel; kw...) =
    project!(nb, src => :all, synapse; kw...)
export project!

"""
    drive!(nb, drive) -> nb
    drive!(nb, target, drive) -> nb

Set the external [`PoissonDrive`](@ref) for the network. The 3-argument form names a `target`
subpopulation (currently only `:all`, the whole network).
"""
function drive!(nb::NetworkBuilder, drive)
    nb.drive = drive
    return nb
end
function drive!(nb::NetworkBuilder, target::Symbol, drive)
    target === :all || error("targeted drive (:$target) is not yet supported; use :all (the whole network)")
    nb.drive = drive
    return nb
end

"""
    drive!(nb, target, synapse; rate, n_ext, p, weight=1.0, delay=1.0, seed, adjust=nothing) -> nb

Attach a streaming external Poisson drive to the `target` sub-population: `n_ext` virtual Poisson sources
(rate `rate` Hz) wired to `target` by `fixed_prob(p)`, delivering through any `synapse` model (the
postsynaptic kinetics). Appended as a drive projection --- a [`PoissonSource`](@ref) over an empty outer
CSR --- materialised at [`build`](@ref) once `target`'s global range is known. `adjust` (e.g.
[`correlate_weights`](@ref)) optionally rescales the external weights.
"""
function drive!(nb::NetworkBuilder, target::Symbol, synapse::AbstractSynapseModel;
        rate, n_ext::Integer, p::Real, weight = 1.0, delay = 1.0, seed = 0x9e3779b97f4a7c15,
        fire_seed = nothing, adjust = nothing, index_type::Type = Int)
    # `seed` keys the wiring (source→target connectome). `fire_seed` keys the per-step Poisson firing; default
    # `nothing` → reuse `seed` (so a lone drive is unchanged). Give two drives the SAME `fire_seed` with
    # DIFFERENT `seed` to make them a shared common-mode source (the same external spikes, independent fan-out).
    push!(nb.projspecs, _ProjSpec(:__poisson__, target, synapse,
        (; rate = rate, n_ext = Int(n_ext), p = p, weight = weight, delay = delay,
            seed = seed % UInt64, fire_seed = fire_seed === nothing ? nothing : (fire_seed % UInt64),
            adjust = adjust, index_type = index_type), nothing))
    return nb
end
export drive!

# --- model merge ------------------------------------------------------------------------------
# Merge the per-group models into one engine model. A single group, or several groups with the
# identical model, stays a bare model (homogeneous fast path). Same-type groups that differ in some
# field values collapse to a `Heterogeneous` whose overridden fields are block per-neuron arrays.
# Different model TYPES become a `MultiModel` (per-group launches over the union SoA).
function _combine_models(models::Vector, sizes::Vector{Int}, arch::AbstractArchitecture)
    M = typeof(first(models))
    all(m -> typeof(m) === M, models) || return on_architecture(arch, MultiModel(models, sizes))   # different TYPES → MultiModel
    length(models) == 1 && return first(models)
    overrides = Pair{Symbol, Any}[]
    for f in fieldnames(M)
        vals = [getfield(m, f) for m in models]
        all(==(first(vals)), vals) && continue                  # identical across groups → stays scalar
        arr = reduce(vcat, [fill(getfield(models[g], f), sizes[g]) for g in eachindex(models)])
        push!(overrides, f => arr)
    end
    isempty(overrides) && return first(models)                  # all fields identical → homogeneous
    # the per-neuron override arrays are read on-device by the fused kernel, so move the assembled model onto
    # `arch` (`Heterogeneous` is `@adapt_structure`d → override arrays become device arrays; a no-op on CPU).
    return on_architecture(arch, Heterogeneous(first(models); NamedTuple(overrides)...))
end

# --- input merge ------------------------------------------------------------------------------
# A single scalar if every group shares the same scalar input; otherwise a length-N block vector.
function _combine_inputs(inputs::Vector, sizes::Vector{Int}, ::Type{T}) where {T}
    if all(x -> x isa Number, inputs) && all(==(first(inputs)), inputs)
        return first(inputs)
    end
    blocks = map(eachindex(inputs)) do g
        x = inputs[g]
        x isa Number && return fill(T(to_current(x)), sizes[g])
        length(x) == sizes[g] || error("population $(g) input length $(length(x)) ≠ N = $(sizes[g])")
        return T.(to_current.(x))
    end
    return reduce(vcat, blocks)
end

# concatenate per-group positions (all-or-nothing); `nothing` if no population gave positions.
function _combine_positions(positions::Vector, sizes::Vector{Int})
    all(isnothing, positions) && return nothing
    any(isnothing, positions) && error("positions must be given for all populations or none")
    return reduce(vcat, positions)
end

# build one projection's connectivity from its spec, resolving src/dst names against the registry.
function _build_projection(spec::_ProjSpec, reg::NamedTuple, positions, arch, N::Int)
    spec.src === :__poisson__ && return _build_drive(spec, reg, arch, N)   # targeted Poisson drive
    sources = _subrange(reg, spec.src)
    targets = _subrange(reg, spec.dst)
    kw = spec.kw
    it = get(kw, :index_type, Int)   # opt-in narrow connectome indices (e.g. Int32 halves the scatter bandwidth)
    # Build the connectome on the HOST (the top-k heap is host-side regardless of `arch`), run any `adjust`
    # weight hook host-side, then move the finished CSR onto `arch` at the end (the Projection below) --- so
    # adjusters like `correlate_weights` never touch a device array, with no host↔device round-trip.
    conn = if haskey(kw, :connectivity) && kw.connectivity !== nothing
        kw.connectivity
    elseif haskey(kw, :kernel) && kw.kernel !== nothing
        positions === nothing && error("distance-kernel projection :$(spec.src)=>:$(spec.dst) needs population `positions`")
        if haskey(kw, :count) && kw.count !== nothing
            distance_fixed_count(CPU(), positions; kernel = kw.kernel, count = kw.count, weight = kw.weight,
                delay = kw.delay, seed = kw.seed, allow_self = get(kw, :allow_self, false),
                period = get(kw, :period, nothing), sources = sources, targets = targets, index_type = it)
        else
            distance_prob(CPU(), positions; kernel = kw.kernel, weight = kw.weight, delay = kw.delay,
                seed = kw.seed, allow_self = get(kw, :allow_self, false),
                period = get(kw, :period, nothing), sources = sources, targets = targets, index_type = it)
        end
    else
        fixed_prob(CPU(), N, N, kw.p; weight = kw.weight, delay = kw.delay, seed = kw.seed,
            allow_self = get(kw, :allow_self, false), sources = sources, targets = targets, index_type = it)
    end
    # optional post-build hook: `project!(…; adjust = conn -> …)` runs over the materialised connectivity
    # (e.g. in-degree-scaled / spatially-correlated weights). A 2-arg adjuster `(conn, ctx)` additionally
    # receives the projection's resolved `(; sources, targets)` ranges --- e.g. `correlate_weights`, which
    # normalises the in-degree over the destination sub-population.
    adj = get(kw, :adjust, nothing)
    adj === nothing || _apply_adjust(adj, conn, sources, targets)
    return Projection(spec.synapse, on_architecture(arch, conn); plasticity = spec.plasticity)   # host → device
end

# Materialise a targeted Poisson drive spec (`drive!(nb, target, synapse; …)`, marked `src = :__poisson__`)
# into a drive projection: build the `n_ext × N` external connectome into the destination sub-population's
# global range, optionally rescale its weights (`adjust`), and wrap a `PoissonSource` over the chosen synapse
# with an empty outer CSR (so the network scatter is a no-op; the wiring lives inside the source).
function _build_drive(spec::_ProjSpec, reg::NamedTuple, arch, N::Int)
    rng = _subrange(reg, spec.dst)
    kw = spec.kw
    wseed = kw.seed ⊻ 0x243f6a8885a308d3     # wiring stream, independent of the per-step firing stream
    extconn = fixed_prob(CPU(), kw.n_ext, N, kw.p; weight = kw.weight, delay = kw.delay,   # build host-side
        seed = wseed, sources = 1:(kw.n_ext), targets = rng, index_type = get(kw, :index_type, Int))
    kw.adjust === nothing || _apply_adjust(kw.adjust, extconn, 1:(kw.n_ext), rng)           # adjust host-side
    extconn = on_architecture(arch, extconn)                                                # then move onto the device
    fseed = kw.fire_seed === nothing ? kw.seed : kw.fire_seed                               # firing stream (shareable)
    return Projection(PoissonSource(spec.synapse, extconn; rate = kw.rate, seed = fseed), _empty_csr(arch, N))
end

# Apply the post-build `adjust` hook, accepting either a 2-arg `(conn, ctx)` adjuster (given the resolved
# `(; sources, targets)` ranges) or a plain 1-arg `conn ->` adjuster.
_apply_adjust(adj, conn, sources, targets) =
    applicable(adj, conn, (; sources, targets)) ? adj(conn, (; sources, targets)) : adj(conn)

"""
    build(nb; input=nothing, schedule=default_schedule()) -> DewdropNetwork

Assemble the builder into a [`DewdropNetwork`](@ref): concatenate the populations into one flat SoA
(recording their ranges in the subpop registry), merge the per-group models and inputs, and
materialise the projections. `input` overrides the per-population inputs with a single global value.
"""
# The single assembly path, shared by `build(::NetworkBuilder)` and `materialize(::FrozenBuilder)` (the
# deferred spec; see NetworkSpec.jl). Concatenates the populations, merges models/inputs, materialises the
# projections, and records the named-projection labels --- parameterised by `tspan` (so a frozen spec can
# override the builder's default duration). The dynamic boundary (heterogeneous models/projspecs) is here.
function _build_network(arch, tspan, names, models, sizes, inputs, positions_in, projspecs, drive;
        input = nothing, schedule::Schedule = default_schedule())
    isempty(names) && error("network has no populations --- add them with `population!`")
    N = sum(sizes)
    reg = merge((all = 1:N,), _registry(names, sizes))   # include the implicit :all so projections can target it
    model = _combine_models(models, sizes, arch)
    T = float_type(model)
    in_ = input === nothing ? _combine_inputs(inputs, sizes, T) : input
    positions = _combine_positions(positions_in, sizes)
    projs = Tuple(_build_projection(spec, reg, positions, arch, N) for spec in projspecs)
    labels = isempty(projspecs) ? nothing :
        Tuple((spec.src === :__poisson__ ? :poisson : spec.src) => spec.dst for spec in projspecs)
    return DewdropNetwork(model, N; input = in_, tspan = tspan, arch = arch,
        schedule = schedule, projections = projs, drive = drive, subpops = reg, positions = positions,
        projlabels = labels)
end

build(nb::NetworkBuilder; input = nothing, schedule::Schedule = default_schedule()) =
    _build_network(nb.arch, nb.tspan, nb.names, nb.models, nb.sizes, nb.inputs, nb.positions, nb.projspecs,
        nb.drive; input = input, schedule = schedule)
export build
