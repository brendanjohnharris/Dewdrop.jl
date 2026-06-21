# * Network builder (M2) --- a small, fluent, mutating helper for constructing E/I networks:
# one population of NE+NI neurons with named `:E` (1:NE) and `:I` (NE+1:end) subpopulations,
# projections added by `connect!`, an external drive by `drive!`, assembled into a
# `DewdropNetwork` by `build`. This removes the manual `fixed_prob` / `sources` / `Projection`
# boilerplate. The accumulated projections are heterogeneous (CUBA/COBA/delta), so `build`
# materialises them into a concretely-typed tuple via a function barrier (the builder boundary
# is the only dynamic point; the simulation hot loop remains fully type-stable).

"""
    NetworkBuilder

A mutable accumulator for an E/I network (see [`network`](@ref), [`project!`](@ref),
[`drive!`](@ref), [`build`](@ref)).
"""
mutable struct NetworkBuilder{M <: AbstractNeuronModel, A <: AbstractArchitecture, T}
    const model::M
    const NE::Int
    const NI::Int
    const arch::A
    const tspan::Tuple{T, T}
    const projections::Vector{Any}
    drive::Any
end

"""
    network(model, NE, NI; arch=CPU(), tspan) -> NetworkBuilder

Begin building an E/I network of `NE` excitatory + `NI` inhibitory `model` neurons (a single
population with named `:E` = `1:NE` and `:I` = `NE+1:NE+NI` subpopulations).
"""
function network(model::AbstractNeuronModel, NE::Integer, NI::Integer;
        arch::AbstractArchitecture = CPU(), tspan)
    T = float_type(model)
    return NetworkBuilder(model, Int(NE), Int(NI), arch, (T(tspan[1]), T(tspan[2])), Any[], nothing)
end
export network

function _subpop(nb::NetworkBuilder, src::Symbol)
    src === :E && return 1:nb.NE
    src === :I && return (nb.NE + 1):(nb.NE + nb.NI)
    src === :all && return 1:(nb.NE + nb.NI)
    return error("unknown subpopulation $src --- use :E, :I, or :all")
end

"""
    project!(nb, src, synapse; p, weight, delay, seed, allow_self=false) -> nb

Add a projection from subpopulation `src` (`:E`, `:I`, or `:all`) onto the whole population,
with random connection probability `p`. `weight` and `delay` may be scalars or per-source
functions of the presynaptic index. (Named `project!` rather than `connect!` to avoid clashing
with `Observables.connect!`, which `Makie` re-exports.)
"""
function project!(nb::NetworkBuilder, src::Symbol, synapse::AbstractSynapseModel;
        p, weight, delay, seed::Unsigned, allow_self::Bool = false)
    N = nb.NE + nb.NI
    conn = fixed_prob(nb.arch, N, N, p; weight = weight, delay = delay, seed = seed,
        sources = _subpop(nb, src), allow_self = allow_self)
    push!(nb.projections, Projection(synapse, conn))
    return nb
end
export project!

"""
    drive!(nb, drive) -> nb

Set the external [`PoissonDrive`](@ref) for the network.
"""
function drive!(nb::NetworkBuilder, drive)
    nb.drive = drive
    return nb
end
export drive!

"""
    build(nb; input=0.0, schedule=default_schedule()) -> DewdropNetwork

Assemble the builder into a [`DewdropNetwork`](@ref) problem.
"""
function build(nb::NetworkBuilder; input = 0.0, schedule::Schedule = default_schedule())
    return DewdropNetwork(nb.model, nb.NE + nb.NI; input = input, tspan = nb.tspan,
        arch = nb.arch, schedule = schedule, projections = Tuple(nb.projections), drive = nb.drive)
end
export build
