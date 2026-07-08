# Plotting front-end. These names are declared here as stubs (so they exist and export before any
# Makie backend is loaded) and given methods by the weak-dependency `TimeseriesMakieExt`, activated by
# `using TimeseriesMakie` alongside a Makie backend (e.g. CairoMakie). The base neural recipes
# (`spikeraster`/`psth`/`ratemap`, plus `traces`/`trajectory`) live in TimeseriesMakie.jl; a
# `DewdropSolution` plots directly through them (`plot(sol)` rasters). See docs/src/guide/plotting.md.

"""
    traceplot(sol, name = :V; of = :all, kwargs...)
    traceplot!(ax, sol, name = :V; of = :all, kwargs...)

Recorded state traces of a [`Trace`](@ref) monitor (`name` is the record key) as stacked lines, one
per unit, over time; `of` restricts to a named subpopulation. Reuses TimeseriesMakie's `traces`
recipe. Requires a Makie backend and `TimeseriesMakie` loaded.
"""
function traceplot end
function traceplot! end
export traceplot, traceplot!

"""
    phaseplane(sol; vars = (:V, :w), neuron = 1, kwargs...)
    phaseplane!(ax, sol; vars = (:V, :w), neuron = 1, kwargs...)

Phase-plane trajectory of one `neuron`'s two recorded state variables (`vars`, each a [`Trace`](@ref)
record key): e.g. the `(V, w)` plane of an [`AdEx`](@ref) unit. Reuses TimeseriesMakie's `trajectory`
recipe. Requires a Makie backend and `TimeseriesMakie` loaded.
"""
function phaseplane end
function phaseplane! end
export phaseplane, phaseplane!

"""
    positionplot(sol; color = :rate, kwargs...)
    positionplot!(ax, sol; color = :rate, kwargs...)

Scatter the per-neuron positions of a spatial network (2-D or 3-D), colored by `color`: `:rate`
(firing rate), `:type` (named subpopulation index), a per-neuron vector, or a fixed color. Requires a
Makie backend and `TimeseriesMakie` loaded.
"""
function positionplot end
function positionplot! end
export positionplot, positionplot!

"""
    connectivity(x; kwargs...)
    connectivity!(ax, x; kwargs...)

Heatmap of a connection-weight matrix densified from a [`SparseCSR`](@ref), a [`Projection`](@ref),
or a [`DewdropNetwork`](@ref) (`post × pre`). Large connectomes are block-mean binned to keep the
image bounded. Requires a Makie backend and `TimeseriesMakie` loaded.
"""
function connectivity end
function connectivity! end
export connectivity, connectivity!
