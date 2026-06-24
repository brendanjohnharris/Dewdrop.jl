# * Beautiful, hierarchical REPL rendering. Every Dewdrop object renders to reflect the structure it
# actually has: a leaf model (neuron / synapse) as a flat aligned parameter sheet; a composite model
# (MultiModel / Heterogeneous), a network, a builder, or a solution as an indented tree whose nodes
# are the real structural units. Purely host-side presentation --- never in a kernel, never on the
# hot path. Dispatch specificity does the routing: a composite's own `show` is more specific than the
# generic leaf method, so the tree intercepts automatically.
#
# Two forms per type: the rich `show(io, ::MIME"text/plain", x)` (the REPL result) and the compact
# `show(io, x)` (one line --- inline, array elements, and how a parent renders a child).

# --- styling: subtle colour, ALWAYS gated on `get(io, :color, false)` (plain when piped/logged) ---
@inline _color(io::IO) = get(io, :color, false)::Bool
function _styled(io::IO, s, kind::Symbol)
    if _color(io)
        kind === :type    ? printstyled(io, s; color = :light_black) :
        kind === :field   ? printstyled(io, s; color = :cyan) :
        kind === :unit    ? printstyled(io, s; color = :light_black) :
        kind === :tree    ? printstyled(io, s; color = :light_black) :
        kind === :section ? printstyled(io, s; bold = true) :
        kind === :dim     ? printstyled(io, s; color = :light_black) :
        print(io, s)
    else
        print(io, s)
    end
    return nothing
end

# the type head: base name in default weight, the `{T}` parameter suffix dimmed (so `LIF{Float64}`
# reads as `LIF` with a faint type tag). Used for the leaf models + synapses (one float param).
function _typehead(io::IO, x)
    T = typeof(x)
    print(io, nameof(T))
    ps = T.parameters
    isempty(ps) || _styled(io, "{" * join(ps, ", ") * "}", :type)
    return nothing
end

# clean float/number formatting (shortest round-trip; ASCII minus → copy-pasteable).
_fmt(x::Real) = string(x)
_fmt(x) = string(x)

# thousands separators for edge counts etc.
function _commas(n::Integer)
    s = string(abs(n))
    parts = String[]
    while length(s) > 3
        pushfirst!(parts, s[(end - 2):end]); s = s[1:(end - 3)]
    end
    pushfirst!(parts, s)
    return (n < 0 ? "-" : "") * join(parts, ",")
end

# host-resident? (avoid triggering a device reduction at show-time for extrema summaries)
_onhost(x::AbstractArray) = x isa Array

# --- units: the canonical display unit for a field, where the dimension is known (built-ins only) ---
# `_field_dims(::Type)` defaults to `nothing` (e.g. @neuron models) → bare numbers, no invented units.
_field_dims(::Type) = nothing
_unit_of(dim::Symbol) =
    dim === :time        ? "ms" :
    dim === :voltage     ? "mV" :
    dim === :conductance ? "nS" :
    dim === :current     ? "pA" :
    dim === :capacitance ? "pF" :
    dim === :resistance  ? "GΩ" :
    dim === :rate        ? "kHz" : ""
@inline _unit_for(::Nothing, ::Symbol) = nothing
@inline _unit_for(dims::NamedTuple, f::Symbol) = haskey(dims, f) ? _unit_of(getfield(dims, f)) : nothing

# the canonical-dimension tables (mirror exactly what each constructor declares in src/Units.jl terms)
_field_dims(::Type{<:LIF}) = (τ = :time, EL = :voltage, Vθ = :voltage, Vr = :voltage, R = :resistance, tref = :time)
_field_dims(::Type{<:AdaptLIF}) = (τ = :time, EL = :voltage, Vθ = :voltage, Vr = :voltage, R = :resistance,
    tref = :time, a = :conductance, b = :current, τw = :time)
_field_dims(::Type{<:AdEx}) = (C = :capacitance, gL = :conductance, EL = :voltage, VT = :voltage, ΔT = :voltage,
    Vr = :voltage, Vpeak = :voltage, a = :conductance, b = :current, τw = :time, tref = :time)
_field_dims(::Type{<:FNSNeuron}) = (C = :capacitance, gL = :conductance, VL = :voltage, VK = :voltage,
    Vθ = :voltage, Vr = :voltage, tref = :time, τK = :time, ΔgK = :conductance)
_field_dims(::Type{<:CurrentSynapse}) = (τ = :time,)
_field_dims(::Type{<:ConductanceSynapse}) = (τ = :time, Erev = :voltage)
_field_dims(::Type{<:DualExpSynapse}) = (τr = :time, τd = :time, Erev = :voltage)

# the aligned `name = value unit` parameter block (shared by neuron + synapse leaves). Each line is
# newline-prefixed so the block appends after a header with no trailing newline.
function _show_params(io::IO, m)
    fns = fieldnames(typeof(m))
    isempty(fns) && return nothing
    dims = _field_dims(typeof(m))
    namew = maximum(f -> length(string(f)), fns)
    vals = String[_fmt(getfield(m, f)) for f in fns]
    valw = maximum(length, vals)
    for (f, v) in zip(fns, vals)
        print(io, "\n  ")
        _styled(io, rpad(string(f), namew), :field)
        print(io, " = ", lpad(v, valw))
        u = _unit_for(dims, f)
        u === nothing || (print(io, " "); _styled(io, u, :unit))
    end
    return nothing
end

# the compact `Name{T}(f1=v1, f2=v2, …)` parameter list (≤4 fields shown).
function _compact_params(io::IO, m)
    fns = fieldnames(typeof(m))
    print(io, "(")
    k = min(length(fns), 4)
    for i in 1:k
        i > 1 && print(io, ", ")
        print(io, fns[i], "=", _fmt(getfield(m, fns[i])))
    end
    length(fns) > k && print(io, ", …")
    print(io, ")")
    return nothing
end

_oneline(x) = sprint(show, x)     # a child's compact render, colour-free, for embedding in a tree head

# render a node's children as a tree. Each child is `(head::String, subchildren::Vector)`; leaves have
# an empty subchildren vector. Newline-prefixed (no trailing newline), nesting via `│ `/`  ` continuation.
function _print_tree(io::IO, children::AbstractVector; prefix::String = "")
    n = length(children)
    for (i, child) in enumerate(children)
        head, subs = child
        last = i == n
        print(io, "\n")
        _styled(io, prefix * (last ? "└─ " : "├─ "), :tree)
        print(io, head)
        isempty(subs) || _print_tree(io, subs; prefix = prefix * (last ? "   " : "│  "))
    end
    return nothing
end

# === neuron model leaves (LIF / AdaptLIF / AdEx / FNSNeuron / @neuron) ===
function Base.show(io::IO, ::MIME"text/plain", m::AbstractNeuronModel)
    get(io, :compact, false) && return show(io, m)
    _typehead(io, m)
    _show_params(io, m)
    print(io, "\n  ")
    _styled(io, "state: ", :section)
    print(io, join(statevars(typeof(m)), ", "))
    return nothing
end
function Base.show(io::IO, m::AbstractNeuronModel)
    _typehead(io, m)
    _compact_params(io, m)
    return nothing
end

# === synapse model leaves ===
_synkind(::CurrentSynapse) = "CUBA"
_synkind(::ConductanceSynapse) = "COBA"
_synkind(::DualExpSynapse) = "COBA"
_synkind(::DeltaSynapse) = "delta"
function Base.show(io::IO, ::MIME"text/plain", s::AbstractSynapseModel)
    get(io, :compact, false) && return show(io, s)
    _typehead(io, s)
    print(io, " ")
    _styled(io, "· " * _synkind(s), :dim)
    _show_params(io, s)
    return nothing
end
function Base.show(io::IO, s::AbstractSynapseModel)
    _typehead(io, s)
    isempty(fieldnames(typeof(s))) || _compact_params(io, s)
    return nothing
end

# A PoissonSource (streaming Poisson drive) renders as its driving statistics --- rate + #sources over the
# wrapped synapse --- not a field dump of the inner synapse + extconn CSR. It inherits the inner synapse's
# kind tag (COBA / CUBA / delta / …).
_synkind(s::PoissonSource) = _synkind(s.synapse)
function Base.show(io::IO, ::MIME"text/plain", s::PoissonSource)
    get(io, :compact, false) && return show(io, s)
    kind = _synkind(s)
    print(io, "PoissonSource ")
    _styled(io, "→ " * string(nameof(typeof(s.synapse))) * (isempty(kind) ? "" : " · " * kind), :dim)
    print(io, "\n  ")
    _styled(io, rpad("rate", 5), :field)
    print(io, " = ", _fmt(s.rate), " ")
    _styled(io, "Hz", :unit)
    print(io, "\n  ")
    _styled(io, rpad("n_ext", 5), :field)
    print(io, " = ", _fmt(npre(s.extconn)))
    return nothing
end
Base.show(io::IO, s::PoissonSource) =
    print(io, "PoissonSource(", _fmt(s.rate), " Hz → ", nameof(typeof(s.synapse)), ")")

# === MultiModel: one tree row per group (range + group model) ===
_total_N(mm::MultiModel) = isempty(mm.ranges) ? 0 : last(last(mm.ranges))
function Base.show(io::IO, ::MIME"text/plain", mm::MultiModel)
    get(io, :compact, false) && return show(io, mm)
    print(io, "MultiModel")
    _styled(io, "{$(float_type(mm))}", :type)
    print(io, " · ", length(mm.models), " groups · N=", _total_N(mm))
    rngs = String["[$(first(r)):$(last(r))]" for r in mm.ranges]
    rw = maximum(length, rngs; init = 0)
    children = Any[(string(rpad(rngs[g], rw), "  ", typeof(mm.models[g])), Any[]) for g in eachindex(mm.models)]
    _print_tree(io, children)
    return nothing
end
Base.show(io::IO, mm::MultiModel) = print(io, "MultiModel(", length(mm.models), " groups, N=", _total_N(mm), ")")

# === Heterogeneous: base model + per-neuron override summaries (never dumps the arrays) ===
function _override_summary(kstr::AbstractString, v::AbstractVector)
    head = string(kstr, "  ", length(v), "-element Vector{", eltype(v), "}")
    if _onhost(v) && !isempty(v)
        lo, hi = extrema(v)
        head *= " ⟨$(_fmt(lo)) … $(_fmt(hi))⟩"
    end
    return head
end
function Base.show(io::IO, ::MIME"text/plain", h::Heterogeneous)
    get(io, :compact, false) && return show(io, h)
    print(io, "Heterogeneous")
    _styled(io, "{$(typeof(h.base))}", :type)
    print(io, " · ")
    _styled(io, "per-neuron", :dim)
    kw = maximum(k -> length(string(k)), keys(h.params); init = 0)
    overrides = Any[(_override_summary(rpad(string(k), kw), v), Any[]) for (k, v) in pairs(h.params)]
    children = Any[("base  " * _oneline(h.base), Any[]), ("overrides", overrides)]
    _print_tree(io, children)
    return nothing
end
Base.show(io::IO, h::Heterogeneous) =
    print(io, "Heterogeneous(", nameof(typeof(h.base)), ", ", length(h.params), " per-neuron)")

# === connectivity ===
function _conn_summary(io::IO, c::SparseCSR)
    print(io, "SparseCSR · ", npre(c), "→", npost(c), " · ", _commas(nedges(c)), " edges")
    if _onhost(c.weight) && _onhost(c.delay) && nedges(c) > 0
        w = extrema(c.weight)
        d = extrema(c.delay)
        unit = eltype(c.delay) <: Integer ? " steps" : " ms"   # resolved (steps) vs unresolved (ms)
        print(io, " · w∈[", _fmt(w[1]), ",", _fmt(w[2]), "] · delay∈[", _fmt(d[1]), ",", _fmt(d[2]), "]", unit)
    end
    return nothing
end
Base.show(io::IO, ::MIME"text/plain", c::SparseCSR) = _conn_summary(io, c)
Base.show(io::IO, c::SparseCSR) = print(io, "SparseCSR(", npre(c), "→", npost(c), ", ", _commas(nedges(c)), " edges)")

# === projection (bare: endpoints unknown → synapse + conn only; the network supplies labels) ===
function Base.show(io::IO, ::MIME"text/plain", p::Projection)
    get(io, :compact, false) && return show(io, p)
    print(io, "Projection")
    children = Any[("synapse  " * _oneline(p.synapse), Any[]), ("conn     " * _oneline(p.conn), Any[])]
    p.plasticity === nothing || push!(children, ("plasticity  " * string(nameof(typeof(p.plasticity))), Any[]))
    _print_tree(io, children)
    return nothing
end
Base.show(io::IO, p::Projection) = print(io, "Projection(", nameof(typeof(p.synapse)), ", ", _commas(nedges(p.conn)), " edges)")

_synkind(::AbstractSynapseModel) = ""    # fallback for custom synapses (no COBA/CUBA/delta tag)

# === PoissonDrive (a leaf node of the network tree) ===
Base.show(io::IO, d::PoissonDrive) = print(io, "PoissonDrive(rate=", _fmt(d.rate), ", weight=", _fmt(d.weight), ")")

# === DewdropNetwork: the full problem tree (populations + projections + drive + noise) ===
# the per-population model: a bare model is shared by every population; a MultiModel resolves the
# group whose range covers `r`; a Heterogeneous shows its base model.
_pop_model_str(model::AbstractNeuronModel, r) = string(typeof(model))
function _pop_model_str(model::MultiModel, r)
    for (m, rg) in zip(model.models, model.ranges)
        first(rg) ≤ first(r) && last(r) ≤ last(rg) && return string(typeof(m))
    end
    return string(typeof(model))
end
_pop_model_str(model::Heterogeneous, r) = string(typeof(model.base))

# a projection's `src → dst` label: exact from the builder's `projlabels`, else recovered by matching
# the connectivity's source/target index ranges back to the subpop registry.
function _recover_name(subpops, idxarr)
    (_onhost(idxarr) && !isempty(idxarr)) || return "?"
    lo, hi = extrema(idxarr)
    for (name, r) in pairs(subpops)
        name === :all && continue
        (first(r) ≤ lo && hi ≤ last(r)) && return string(name)
    end
    return "[$lo:$hi]"
end
function _proj_label(net::DewdropNetwork, i::Int)
    if net.projlabels !== nothing
        pr = net.projlabels[i]
        return string(pr.first, " → ", pr.second)
    end
    p = net.projections[i]
    return string(_recover_name(net.subpops, p.conn.src), " → ", _recover_name(net.subpops, p.conn.post))
end
function _proj_head(net::DewdropNetwork, i::Int)
    p = net.projections[i]
    kind = _synkind(p.synapse)
    tag = isempty(kind) ? "" : " · " * kind
    return string(_proj_label(net, i), "  ", nameof(typeof(p.synapse)), tag, "  ", _commas(nedges(p.conn)), " edges")
end
_npops(subpops) = count(!=(:all), keys(subpops))

function Base.show(io::IO, ::MIME"text/plain", net::DewdropNetwork)
    get(io, :compact, false) && return show(io, net)
    print(io, "DewdropNetwork · N=", net.n, " · t∈[", _fmt(net.tspan[1]), ",", _fmt(net.tspan[2]), "] ms · ",
        nameof(typeof(net.arch)))
    children = Any[]
    pops = [(string(name), r) for (name, r) in pairs(net.subpops) if name !== :all]
    nw = maximum(p -> length(p[1]), pops; init = 0)
    rngs = String["[$(first(r)):$(last(r))]" for (_, r) in pops]
    rw = maximum(length, rngs; init = 0)
    popkids = Any[(string(rpad(pops[i][1], nw), "  ", rpad(rngs[i], rw), "  ", _pop_model_str(net.model, pops[i][2])), Any[])
                  for i in eachindex(pops)]
    isempty(popkids) || push!(children, ("populations ($(length(popkids)))", popkids))
    if !isempty(net.projections)
        projkids = Any[(_proj_head(net, i), Any[]) for i in eachindex(net.projections)]
        push!(children, ("projections ($(length(projkids)))", projkids))
    end
    net.drive === nothing || push!(children, ("drive  " * _oneline(net.drive), Any[]))
    net.noise === nothing || push!(children, ("noise  " * _oneline(net.noise), Any[]))
    _print_tree(io, children)
    return nothing
end
Base.show(io::IO, net::DewdropNetwork) =
    print(io, "DewdropNetwork(N=", net.n, ", ", _npops(net.subpops), " pops, ", length(net.projections), " projs)")

# === NetworkBuilder: the exact, unbuilt named hierarchy ===
function _builder_proj_head(s::_ProjSpec)
    head = string(s.src, " → ", s.dst, "  ", nameof(typeof(s.synapse)))
    kw = s.kw
    parts = String[]
    (haskey(kw, :p) && kw.p isa Number) && push!(parts, "p=$(_fmt(kw.p))")
    (haskey(kw, :weight) && kw.weight isa Number) && push!(parts, "w=$(_fmt(kw.weight))")
    (haskey(kw, :delay) && kw.delay isa Number) && push!(parts, "delay=$(_fmt(kw.delay))")
    isempty(parts) || (head *= "  " * join(parts, " "))
    return head
end
# shared builder/spec tree render (NetworkBuilder + the structured FrozenBuilder spec carry the same fields).
function _show_builder(io::IO, arch, tspan, names, models, sizes, projspecs, drive; head::String, tag::String)
    print(io, head, " · ", nameof(typeof(arch)), " · t∈[", _fmt(tspan[1]), ",", _fmt(tspan[2]), "] ms · ")
    _styled(io, tag, :dim)
    children = Any[]
    popkids = Any[(string(names[i], "  ", sizes[i], "  ", typeof(models[i])), Any[]) for i in eachindex(names)]
    isempty(popkids) || push!(children, ("populations ($(length(popkids)))", popkids))
    projkids = Any[(_builder_proj_head(s), Any[]) for s in projspecs]
    isempty(projkids) || push!(children, ("projections ($(length(projkids)))", projkids))
    drive === nothing || push!(children, ("drive  " * _oneline(drive), Any[]))
    _print_tree(io, children)
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", nb::NetworkBuilder)
    get(io, :compact, false) && return show(io, nb)
    _show_builder(io, nb.arch, nb.tspan, nb.names, nb.models, nb.sizes, nb.projspecs, nb.drive;
        head = "NetworkBuilder", tag = "(unbuilt)")
    return nothing
end
Base.show(io::IO, nb::NetworkBuilder) =
    print(io, "NetworkBuilder(", length(nb.names), " pops, ", length(nb.projspecs), " projs, unbuilt)")

# === deferred network specs (NetworkSpec.jl) ===
# structured (frozen builder): the same populations/projection-recipe tree, marked unmaterialised.
function Base.show(io::IO, ::MIME"text/plain", spec::FrozenBuilder)
    get(io, :compact, false) && return show(io, spec)
    _show_builder(io, spec.arch, spec.tspan, spec.names, spec.models, spec.sizes, spec.projspecs, spec.drive;
        head = "NetworkSpec", tag = "(spec, unmaterialised)")
    return nothing
end
Base.show(io::IO, spec::FrozenBuilder) =
    print(io, "NetworkSpec(", length(spec.names), " pops, ", length(spec.projspecs), " projs, unmaterialised)")

# thunk (deferred constructor): the constructor label + captured params (compact, like a leaf).
function Base.show(io::IO, ::MIME"text/plain", spec::DeferredNetwork)
    get(io, :compact, false) && return show(io, spec)
    print(io, "NetworkSpec")
    _styled(io, "(:$(spec.label))", :type)
    print(io, " · ")
    _styled(io, "deferred", :dim)
    fns = keys(spec.kw)
    isempty(fns) && return nothing
    namew = maximum(f -> length(string(f)), fns)
    for f in fns
        print(io, "\n  ")
        _styled(io, rpad(string(f), namew), :field)
        print(io, " = ", _fmt(getfield(spec.kw, f)))
    end
    return nothing
end
function Base.show(io::IO, spec::DeferredNetwork)
    print(io, "NetworkSpec(:", spec.label)
    isempty(spec.kw) || print(io, ", ", length(spec.kw), " params")
    print(io, ", deferred)")
    return nothing
end

# === solutions ===
# mean firing rate (Hz) over a set of per-unit spike counts and a duration in ms (canonical time).
_rate_hz(counts, n, dur_ms) = (sum(counts) / n) / dur_ms * 1000

function _sol_rate_line(sol::DewdropSolution, dur)
    parts = String[]
    for (name, r) in pairs(sol.subpops)
        name === :all && continue
        push!(parts, string(name, " ", round(_rate_hz(@view(sol.spike_count[r]), length(r), dur); digits = 1), " Hz"))
    end
    overall = round(_rate_hz(sol.spike_count, length(sol.spike_count), dur); digits = 1)
    return isempty(parts) ? string("mean ", overall, " Hz") : string(join(parts, " · "), "  (mean ", overall, " Hz)")
end
function Base.show(io::IO, ::MIME"text/plain", sol::DewdropSolution)
    get(io, :compact, false) && return show(io, sol)
    dur = sol.nsteps * sol.dt
    print(io, "DewdropSolution · N=", length(sol.spike_count), " · ", sol.nsteps, " steps × dt=", _fmt(sol.dt),
        " ms = ", _fmt(dur), " ms")
    children = Any[]
    pops = [(string(name), r) for (name, r) in pairs(sol.subpops) if name !== :all]
    nw = maximum(p -> length(p[1]), pops; init = 0)
    rngs = String["[$(first(r)):$(last(r))]" for (_, r) in pops]
    rw = maximum(length, rngs; init = 0)
    popkids = Any[(string(rpad(pops[i][1], nw), "  ", rpad(rngs[i], rw)), Any[]) for i in eachindex(pops)]
    isempty(popkids) || push!(children, ("populations ($(length(popkids)))", popkids))
    isempty(sol.record) ||
        push!(children, ("recorded  " * join([string(k, " (", r.kind, ")") for (k, r) in pairs(sol.record)], ", "), Any[]))
    push!(children, ("firing rate  " * _sol_rate_line(sol, dur), Any[]))
    _print_tree(io, children)
    return nothing
end
Base.show(io::IO, sol::DewdropSolution) =
    print(io, "DewdropSolution(N=", length(sol.spike_count), ", ", _fmt(sol.nsteps * sol.dt), " ms)")

function Base.show(io::IO, ::MIME"text/plain", ss::SubSolution)
    get(io, :compact, false) && return show(io, ss)
    print(io, "SubSolution · ", ss.name, " [", first(ss.range), ":", last(ss.range), "] · ", length(ss.range), " neurons")
    print(io, "\n  ")
    _styled(io, "firing rate: ", :section)
    print(io, round(_rate_hz(ss.spike_count, length(ss.range), duration(ss.parent)); digits = 1), " Hz")
    return nothing
end
Base.show(io::IO, ss::SubSolution) = print(io, "SubSolution(", ss.name, ", ", length(ss.range), " neurons)")

function Base.show(io::IO, ::MIME"text/plain", bsol::BatchedSolution)
    get(io, :compact, false) && return show(io, bsol)
    N = size(bsol.spike_count, 1)
    dur = bsol.nsteps * bsol.dt
    print(io, "BatchedSolution · N=", N, " × B=", bsol.batch, " · ", bsol.nsteps, " steps × dt=", _fmt(bsol.dt), " ms")
    print(io, "\n  ")
    _styled(io, "firing rate: ", :section)
    print(io, round(_rate_hz(bsol.spike_count, length(bsol.spike_count), dur); digits = 1), " Hz  (mean over ", bsol.batch, " instances)")
    isempty(bsol.record) || (print(io, "\n  "); _styled(io, "recorded: ", :section);
        print(io, join([string(k, " (", r.kind, ")") for (k, r) in pairs(bsol.record)], ", ")))
    return nothing
end
Base.show(io::IO, bsol::BatchedSolution) = print(io, "BatchedSolution(N=", size(bsol.spike_count, 1), ", B=", bsol.batch, ")")

# === delay wrapper (so `steps(5)` round-trips rather than printing `Dewdrop.Steps(5)`) ===
Base.show(io::IO, s::Steps) = print(io, "steps(", s.n, ")")

# === BatchedModel (per-member params; Heterogeneous on the batch axis) ===
_batched_B(bm::BatchedModel) = isempty(bm.params) ? 0 : length(first(bm.params))
function Base.show(io::IO, ::MIME"text/plain", bm::BatchedModel)
    get(io, :compact, false) && return show(io, bm)
    print(io, "BatchedModel")
    _styled(io, "{$(typeof(bm.base))}", :type)
    print(io, " · ")
    _styled(io, "per-member (B=$(_batched_B(bm)))", :dim)
    kw = maximum(k -> length(string(k)), keys(bm.params); init = 0)
    overrides = Any[(_override_summary(rpad(string(k), kw), v), Any[]) for (k, v) in pairs(bm.params)]
    children = Any[("base  " * _oneline(bm.base), Any[]), ("per-member", overrides)]
    _print_tree(io, children)
    return nothing
end
Base.show(io::IO, bm::BatchedModel) =
    print(io, "BatchedModel(", nameof(typeof(bm.base)), ", ", length(bm.params), " params ×", _batched_B(bm), ")")

# === NetworkBatch ===
function Base.show(io::IO, ::MIME"text/plain", b::NetworkBatch)
    get(io, :compact, false) && return show(io, b)
    B = length(b.members)
    print(io, "NetworkBatch · ", B, " member", B == 1 ? "" : "s")
    isempty(b.members) && return nothing
    print(io, "\n  ")
    _styled(io, "first: ", :section)
    show(IOContext(io, :compact => true), first(b.members))
    return nothing
end
Base.show(io::IO, b::NetworkBatch) = print(io, "NetworkBatch(", length(b.members), " members)")

# === BatchSolution ===
_member_rate(bs::BatchSolution, b::Integer) =
    (sc = bs.spike_counts[b]; isempty(sc) ? 0.0 : sum(sc) / length(sc) / bs.duration * 1000)   # mean Hz
function Base.show(io::IO, ::MIME"text/plain", bs::BatchSolution)
    get(io, :compact, false) && return show(io, bs)
    B = length(bs.spike_counts)
    print(io, "BatchSolution · ", B, " member", B == 1 ? "" : "s", " · mode ")
    _styled(io, ":$(bs.mode)", :type)
    print(io, " · ", _fmt(bs.duration), " ms")
    print(io, "\n  ")
    _styled(io, "firing rate: ", :section)
    k = min(B, 6)
    print(io, join([string("m", b, " ", round(_member_rate(bs, b); digits = 1), " Hz") for b in 1:k], " · "))
    B > k && print(io, " · … (", B, " total)")
    return nothing
end
Base.show(io::IO, bs::BatchSolution) = print(io, "BatchSolution(", length(bs.spike_counts), " members, :", bs.mode, ")")
