# * Batching: run B network "members" together. The general execution is block-diagonal: stack the B
# networks into one (ΣN)-neuron network whose connectome is block-diagonal (member b offset by ΣN_{<b}, no
# cross-member edges), solved by the existing scalar engine (no new kernels). This handles distinct
# models / weights / delays / topology across members; each member's block runs independently (so with no
# drive it is bit-identical to that member solved alone). The memory-optimal shared-CSR ensemble (`batch=B`)
# is the special case for input/v0/seed-only variation; routing to it automatically is a follow-on.
#
# Members are produced by `batch(...)`: an explicit vector (of networks or specs), a base + generator
# function, or a parameter sweep over a base. `solve(::NetworkBatch, alg; tspan, …)` materialises the
# members, stacks them block-diagonally, and solves; per-member results are addressed via the `memberK`
# subpopulations (`sol[:member2]`, `firing_rate(sol, :member1)`).

"""
    NetworkBatch

A batch of `B` network members run together (see [`batch`](@ref)). Each member materialises to a
`DewdropNetwork`. Solved with `solve(batch, alg; tspan, …)`; member `b` is addressed on the solution as
`sol[:memberB]`.
"""
struct NetworkBatch{M}
    members::Vector{M}
end
export NetworkBatch

"""
    nmembers(b::NetworkBatch) -> Int

The number of members `B` in the batch.
"""
nmembers(b::NetworkBatch) = length(b.members)

"""
    batch(items::AbstractVector) -> NetworkBatch
    batch(f, base; n) -> NetworkBatch
    batch(base; param = values, …) -> NetworkBatch

Construct a [`NetworkBatch`](@ref) of `B` members.
- a vector of `DewdropNetwork`s / [`AbstractNetworkSpec`](@ref)s: the members directly;
- a generator: member `i` is `f(base, i)` for `i in 1:n`;
- a parameter sweep over `base` (a neuron model / spec / network): `B = length(values)`, member `i` sets each
  `param` to `values[i]` (zipped). `cartesian = true` sweeps the Cartesian product.
"""
batch(items::AbstractVector) = NetworkBatch(collect(items))
batch(f, base; n::Integer) = NetworkBatch([f(base, i) for i in 1:Int(n)])
export batch, nmembers

# parameter sweep
# rebuild an isbits model/struct with the named fields overridden (positional reconstruction; no extra dep).
function _setparams(x, nt::NamedTuple)
    T = typeof(x)
    for k in keys(nt)
        hasfield(T, k) || error("sweep parameter `$k` is not a field of $T")
    end
    vals = ntuple(fieldcount(T)) do i
        f = fieldname(T, i)
        haskey(nt, f) ? convert(fieldtype(T, i), nt[f]) : getfield(x, i)
    end
    return T(vals...)
end

# apply one member's sweep values. A neuron model varies its own params; on a network a pure `input` sweep
# keeps the same model + connectome (→ the fused Mode-0 ensemble), while a model-param sweep varies the
# model but shares the connectome (→ multi-run / block).
_apply_sweep(base::AbstractNeuronModel, nt::NamedTuple) = _setparams(base, nt)
function _apply_sweep(net::DewdropNetwork, nt::NamedTuple)
    keys(nt) == (:input,) && return _swap_input(net, nt.input)
    return _setparams(net, (; model = _setparams(net.model, nt)))
end
# Rebuild a network with named fields replaced, carrying every other field through. The one place that
# knows `DewdropNetwork`'s field list, so a new field cannot be silently dropped by a mode's rebuild
# every mode's rebuild.
function _respec(net::DewdropNetwork; kw...)
    over = NamedTuple(kw)
    keep(f) = haskey(over, f) ? over[f] : getfield(net, f)
    return DewdropNetwork(
        keep(:model), net.n; input = keep(:input), tspan = keep(:tspan), arch = keep(:arch),
        schedule = keep(:schedule), projections = keep(:projections), drive = keep(:drive),
        noise = keep(:noise), subpops = keep(:subpops), positions = keep(:positions),
        projlabels = keep(:projlabels), stimuli = keep(:stimuli),
    )
end

# a fresh network with `input` swapped but the same model + projections objects (`===`), so a pure-input
# sweep is detected as a shared-connectivity batch and runs as the fused Mode-0 ensemble.
_swap_input(net::DewdropNetwork, input) = _respec(net; input = input)

function batch(base; cartesian::Bool = false, kw...)
    isempty(kw) && error("batch(base; param = values, …) needs at least one swept parameter")
    nt = (; kw...)
    if cartesian
        combos = vec(collect(Iterators.product(values(nt)...)))
        members = [_apply_sweep(base, NamedTuple{keys(nt)}(c)) for c in combos]
    else
        lens = map(length, values(nt))
        n = first(lens)
        all(==(n), lens) || error("zipped sweep needs equal-length vectors (got lengths $lens); use cartesian = true")
        members = [_apply_sweep(base, NamedTuple{keys(nt)}(map(v -> v[i], values(nt)))) for i in 1:n]
    end
    return NetworkBatch(members)
end

# block-diagonal assembly
_expand_input(x::Number, n::Int) = fill(x, n)
_expand_input(x::AbstractVector, n::Int) =
    (length(x) == n || error("input length $(length(x)) ≠ N = $n"); collect(x))

# a member projection's edges shifted into its block (src/post offset by `off`); the delays stay
# unresolved (ms / steps) and resolve at the stacked network's `init`.
function _offset_edges(conn::SparseCSR, off::Int)
    src, post, w, d = _host_edges(conn)
    return [(Int(src[e]) + off, Int(post[e]) + off, w[e], d[e]) for e in 1:nedges(conn)]
end

# The edge arrays as host `Array`s. A member built on a GPU holds them as device arrays, which the
# edge-list comprehensions above would otherwise read one scalar at a time (forbidden under
# `allowscalar(false)`, and ruinous otherwise). A no-op on CPU.
_host_edges(conn::SparseCSR) = (Array(conn.src), Array(conn.post), Array(conn.weight), Array(conn.delay))

# per-member stateful-synapse offset (so block-stacking carries every member's drive)
# A synapse is block-mergeable when its behaviour is fully captured by its edge list + shared scalar params,
# so the B members' projections share one synapse over a concatenated, offset connectome (the default:
# FrozenDualExpSynapse, DualExpSynapse, CUBA/COBA, delta). A synapse that carries its own internal per-member
# wiring is not mergeable: a streaming `PoissonSource` or replayed `SpikeSourceArray` drive holds an
# `extconn` targeting member-local indices `1:N`, so reusing member 1's would leave members 2..B undriven.
# Those get one projection per member, each offset into its block (see `_block_diagonal`).
_block_mergeable(s::AbstractSynapseModel) = !is_source_synapse(s)

# Shift a synapse's internal wiring into member b's block: targets += `off`, post-dimension grows to `Ntot`.
# Edge-defined synapses carry none → identity. A `PoissonSource` offsets its `extconn`'s post (targets) only;
# sources, weights, delays and `seed` are untouched, so the member's per-source Poisson draw is
# bit-identical to its standalone solve; only the deposit lands in the member's block.
_offset_synapse(syn::AbstractSynapseModel, off::Int, Ntot::Int, arch) = syn
_offset_synapse(p::PoissonSource, off::Int, Ntot::Int, arch) =
    PoissonSource(p.synapse, _shift_post(p.extconn, off, Ntot, arch), p.rate, p.seed)
# the replay pattern is shared verbatim; only the fan-out moves into the member's block
_offset_synapse(s::SpikeSourceArray, off::Int, Ntot::Int, arch) =
    SpikeSourceArray(s.synapse, _shift_post(s.extconn, off, Ntot, arch), s.spikes)

# A CSR with every post index shifted by `off` and the post dimension grown to `Ntot` (sources / weights /
# delays unchanged), rebuilt on `arch`. Mirrors `_offset_edges` but shifts only the target index.
function _shift_post(conn::SparseCSR, off::Int, Ntot::Int, arch)
    src, post, w, d = _host_edges(conn)
    edges = [(Int(src[e]), Int(post[e]) + off, w[e], d[e]) for e in 1:nedges(conn)]
    return SparseCSR(arch, edges; npre = npre(conn), npost = Ntot)
end

"""
    _block_diagonal(nets) -> DewdropNetwork

Stack networks `nets` into one block-diagonal network: member `b` occupies a contiguous block (no
cross-member edges), so the scalar engine runs the B members independently. Members must share projection
structure (same count + synapse types per index). Per-member ranges are recorded as `memberK` subpops.
"""
function _block_diagonal(nets::AbstractVector{<:DewdropNetwork})
    isempty(nets) && error("batch needs at least one member")
    B = length(nets)
    arch = first(nets).arch
    all(net -> net.arch == arch, nets) || error("block batch: all members must share the architecture")
    nproj = length(first(nets).projections)
    all(net -> length(net.projections) == nproj, nets) ||
        error("block batch: every member must have the same number of projections (got $(unique([length(net.projections) for net in nets])))")

    Ns = [net.n for net in nets]
    offs = [0; cumsum(Ns)[1:(end - 1)]]   # exclusive prefix sum of member sizes (member b's base offset)
    Ntot = sum(Ns)
    model = B == 1 ? first(nets).model : MultiModel([net.model for net in nets], Ns)
    input = reduce(vcat, [_expand_input(net.input, net.n) for net in nets])
    # Stacked projections: a block-mergeable synapse shares one projection over the members' concatenated,
    # offset edges; a non-mergeable one (a streaming drive) gets one projection per member, offset into its
    # block, so every member is driven by its own source rather than member 1's (otherwise members 2..B are
    # silently undriven; their `extconn` would still point at member 1's block).
    projlist = Projection[]
    for j in 1:nproj
        syns = [net.projections[j].synapse for net in nets]
        plas = [net.projections[j].plasticity for net in nets]
        syn, rule = first(syns), first(plas)
        # Merge only when the members really share the projection's synapse and learning rule: the merged
        # projection carries one of each, so members differing in a synaptic constant (τ, τr/τd, Erev) would
        # all silently run member 1's. When they differ, each member gets its own projection over its own
        # offset edges, which is the same branch a non-mergeable (self-wired) synapse already takes.
        if _block_mergeable(syn) && all(==(syn), syns) && all(==(rule), plas)
            edges = reduce(vcat, [_offset_edges(nets[b].projections[j].conn, offs[b]) for b in 1:B])
            push!(projlist, Projection(syn, SparseCSR(arch, edges; npre = Ntot, npost = Ntot); plasticity = rule))
        else
            for b in 1:B
                synb = _offset_synapse(syns[b], offs[b], Ntot, arch)
                edgesb = _offset_edges(nets[b].projections[j].conn, offs[b])
                push!(
                    projlist,
                    Projection(synb, SparseCSR(arch, edgesb; npre = Ntot, npost = Ntot); plasticity = plas[b])
                )
            end
        end
    end
    projs = Tuple(projlist)
    subpops = NamedTuple(Symbol("member", b) => (offs[b] + 1):(offs[b] + Ns[b]) for b in 1:B)
    # Carry the network `drive` field and the SDE `noise` term onto the stacked network. Both are keyed by
    # global neuron index, so a uniform member drive/noise applies correctly across every block; per-member
    # differing drive/noise fields are not block-expressible (attach a per-member drive as a projection via
    # `drive!` / `PoissonSource`, which offsets per member).
    return DewdropNetwork(
        model, Ntot; input = input, tspan = first(nets).tspan, arch = arch,
        schedule = first(nets).schedule, projections = projs, drive = first(nets).drive,
        noise = first(nets).noise, subpops = subpops, stimuli = first(nets).stimuli
    )
end

# the unified batch result
"""
    BatchSolution

The result of solving a [`NetworkBatch`](@ref): per-member per-neuron spike counts (`bs[b]`) over a shared
`duration`, the execution `mode` that ran (`:shared` / `:multirun` / `:block`), and `raw` (the underlying
solution(s) for full access). `firing_rate(bs)` gives the per-member rates.
"""
struct BatchSolution{T, D <: Real, R}
    spike_counts::Vector{Vector{T}}   # always host-resident: every mode collects (a device run included)
    duration::D                       # the member float type, so `firing_rate` keeps the model's precision
    mode::Symbol
    raw::R
end
Base.getindex(bs::BatchSolution, b::Integer) = bs.spike_counts[b]
Base.length(bs::BatchSolution) = length(bs.spike_counts)
nmembers(bs::BatchSolution) = length(bs.spike_counts)
firing_rate(bs::BatchSolution) = [sc ./ bs.duration for sc in bs.spike_counts]
firing_rate(bs::BatchSolution, b::Integer) = bs.spike_counts[b] ./ bs.duration
export BatchSolution

# execution modes (all reuse existing engines; only fused Mode A would touch a kernel)
_materialize_member(net::DewdropNetwork, alg, tspan) = net
_materialize_member(spec::AbstractNetworkSpec, alg, tspan) = materialize(spec, alg; tspan = tspan)

# Every execution mode keys these off member 1, and none of them is expressible per member, so a
# member that differs is refused rather than silently replaced. Per-member external input belongs in
# `input` (which is per member), or in a per-member projection via `drive!` / `PoissonSource`.
function _check_uniform_members(nets)
    length(nets) <= 1 && return nothing
    for (field, what) in ((:drive, "drive"), (:noise, "noise"), (:stimuli, "stimuli"), (:schedule, "schedule"))
        vals = [getfield(n, field) for n in nets]
        all(v -> v == first(vals), vals) || throw(ArgumentError(
            "batch members differ in `$what`, which is not expressible per member; give every member the " *
                "same `$what` and vary `input` (per member) or attach a per-member projection instead"))
    end
    return nothing
end

# pick the mode from what varies across members (cheap `===` identity checks).
function _choose_mode(nets)
    length(nets) ≤ 1 && return :block
    shared = all(n -> n.projections === nets[1].projections && n.n == nets[1].n, nets)
    shared || return :block                                   # distinct topology → block-diagonal
    all(n -> n.model === nets[1].model, nets) && return :shared   # same model too → fused Mode-0 ensemble
    # shared connectome, per-member model: prefer fused Mode A (one launch, O(edges)) if the model type is
    # uniform; else the shared-connectome multi-run.
    M = typeof(nets[1].model)
    return (all(n -> typeof(n.model) === M, nets) && _fusable_model(M)) ? :fused : :multirun
end

# `:fused` diffs the model's own fields, while `_resolve_member` (src/Batch.jl) matches those override keys
# against the underlying scalar model's fields. The two agree only for a model that is its own leaf; a
# wrapper (`Heterogeneous`) would contribute keys like `:base`/`:params` that match nothing, leaving every
# member silently running member 1's parameters. Those go to the multi-run instead.
_fusable_model(::Type{M}) where {M} = _leaf_model(M) === M

# Mode 0, fused shared-CSR ensemble: one network broadcast over B `(N,B)` state columns; per-column input.
function _solve_shared(nets, alg; kwargs...)
    B = length(nets)
    inputmat = reduce(hcat, [_expand_input(net.input, net.n) for net in nets])
    bsol = solve(nets[1], alg; batch = B, input = inputmat, kwargs...)
    return BatchSolution([collect(view(bsol.spike_count, :, b)) for b in 1:B], bsol.nsteps * bsol.dt, :shared, bsol)
end

# Mode A-lite (shared-connectivity multi-run): resolve the connectome once and share the array across B
# separate scalar solves (each with its own model): Mode-A memory (O(edges)) with no kernel changes.
# The B solves are independent and the shared connectome is read-only, so `threads` runs them in parallel
# (outer over members; each inner solve forced single-threaded via `Serial`, bit-identical, to avoid
# nesting the per-neuron threading). `threads = :auto` threads when there are ≥2 members and >1 thread.
function _solve_multirun(nets, alg; threads = :auto, kwargs...)
    rprojs = Tuple(
        Projection(p.synapse, _resolve_delays(p.conn, alg.dt); plasticity = p.plasticity)
            for p in first(nets).projections
    )          # resolve delays once → one shared connectome
    _net(net) = _respec(net; projections = rprojs)
    dothread = threads === true || (threads === :auto && length(nets) ≥ 2 && Threads.nthreads() > 1)
    sols = Vector{Any}(undef, length(nets))
    if dothread
        # single-threaded inner (user `backend =` overrides). A per-neuron model (Heterogeneous /
        # MultiModel) has no Serial method, so it keeps the fused loop, which is bit-identical at any
        # thread count and gates its own threading on work size.
        base_bk = _is_hetero(first(nets).model) ? Fused() : Serial()
        inner = merge((; backend = base_bk), (; kwargs...))
        Threads.@threads for b in eachindex(nets)
            sols[b] = solve(_net(nets[b]), alg; inner...)
        end
    else
        for b in eachindex(nets)
            sols[b] = solve(_net(nets[b]), alg; kwargs...)
        end
    end
    return BatchSolution([collect(s.spike_count) for s in sols], duration(first(sols)), :multirun, sols)
end

# Fused Mode A, per-member params in the (N,B) megakernel: one shared connectome, a `BatchedModel` with
# per-member override arrays, one fused launch. Best of both (fused throughput + O(edges) memory); needs a
# uniform model type across members.
function _solve_fused(nets, alg; kwargs...)
    B = length(nets)
    base = first(nets)
    M = typeof(base.model)
    all(net -> typeof(net.model) === M, nets) ||
        error(":fused needs a uniform neuron-model type across members (got $(unique([typeof(net.model) for net in nets])))")
    _fusable_model(M) ||
        error(
        ":fused cannot sweep a $(nameof(M)) model: its per-member overrides are matched against the " *
            "underlying $(nameof(_leaf_model(M))) fields, so the wrapper's own fields would be silently " *
            "ignored and every member would run member 1's parameters. Use mode = :multirun."
    )
    overrides = Pair{Symbol, Any}[]
    for f in fieldnames(M)
        vals = [getfield(net.model, f) for net in nets]
        all(==(first(vals)), vals) || push!(overrides, f => collect(vals))   # only the fields that vary
    end
    inputmat = reduce(hcat, [_expand_input(net.input, net.n) for net in nets])
    prob = _respec(base)
    # Hand the overrides to `solve` rather than wrapping the model here: it uploads them to `arch`,
    # which a `BatchedModel` built from host vectors would not survive on a GPU.
    bsol = solve(prob, alg; batch = B, input = inputmat, model_overrides = NamedTuple(overrides), kwargs...)
    return BatchSolution([collect(view(bsol.spike_count, :, b)) for b in 1:B], bsol.nsteps * bsol.dt, :fused, bsol)
end

# Mode B, block-diagonal: stack the B members into one network, one scalar solve.
function _solve_block(nets, alg; kwargs...)
    sol = solve(_block_diagonal(nets), alg; kwargs...)
    Ns = [net.n for net in nets]
    offs = [0; cumsum(Ns)[1:(end - 1)]]   # exclusive prefix sum of member sizes (member b's base offset)
    return BatchSolution(
        [collect(sol.spike_count[(offs[b] + 1):(offs[b] + Ns[b])]) for b in eachindex(nets)],
        duration(sol), :block, sol
    )
end

"""
    solve(b::NetworkBatch, alg; tspan = nothing, mode = :auto, kwargs…) -> BatchSolution

Solve all members, returning a [`BatchSolution`](@ref). `mode = :auto` routes by what varies: shared
model + connectome (vary input) → the fused `:shared` ensemble; shared connectome, per-member model →
`:multirun` (B solves sharing the connectome array); distinct topology → `:block` (block-diagonal). Force a
mode with `mode = :shared` / `:multirun` / `:block`.
"""
function CommonSolve.solve(
        b::NetworkBatch, alg::FixedStep; tspan = nothing, mode::Symbol = :auto,
        threads = :auto, kwargs...
    )
    nets = DewdropNetwork[_materialize_member(m, alg, tspan) for m in b.members]
    _check_uniform_members(nets)
    m = mode === :auto ? _choose_mode(nets) : mode
    m === :shared && return _solve_shared(nets, alg; kwargs...)
    m === :multirun && return _solve_multirun(nets, alg; threads = threads, kwargs...)
    m === :fused && return _solve_fused(nets, alg; kwargs...)
    m === :block && return _solve_block(nets, alg; kwargs...)
    return error("unknown batch mode :$m (use :auto / :shared / :multirun / :fused / :block)")
end
