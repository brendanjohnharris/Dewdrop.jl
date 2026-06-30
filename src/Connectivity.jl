# * Connectivity interface (M0 contract 5)
# Connectivity is an interface --- `for_each_post(f, conn, pre)` walks a presynaptic
# neuron's out-edges --- backed by CSR-parallel arrays over *source* neurons, never a
# dense [post x pre] matrix. Per-synapse weight and delay (in integer time steps)
# live in CSR-parallel arrays. This keeps event-driven sparse scatter and (later)
# procedural/JIT-regenerated connectivity expressible, and makes the whole structure
# Adapt-movable to a device.

"""
    AbstractConnectivity

Interface for synaptic connectivity. Implementations expose [`for_each_post`](@ref),
[`npre`](@ref), [`npost`](@ref) and [`nedges`](@ref). The canonical implementation is
[`SparseCSR`](@ref).
"""
abstract type AbstractConnectivity end

"""
    SparseCSR(arch, edges; npre, npost)

Compressed-sparse-row connectivity grouped by presynaptic neuron. `edges` is a
collection of `(pre, post, weight, delay_steps)` tuples; out-edges of presynaptic
neuron `i` occupy `rowptr[i]:rowptr[i+1]-1` of the parallel `post` / `weight` /
`delay` arrays, in the order given. Arrays are allocated through `arch`.
"""
struct SparseCSR{
        RV <: AbstractVector{<:Integer},
        PV <: AbstractVector{<:Integer},
        WV <: AbstractVector,
        DV <: AbstractVector{<:Real},   # Int = resolved steps (hot path); Float = unresolved ms delays
    } <: AbstractConnectivity
    rowptr::RV   # length npre + 1, 1-based: out-edges of pre i are rowptr[i]:rowptr[i+1]-1
    post::PV     # length nedges, postsynaptic target indices
    weight::WV   # length nedges, per-synapse weights
    delay::DV    # length nedges, per-synapse conduction delays (integer time steps)
    src::PV      # length nedges, presynaptic source of each edge (the inverse of rowptr) ---
    # materialised so the GPU scatter can be EDGE-parallel (one thread per synapse) without a
    # per-edge binary search; sorted ascending (edges are grouped by source), so the idle-thread
    # `spiked[src[e]]` reads stay coalesced at sparse firing.
    maxdeg::Int  # the largest out-degree (longest row); sizes the compacted 2-level scatter launch
    npre::Int
    npost::Int
end
Adapt.@adapt_structure SparseCSR
export SparseCSR

# --- Delay units. A synaptic delay is a PHYSICAL TIME (ms) by default --- the same units as `dt` and
# every other quantity --- resolved to integer steps at the solve `dt`, so its meaning is dt-independent.
# `steps(n)` is the escape for an exact step count. The connectome stores the delay AS GIVEN (Float ms or
# Int steps); `init` resolves it via [`_resolve_delays`](@ref) once `dt` is known (the hot-path scatter
# always reads the resolved integer-step connectome). ---
struct Steps
    n::Int
end
"""
    steps(n)

An explicit synaptic delay of `n` integer time steps (units of `dt`), bypassing the default millisecond
interpretation --- e.g. `delay = steps(5)`. A bare number is a delay in milliseconds. Requires `n ≥ 1`
(the fixed-step engine delivers in the next step, so it cannot represent a within-step delay).
"""
steps(n::Integer) = n ≥ 1 ? Steps(Int(n)) :
    throw(ArgumentError("steps(n) requires n ≥ 1 (the fixed-step engine cannot deliver within a step); got $n"))
export steps

@inline _ms_to_steps(ms::Real, dt::Real) = max(1, round(Int, ms / dt))   # physical ms → integer steps (≥1)
@inline _delayval(d::Steps) = d.n           # explicit steps → stored Int (resolution is the identity)
@inline _delayval(d::Real) = Float64(d)     # ms → stored Float (resolution converts at the solve dt)
# At the LOW-LEVEL `SparseCSR(arch, edges)` boundary an edge delay is already a stored value: a raw Int
# IS a step count, a Float IS ms; a `Steps` wrapper (hand-built edges) is unwrapped to its Int step count.
@inline _edgedelay(d::Steps) = d.n
@inline _edgedelay(d::Real) = d

function SparseCSR(
        arch::AbstractArchitecture, edges; npre::Integer, npost::Integer,
        index_type::Type{IT} = Int,
    ) where {IT <: Integer}
    Tw = isempty(edges) ? Float32 : typeof(edges[1][3])
    nE = length(edges)

    counts = zeros(Int, npre)
    for e in edges
        counts[e[1]] += 1
    end
    # `index_type = Int32` halves the rowptr/post/delay bandwidth on the scatter (the
    # bandwidth-bound hot path) --- safe whenever nedges < 2^31. Counts/positions are computed in
    # `Int` and stored narrowed.
    rowptr = Vector{IT}(undef, npre + 1)
    rowptr[1] = 1
    maxdeg = 0
    for i in 1:npre
        rowptr[i + 1] = rowptr[i] + counts[i]
        maxdeg = max(maxdeg, counts[i])
    end

    Td0 = isempty(edges) ? IT : typeof(_edgedelay(edges[1][4]))
    Td = Td0 <: Integer ? IT : Td0   # step (Int) delays → the narrow index type; ms (Float) keep precision
    post = Vector{IT}(undef, nE)
    weight = Vector{Tw}(undef, nE)
    delay = Vector{Td}(undef, nE)
    src = Vector{IT}(undef, nE)
    fillpos = Vector{Int}(rowptr)
    for e in edges
        pre, p, w, d = e
        idx = fillpos[pre]
        post[idx] = p
        weight[idx] = w
        delay[idx] = _edgedelay(d)
        src[idx] = pre
        fillpos[pre] += 1
    end

    return SparseCSR(
        on_architecture(arch, rowptr),
        on_architecture(arch, post),
        on_architecture(arch, weight),
        on_architecture(arch, delay),
        on_architecture(arch, src),
        maxdeg, Int(npre), Int(npost),
    )
end

npre(c::SparseCSR) = c.npre
npost(c::SparseCSR) = c.npost
nedges(c::SparseCSR) = length(c.post)

# An empty (0-edge) CSR over the whole network, for projections whose contribution is not network-spike
# driven (a prescribed/streaming external drive): makes the per-step network scatter a no-op while
# satisfying the synapse-state interface (the real wiring, if any, lives inside the synapse model).
_empty_csr(arch, N) = SparseCSR(arch, Tuple{Int, Int, Float64, Int}[]; npre = N, npost = N)

"""
    _resolve_delays(conn, dt) -> SparseCSR

Resolve a connectome's delays to integer time steps at the solve `dt`. Explicit-step (Int) delays pass
through unchanged (the identity); physical (Float ms) delays convert via [`_ms_to_steps`](@ref) into the
connectome's index type. Called once at `init`, so the hot-path scatter always reads integer steps.
"""
function _resolve_delays(conn::SparseCSR, dt)
    eltype(conn.delay) <: Integer && return conn     # explicit steps → identity (already resolved)
    IT = eltype(conn.src)                            # Float ms delays → Int steps (at the solve dt)
    stepd = IT.(_ms_to_steps.(conn.delay, dt))
    return SparseCSR(conn.rowptr, conn.post, conn.weight, stepd, conn.src, conn.maxdeg, conn.npre, conn.npost)
end

"""
    for_each_post(f, conn, pre)

Call `f(post, weight, delay)` for each out-edge of presynaptic neuron `pre`. Designed
to run inside a per-presynaptic-neuron kernel/loop body (device-side indexing), so the
event-driven scatter touches only a spiking neuron's own row.
"""
@inline function for_each_post(f::F, c::SparseCSR, pre::Integer) where {F}
    @inbounds for e in c.rowptr[pre]:(c.rowptr[pre + 1] - 1)
        f(c.post[e], c.weight[e], c.delay[e])
    end
    return nothing
end

"""
    correlate_weights!(conn, J; targets = 1:npost(conn), jitter = 0.05, seed, count_empty = true) -> conn

Rescale a connectome's weights in place to **in-degree-normalised** values --- the standard
balanced-network `1/√k` scaling. Each edge into post-neuron `p` gets mean weight `wₘ = J_rec / √k[p]`
(with `k[p]` the in-degree of `p` over `targets`), plus a relative Gaussian jitter `N(0,1)·jitter·wₘ`
drawn reproducibly from the counter RNG keyed by edge index and `seed`. The recurrent scale is

    J_rec = J · Σ k[p] / Σ √max(k[p], 1)      (`count_empty = true`, the default --- BrainPy's convention)

so a uniformly-connected target population (all `k[p] = k`) gives every weight ≈ `J`. With the default
`count_empty = true` a zero-in-degree target contributes `√1 = 1` to the denominator, exactly reproducing
BrainPy's vectorised `√max(k,1)` (it avoids a NaN and slightly shrinks every weight). With
`count_empty = false` the denominator sums `√k[p]` over **connected** targets only (`k[p] ≥ 1`), so isolated
neurons don't affect the scale --- the more principled form; the two agree when every target is wired (a
dense spatial sheet). `targets` is the post sub-population the in-degrees are counted over (a contiguous
global-index range), so a projection into a sub-population normalises against that sub-population.
Reproducible from the seed. The in-degree count and per-edge weight assignment are inherently serial, so
they run on a host copy of the post indices --- a no-op for a CPU connectome, a single bulk device→host copy
for a GPU one --- and the result is written back in one bulk `copyto!`, so it works on a GPU connectome too.
"""
function correlate_weights!(conn::SparseCSR, J::Real; targets = 1:npost(conn),
        jitter::Real = 0.05, seed::Unsigned, count_empty::Bool = true)
    lo = first(targets)
    n = length(targets)
    post = Array(conn.post)          # host copy (no scalar device indexing); bulk on GPU, no-op-ish on CPU
    k = zeros(Int, n)
    @inbounds for p in post
        k[Int(p) - lo + 1] += 1
    end
    sum_k = 0.0
    sum_sqrt = 0.0
    @inbounds for kk in k
        if kk > 0
            sum_k += kk
            sum_sqrt += sqrt(kk)
        elseif count_empty           # BrainPy √max(k,1): a zero-in-degree target contributes √1 = 1
            sum_sqrt += 1.0
        end                          # count_empty = false → empty targets excluded (the principled form)
    end
    J_rec = sum_sqrt > 0 ? J * sum_k / sum_sqrt : 0.0
    T = eltype(conn.weight)
    w = Vector{T}(undef, length(post))
    @inbounds for e in eachindex(post)
        wm = J_rec / sqrt(k[Int(post[e]) - lo + 1])
        w[e] = T(wm + draw_normal(Float64, seed, e, 0) * jitter * wm)
    end
    copyto!(conn.weight, w)          # single bulk host→device write-back
    return conn
end
export correlate_weights!

"""
    correlate_weights(J; jitter = 0.05, seed, count_empty = true) -> adjuster

Curried [`correlate_weights!`](@ref) for the builder's `adjust` hook:
`project!(…; adjust = correlate_weights(J; seed))`. The hook supplies the projection's resolved
destination range, so the in-degree normalisation is taken over the actual post sub-population.
`count_empty` selects the zero-in-degree convention (default `true` = BrainPy's `√max(k,1)`; see
[`correlate_weights!`](@ref)).
"""
correlate_weights(J::Real; jitter::Real = 0.05, seed::Unsigned, count_empty::Bool = true) =
    (conn, ctx) -> correlate_weights!(conn, J; targets = ctx.targets, jitter = jitter, seed = seed, count_empty = count_empty)
export correlate_weights

"""
    fixed_prob(arch, npre, npost, p; weight, delay, seed, allow_self=true, sources=1:npre, targets=1:npost)

Random connectivity in which each `(pre, post)` edge is present with probability `p`,
sampled reproducibly from the counter-based RNG keyed by `(pre, post)` (so a given `seed`
yields a fixed, copyable connectome). `weight` and `delay` may be scalars or functions of
the presynaptic index `pre` --- the latter gives excitatory/inhibitory neurons signed
weights. `sources` / `targets` restrict the presynaptic / postsynaptic neuron sets (for
named-subpopulation projections, e.g. `:E => :I`); flat `pre` / `post` indices stay
absolute (`1:npre` / `1:npost`). Returns a [`SparseCSR`](@ref).
"""
function fixed_prob(
        arch::AbstractArchitecture, npre::Integer, npost::Integer, p::Real;
        weight, delay, seed::Unsigned, allow_self::Bool = true, sources = 1:npre, targets = 1:npost,
        index_type::Type = Int,
    )
    wtype = typeof(to_weight(weight isa Function ? weight(1) : weight))
    dtype = typeof(_delayval(delay isa Function ? delay(1) : delay))   # Int (steps) or Float (ms)
    edges = Tuple{Int, Int, wtype, dtype}[]
    pT = Float64(p)
    pT > 0 || return SparseCSR(arch, edges; npre = npre, npost = npost, index_type = index_type)
    ntargets = length(targets)
    sizehint!(edges, ceil(Int, 1.1 * pT * length(sources) * ntargets))
    # Geometric-gap sampling: instead of testing every (pre, post) pair (O(npre·npost)), draw
    # the run of skipped targets before each edge from a geometric distribution, so cost scales
    # with the number of EDGES (~p·npre·npost). The k-th gap draw for source `pre` is keyed
    # (seed, k, pre), so a given seed still yields a fixed, copyable connectome (a different
    # realisation from per-pair sampling, but the same Bernoulli(p) marginal per target).
    # Sampling runs over the LOCAL target index `1:ntargets`, then maps through `targets` to the
    # absolute post index; with the default `targets = 1:npost` the map is the identity, so the
    # realised connectome (and draw sequence) is unchanged.
    invlog = inv(log1p(-pT))                              # 1/log(1-p) < 0
    for pre in sources
        w = wtype(to_weight(weight isa Function ? weight(pre) : weight))
        d = _delayval(delay isa Function ? delay(pre) : delay)   # ms (Float) or steps (Int); resolved at init
        local_post = 0
        k = 0
        while true
            k += 1
            u = draw_uniform(Float64, seed, k, pre)
            local_post += (u > 0.0 ? floor(Int, log(u) * invlog) : ntargets) + 1   # skip a geometric gap
            local_post > ntargets && break
            post = Int(targets[local_post])              # local target index → absolute post index
            (!allow_self && pre == post) && continue
            push!(edges, (Int(pre), post, w, d))
        end
    end
    return SparseCSR(arch, edges; npre = npre, npost = npost, index_type = index_type)
end
export fixed_prob
