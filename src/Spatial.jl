# * Spatial / structured connectivity.
# Neurons carry positions (D-dimensional coordinate tuples); `distance_prob` then connects each
# ordered pair with a probability that is a function of their separation (a connection kernel),
# optionally on a periodic box. This expresses distance-dependent, ring and grid topologies as a
# single primitive: positions + a kernel. Sampling is per-pair (the probability varies with the
# target, so the geometric skip used by `fixed_prob` does not apply) but stays seed-reproducible
# via the counter-based RNG keyed by (pre, post).

# --- Position layouts: a vector of `NTuple{D,Float64}` coordinates ---
"""
    line_positions(n; spacing=1.0)

`n` neurons evenly spaced on a line.
"""
line_positions(n::Integer; spacing = 1.0) = [(spacing * (i - 1),) for i in 1:n]

"""
    grid_positions(nx, ny; spacing=1.0, centered=false)

`nx*ny` neurons on a rectangular grid (x varies fastest: neuron `(j-1)*nx + i` sits at
`((i-1)·spacing, (j-1)·spacing)`). With `centered=true` the points are cell-centred ---
`((i-0.5)·spacing, (j-0.5)·spacing)` --- tiling `[0, nx·spacing] × [0, ny·spacing]` symmetrically,
the convention for a periodic box (e.g. a spatial E/I sheet).
"""
function grid_positions(nx::Integer, ny::Integer; spacing = 1.0, centered::Bool = false)
    centered || return [(spacing * (i - 1), spacing * (j - 1)) for j in 1:ny for i in 1:nx]
    return [(spacing * (i - 0.5), spacing * (j - 0.5)) for j in 1:ny for i in 1:nx]
end

"""
    ring_positions(n; radius=1.0)

`n` neurons evenly placed on a circle (a ring topology; the wraparound is built into the
geometry, so no `period` is needed).
"""
ring_positions(n::Integer; radius = 1.0) =
    [(radius * cos(2π * (i - 1) / n), radius * sin(2π * (i - 1) / n)) for i in 1:n]

export line_positions, grid_positions, ring_positions

# Euclidean distance, optionally on a periodic box of side lengths `period` (per dimension).
@inline function distance(a::NTuple{D}, b::NTuple{D}, period) where {D}
    s = zero(promote_type(eltype(a), eltype(b)))
    @inbounds for k in 1:D
        δ = abs(a[k] - b[k])
        period !== nothing && (δ = min(δ, period[k] - δ))   # minimum-image wraparound
        s += δ * δ
    end
    return sqrt(s)
end

# --- Connection kernels: distance → probability in [0, 1] ---
"""
    gaussian_kernel(σ; pmax=1.0)

A Gaussian connection kernel `d -> pmax·exp(-d²/2σ²)`.
"""
gaussian_kernel(σ; pmax = 1.0) = d -> pmax * exp(-d^2 / (2 * σ^2))

"""
    exponential_kernel(λ; pmax=1.0)

An exponential connection kernel `d -> pmax·exp(-d/λ)`.
"""
exponential_kernel(λ; pmax = 1.0) = d -> pmax * exp(-d / λ)

"""
    box_kernel(r; p=1.0)

A top-hat kernel: probability `p` within radius `r`, else 0 (local/neighbourhood connectivity).
"""
box_kernel(r; p = 1.0) = d -> d ≤ r ? p : zero(p)

export gaussian_kernel, exponential_kernel, box_kernel

"""
    distance_prob(arch, positions; kernel, weight, delay, seed, allow_self=false, period=nothing, sources=eachindex(positions), targets=eachindex(positions))

Distance-dependent random connectivity: each ordered `(pre, post)` pair (with `pre` in
`sources`, `post` in `targets`) is connected with probability
`kernel(distance(positions[pre], positions[post]))`, sampled reproducibly from the
counter-based RNG. `weight`/`delay` may be scalars or per-source functions; `period` (a tuple
of box side lengths) enables periodic boundaries. `sources` / `targets` restrict the
presynaptic / postsynaptic neuron sets (for named-subpopulation projections, e.g. `:E => :I`);
flat indices stay absolute. Returns a [`SparseCSR`](@ref). Together with
[`grid_positions`](@ref)/[`ring_positions`](@ref) and the kernel helpers this expresses
distance-kernel, ring and grid topologies.
"""
function distance_prob(arch::AbstractArchitecture, positions; kernel, weight, delay,
        seed::Unsigned, allow_self::Bool = false, period = nothing,
        sources = eachindex(positions), targets = eachindex(positions), index_type::Type = Int)
    npost = length(positions)
    wtype = typeof(to_weight(weight isa Function ? weight(1) : weight))
    dtype = typeof(_delayval(delay isa Function ? delay(1) : delay))   # Int (steps) or Float (ms)
    edges = Tuple{Int, Int, wtype, dtype}[]
    for pre in sources
        w = wtype(to_weight(weight isa Function ? weight(pre) : weight))
        d = _delayval(delay isa Function ? delay(pre) : delay)         # ms or steps; resolved at init
        ppos = positions[pre]
        for post in targets
            (!allow_self && pre == post) && continue
            draw_uniform(Float64, seed, pre, post) < kernel(distance(ppos, positions[post], period)) || continue
            push!(edges, (Int(pre), Int(post), w, d))
        end
    end
    return SparseCSR(arch, edges; npre = npost, npost = npost, index_type = index_type)
end
export distance_prob

"""
    random_positions(N, domain; seed, sort=false) -> Vector{NTuple{D,Float64}}

`N` uniformly-random positions in the box `domain` (a tuple of per-dimension side lengths, e.g.
`(0.5, 0.5)`), drawn reproducibly from the counter-based RNG keyed by `(seed, neuron, dim)`. Pair
with [`distance_prob`](@ref)/[`distance_fixed_count`](@ref) for the random-placement populations of a
spatial network (e.g. a randomly placed inhibitory population). `sort=true` lexicographically sorts the
points.
"""
function random_positions(N::Integer, domain::Tuple; seed::Unsigned, sort::Bool = false)
    D = length(domain)
    pts = [ntuple(d -> Float64(domain[d]) * draw_uniform(Float64, seed, i, d), D) for i in 1:Int(N)]
    sort && sort!(pts)
    return pts
end
export random_positions

# --- Fixed-count distance connectivity (Gumbel-max top-k) ---
# A bounded min-heap retaining the `K` largest-score items streamed past it: the Gumbel-max trick
# samples exactly `K` edges without replacement, weighted ∝ kernel(distance), in O(pairs) time and
# O(K) memory (the full score matrix is never materialised). The min-heap root is the smallest kept
# score, so a new higher score evicts it.
mutable struct _TopK
    sc::Vector{Float64}
    pre::Vector{Int}
    post::Vector{Int}
    K::Int
    n::Int
end
_TopK(K::Int) = _TopK(Vector{Float64}(undef, K), Vector{Int}(undef, K), Vector{Int}(undef, K), K, 0)

@inline function _swap!(h::_TopK, i, j)
    h.sc[i], h.sc[j] = h.sc[j], h.sc[i]
    h.pre[i], h.pre[j] = h.pre[j], h.pre[i]
    h.post[i], h.post[j] = h.post[j], h.post[i]
    return nothing
end
function _topk_push!(h::_TopK, score, pre, post)
    h.K == 0 && return h
    if h.n < h.K                                              # heap not full → insert + sift up
        h.n += 1
        i = h.n
        @inbounds (h.sc[i] = score; h.pre[i] = pre; h.post[i] = post)
        @inbounds while i > 1 && h.sc[i] < h.sc[i >> 1]
            _swap!(h, i, i >> 1)
            i >>= 1
        end
    elseif @inbounds(score > h.sc[1])                         # better than the smallest kept → evict root, sift down
        @inbounds (h.sc[1] = score; h.pre[1] = pre; h.post[1] = post)
        i = 1
        @inbounds while true
            l = 2i; r = 2i + 1; m = i
            (l ≤ h.n && h.sc[l] < h.sc[m]) && (m = l)
            (r ≤ h.n && h.sc[r] < h.sc[m]) && (m = r)
            m == i && break
            _swap!(h, i, m)
            i = m
        end
    end
    return h
end

# Gumbel(0,1) variate from a uniform `u` (the max-trick perturbation): `-log(-log u)`.
@inline _gumbel(u) = -log(-log(max(u, floatmin(Float64))))

"""
    distance_fixed_count(arch, positions; kernel, count, weight, delay, seed, allow_self=false, period=nothing, sources=…, targets=…)

Distance-dependent connectivity with an EXACT total edge `count`: samples `count` `(pre, post)` pairs
without replacement, with probability ∝ `kernel(distance(positions[pre], positions[post]))`, via the
Gumbel-max top-k trick (`score = log p + Gumbel`, keep the top `count`), reproducibly from the
counter-based RNG. Unlike per-pair Bernoulli [`distance_prob`](@ref) (random edge count), this fixes
the count exactly --- a fixed-degree connectivity. `sources`/`targets` restrict the pair set; pairs
with zero kernel probability are never selected. O(|sources|·|targets|) time, O(count) memory.
"""
function distance_fixed_count(arch::AbstractArchitecture, positions; kernel, count, weight, delay,
        seed::Unsigned, allow_self::Bool = false, period = nothing,
        sources = eachindex(positions), targets = eachindex(positions), index_type::Type = Int)
    cnt = Int(count)
    cnt ≥ 0 || throw(ArgumentError("count must be ≥ 0 (got $cnt)"))
    h = _TopK(cnt)
    for pre in sources
        ppos = positions[pre]
        for post in targets
            (!allow_self && pre == post) && continue
            p = kernel(distance(ppos, positions[post], period))
            p > 0 || continue                                 # zero-probability pairs are never selected
            u = draw_uniform(Float64, seed, pre, post)
            _topk_push!(h, log(p) + _gumbel(u), Int(pre), Int(post))
        end
    end
    wtype = typeof(to_weight(weight isa Function ? weight(1) : weight))
    dtype = typeof(_delayval(delay isa Function ? delay(1) : delay))   # Int (steps) or Float (ms)
    edges = Tuple{Int, Int, wtype, dtype}[]
    sizehint!(edges, h.n)
    for k in 1:h.n
        pre, post = h.pre[k], h.post[k]
        w = wtype(to_weight(weight isa Function ? weight(pre) : weight))
        d = _delayval(delay isa Function ? delay(pre) : delay)         # ms or steps; resolved at init
        push!(edges, (pre, post, w, d))
    end
    sort!(edges; by = e -> (e[1], e[2]))                      # CSR order (by source, then target)
    npost = length(positions)
    return SparseCSR(arch, edges; npre = npost, npost = npost, index_type = index_type)
end
export distance_fixed_count
