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
        DV <: AbstractVector{<:Integer},
    } <: AbstractConnectivity
    rowptr::RV   # length npre + 1, 1-based: out-edges of pre i are rowptr[i]:rowptr[i+1]-1
    post::PV     # length nedges, postsynaptic target indices
    weight::WV   # length nedges, per-synapse weights
    delay::DV    # length nedges, per-synapse conduction delays (integer time steps)
    npre::Int
    npost::Int
end
Adapt.@adapt_structure SparseCSR
export SparseCSR

function SparseCSR(arch::AbstractArchitecture, edges; npre::Integer, npost::Integer)
    Tw = isempty(edges) ? Float32 : typeof(edges[1][3])
    nE = length(edges)

    counts = zeros(Int, npre)
    for e in edges
        counts[e[1]] += 1
    end
    rowptr = Vector{Int}(undef, npre + 1)
    rowptr[1] = 1
    for i in 1:npre
        rowptr[i + 1] = rowptr[i] + counts[i]
    end

    post = Vector{Int}(undef, nE)
    weight = Vector{Tw}(undef, nE)
    delay = Vector{Int}(undef, nE)
    fillpos = copy(rowptr)
    for e in edges
        pre, p, w, d = e
        idx = fillpos[pre]
        post[idx] = p
        weight[idx] = w
        delay[idx] = d
        fillpos[pre] += 1
    end

    return SparseCSR(
        on_architecture(arch, rowptr),
        on_architecture(arch, post),
        on_architecture(arch, weight),
        on_architecture(arch, delay),
        Int(npre), Int(npost),
    )
end

npre(c::SparseCSR) = c.npre
npost(c::SparseCSR) = c.npost
nedges(c::SparseCSR) = length(c.post)

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
    fixed_prob(arch, npre, npost, p; weight, delay, seed, allow_self=true)

Random connectivity in which each `(pre, post)` edge is present with probability `p`,
sampled reproducibly from the counter-based RNG keyed by `(pre, post)` (so a given `seed`
yields a fixed, copyable connectome). `weight` and `delay` may be scalars or functions of
the presynaptic index `pre` --- the latter gives excitatory/inhibitory neurons signed
weights. Returns a [`SparseCSR`](@ref).
"""
function fixed_prob(arch::AbstractArchitecture, npre::Integer, npost::Integer, p::Real;
        weight, delay, seed::Unsigned, allow_self::Bool = true)
    wtype = weight isa Function ? typeof(weight(1)) : typeof(weight)
    edges = Tuple{Int, Int, wtype, Int}[]
    pT = Float64(p)
    for pre in 1:npre
        w = wtype(weight isa Function ? weight(pre) : weight)
        d = Int(delay isa Function ? delay(pre) : delay)
        for post in 1:npost
            (!allow_self && pre == post) && continue
            draw_uniform(Float64, seed, pre, post) < pT || continue
            push!(edges, (pre, Int(post), w, d))
        end
    end
    return SparseCSR(arch, edges; npre = npre, npost = npost)
end
export fixed_prob
