# * Ensemble (tensor) batching: run B independent network instances at once.
#
# This is the OUTER/ensemble axis (parameter / seed / input sweeps), distinct from the inner
# device axis. The chosen strategy is the SHARED-CONNECTIVITY tensor batch: one
# SparseCSR + one ring length L are BROADCAST across all B instances (read-only), while the
# per-neuron state gains a trailing batch dimension: V/refrac/spiked/spike_count/Isyn/g go
# (N,) → (N,B) and the delay ring (N_post,L) → (N_post,B,L). Each (neuron,batch) is independent,
# so the dense work is one fused KernelAbstractions megakernel over a 2-D (N,B) ndrange and the
# sparse scatter is one kernel over (pre,B) into the shared CSR.
#
# LIMITATIONS (the BrainPy wall, by design): connectivity STRUCTURE and per-synapse DELAYS must
# be IDENTICAL across the batch (probabilistic connectivity construction is variable-size and
# cannot be tensorised; the ring length L is sized once). Under shared structure you may sweep
# initial V, the external input, and the per-instance drive stream (and, later, synaptic
# weights). For per-instance TOPOLOGY use block-diagonal concatenation (one bigger network).
#
# Independence comes from the counter RNG's free batch axis: each column draws its drive from
# stream `streams[b]` (the Philox high counter word), so the B instances are independent and
# bit-reproducible. With `streams` all-zero the columns share the scalar (B=1) drive, which is
# the bit-exact reference: column b then equals a scalar solve with that column's input/v0.
#
# This is a SEPARATE path: the scalar B=1 engine (DewdropIntegrator + the broadcast/fused step)
# is untouched, so the existing suite stays bit-for-bit identical. The batched path uses KA
# kernels on every backend (CPU threads, JLArrays, CUDA), so it is correctness-testable without
# a GPU and delivers the ensemble throughput win on the device.

using KernelAbstractions: @kernel, @index, get_backend, synchronize
import Atomix

# Batched population: a 2-D StructArray with (N,B) columns (one per state var)
function _build_batched_population(arch, ::Type{T}, N::Int, B::Int, ::Val{names}) where {T, names}
    cols = ntuple(_ -> fill!(allocate(arch, T, N, B), zero(T)), Val(length(names)))
    return Population(StructArray(NamedTuple{names}(cols)))
end
batched_population(arch, model::AbstractNeuronModel, N::Integer, B::Integer) =
    _build_batched_population(arch, float_type(model), Int(N), Int(B), Val(statevars(typeof(model))))

# Batched delay ring: (N_post, B, L). The slot index mod(now+delay,L)+1 is SHARED across B
# (delay/L are per-structure), so only the target index gains a batch coordinate.
struct BatchedRing{A <: AbstractArray}
    slots::A     # (N_post, B, L)
    L::Int
end
Adapt.@adapt_structure BatchedRing
function BatchedRing(arch, ::Type{T}, N::Integer, B::Integer, maxdelay::Integer) where {T}
    L = Int(maxdelay) + 1
    return BatchedRing(fill!(allocate(arch, T, Int(N), Int(B), L), zero(T)), L)
end

# Batched synaptic state: ONE generic state (the (N,B) analogue of the scalar `SynState`) holding accumulators
# `(N,B)`, ring `(N_post,B,L)`, SHARED connectivity, the synapse model (dispatches the descriptor hooks),
# and the coefficients. The coefficients are EITHER a uniform NamedTuple (the no-sweep default: a scalar
# per coefficient, held by value in the isbits struct → GPU-uniform, byte-identical to a scalar solve) OR a
# `ColCoeffs` of length-B per-column device vectors (a per-member parameter sweep). The per-synapse physics
# is the SAME `_syn_kinetics` the scalar path uses, so batchability of ANY parameter is generic; no
# per-synapse batched type, no `_col`, no per-parameter plumbing.
struct BatchedSyn{S <: AbstractSynapseModel, ACC <: NamedTuple, R, C, K} <: AbstractSynapseState
    model::S
    acc::ACC
    buf::R
    conn::C
    coeffs::K
end
Adapt.@adapt_structure BatchedSyn

# Per-column coefficients (a per-member sweep): a NamedTuple of length-B device vectors, resolved to a
# per-member scalar NamedTuple at column `b`. A plain NamedTuple (the no-sweep default) ignores `b`; the
# same seam shape as `_resolve(m, i, b) = m` on the neuron side (bit-identical, zero-cost, GPU-uniform).
struct ColCoeffs{NT <: NamedTuple}
    cols::NT
end
Adapt.@adapt_structure ColCoeffs
@inline _resolve_coeffs(c::NamedTuple, b) = c
@inline _resolve_coeffs(c::ColCoeffs, b) = map(col -> (@inbounds col[b]), c.cols)

# (N,B) accumulator read/write at cell (i,b); the batched analogue of the scalar `_read_acc`/`_write_acc!`.
@inline _read_acc(acc::NamedTuple, i, b) = map(a -> (@inbounds a[i, b]), values(acc))
@inline function _write_acc!(acc::NamedTuple, i, b, vals::Tuple)
    map((a, x) -> (@inbounds a[i, b] = x), values(acc), vals)
    return nothing
end

# Batched streaming Poisson drive: the (N,B) analogue of `PoissonSourceState` (src/PoissonSource.jl).
# Generates the `n_ext` virtual sources' Poisson events once per step (SHARED across the B columns; `srcidx`
# is the source row, so the counter-RNG draw ignores the column) and scatters them through `extconn` into
# the wrapped synapse's (N,B,L) ring (all columns), so every member is driven by the SAME realization (the
# right default for a parameter sweep at fixed connectome). The constructor (dispatching on `PoissonSource`)
# lives in src/PoissonSource.jl, included after this file.
struct BatchedPoissonSourceState{IS <: AbstractSynapseState, EC, CC, BUF, MV, IDX, T} <: AbstractSynapseState
    inner::IS        # batched inner synapse state (its (N,B,L) ring is the deposit target)
    extconn::EC      # n_ext virtual sources → post (per-edge weights + delays)
    conn::CC         # empty outer CSR → the per-state network scatter is a no-op
    buf::BUF         # === inner.buf (BatchedRing)
    spiked::MV       # (n_ext, B) per-source firing mask
    srcidx::IDX      # (n_ext, B) source-row index (constant) → the draw is shared across columns
    p_spike::T
    seed::UInt64
end
Adapt.@adapt_structure BatchedPoissonSourceState

# Once per step: which virtual sources fire (counter RNG keyed by (seed, step, SOURCE); the same draw in
# every column), then scatter their events through `extconn` into the wrapped synapse's (N,B,L) ring.
@inline function _batched_synprestep!(s::BatchedPoissonSourceState, integ)
    n = integ.n
    @. s.spiked = draw_uniform(Float64, s.seed, n, s.srcidx) < s.p_spike   # shared across columns (srcidx = source row)
    batched_scatter!(s.buf, s.extconn, s.spiked, n; sync = false)          # deposit into all B columns of the inner ring
    return nothing
end
@inline _batched_synprestep!(::AbstractSynapseState, integ) = nothing       # non-drive synapses have no prestep
@inline _batched_synprestep_all!(::Tuple{}, integ) = nothing
@inline _batched_synprestep_all!(s::Tuple, integ) =
    (_batched_synprestep!(first(s), integ); _batched_synprestep_all!(Base.tail(s), integ))

# deliver/accumulate/decay delegate to the wrapped (batched) synapse, exactly like the scalar PoissonSource.
@inline _bsyn_one(s::BatchedPoissonSourceState, i, b, n, v, gtot, itot) = _bsyn_one(s.inner, i, b, n, v, gtot, itot)

# Batched SpikeSourceArray: the deterministic sibling of the batched Poisson source. The replay pattern is
# SHARED across the B columns (like the Poisson source's shared draw); each step broadcasts this step's firing
# column into the (n_ext, B) mask and scatters into all B columns of the inner ring.
struct BatchedSpikeSourceArrayState{IS <: AbstractSynapseState, EC, CC, BUF, MV, SP} <: AbstractSynapseState
    inner::IS
    extconn::EC
    conn::CC
    buf::BUF          # === inner.buf (BatchedRing)
    spiked::MV        # (n_ext, B) firing-mask scratch, filled from `spikes` each step
    spikes::SP        # n_ext × nsteps shared replay pattern
end
Adapt.@adapt_structure BatchedSpikeSourceArrayState
# `_make_batched_synstate(::SpikeSourceArray)` lives in PoissonSource.jl (which defines the model type, included
# after this file); it constructs this state, exactly as the Poisson-source builder does.

@inline function _batched_synprestep!(s::BatchedSpikeSourceArrayState, integ)
    n = integ.n
    s.spiked .= view(s.spikes, :, n + 1)                              # broadcast this step's column across all B columns
    batched_scatter!(s.buf, s.extconn, s.spiked, n; sync = false)
    return nothing
end
@inline _bsyn_one(s::BatchedSpikeSourceArrayState, i, b, n, v, gtot, itot) = _bsyn_one(s.inner, i, b, n, v, gtot, itot)

# One generic batched synapse-state builder (replaces the five hand-written `_make_batched_synstate` AND all
# of `_with_member_params`/`_mp`/`_col`): allocate the `(N,B)` accumulators named by `_syn_accumulators`,
# size the batched ring, and build the coefficients: uniform when nothing is swept, else per-column.
# `over` is a NamedTuple keyed by a physical field or a coefficient name → length-B values. (`PoissonSource`
# keeps its own more-specific method, which forwards `over` to the inner synapse.)
function _make_batched_synstate(arch, m::AbstractSynapseModel, conn, ::Type{T}, N, B, dt, over) where {T}
    names = _syn_accumulators(typeof(m))
    acc = NamedTuple{names}(ntuple(_ -> fill!(allocate(arch, T, Int(N), Int(B)), zero(T)), length(names)))
    buf = BatchedRing(arch, T, N, B, maximum(conn.delay; init = 0))
    return BatchedSyn(m, acc, buf, conn, _build_coeffs(arch, m, dt, Int(B), over, T))
end

# The batch coefficients: the uniform per-synapse coefficients (byte-identical to the scalar path; the exact
# `_syn_coeffs` wrapping, NO blanket `T`) when nothing is swept, else a `ColCoeffs` of length-B device
# vectors. `over` may key on a PHYSICAL parameter (τ, τr, τd, Erev, …) or a derived coefficient (a, decay_r,
# …); a physical sweep re-derives the coefficients per member through the synapse's own `_syn_coeffs`.
function _build_coeffs(arch, m, dt, B::Int, over, ::Type{T}) where {T}
    isempty(over) && return _syn_coeffs(m, dt, T)
    rows = [_coeffs_col(m, dt, over, b, T) for b in 1:B]
    ks = keys(first(rows))
    return ColCoeffs(NamedTuple{ks}(ntuple(j -> on_architecture(arch, T[r[ks[j]] for r in rows]), length(ks))))
end
# Member `b`'s coefficient NamedTuple: rebuild the synapse model with its swept PHYSICAL fields, derive the
# coefficients, then overlay any directly-swept COEFFICIENTS (e.g. `a`). An override key that is neither a
# model field nor a coefficient is a clear error rather than a silent no-op.
function _coeffs_col(m, dt, over, b, ::Type{T}) where {T}
    physk = filter(k -> hasfield(typeof(m), k), keys(over))
    coeffk = filter(k -> !hasfield(typeof(m), k), keys(over))
    mb = isempty(physk) ? m : _rebuild_syn(m, NamedTuple{physk}(map(k -> over[k][b], physk)))
    c = _syn_coeffs(mb, dt, T)
    all(k -> haskey(c, k), coeffk) ||
        error("syn_overrides key(s) $(filter(k -> !haskey(c, k), coeffk)) are neither a field of $(typeof(m)) nor a coefficient $(keys(c))")
    return isempty(coeffk) ? c : merge(c, NamedTuple{Tuple(coeffk)}(map(k -> T(over[k][b]), coeffk)))
end
# Rebuild an isbits synapse model with the named fields overridden (positional; no extra dependency). Each
# overridden value is converted to the field's type, so a Float64 sweep vector rebuilds a Float32 model
# (the swept coefficients are re-derived in `_build_coeffs`'s eltype `T` regardless).
@inline _rebuild_syn(m::M, nt::NamedTuple) where {M} =
    M.name.wrapper(map(f -> haskey(nt, f) ? convert(fieldtype(M, f), nt[f]) : getfield(m, f), fieldnames(M))...)

# Per-projection synaptic contribution for cell (i,b): deliver (read+clear the ring slot due at
# step n) → accumulate (gtot/itot) → decay, unrolled over the projection tuple at compile time.
@inline _bsyn_contribute(::Tuple{}, i, b, n, v, gtot, itot) = (v, gtot, itot)
@inline function _bsyn_contribute(syns::Tuple, i, b, n, v, gtot, itot)
    v, gtot, itot = _bsyn_one(first(syns), i, b, n, v, gtot, itot)
    return _bsyn_contribute(Base.tail(syns), i, b, n, v, gtot, itot)
end
# Generic batched per-cell contribution (replaces the five hand-written bodies): read + clear the ring slot,
# resolve this member's coefficients (uniform → the shared NamedTuple; swept → the per-column values), then
# apply the SAME `_syn_kinetics` the scalar `_syn_one` uses; the whole point of the collapse. Byte-identical
# to the prior `_col`-based bodies (a uniform coefficient equals `_col(scalar, b)`; a swept one equals
# `_col(vector, b)`).
@inline function _bsyn_one(s::BatchedSyn, i, b, n, v, gtot, itot)
    slot = mod(n, s.buf.L) + 1
    @inbounds due = s.buf.slots[i, b, slot]
    @inbounds s.buf.slots[i, b, slot] = zero(due)
    return _bsyn_apply(_syn_couple(typeof(s.model)), s, i, b, due, v, gtot, itot)
end
@inline _bsyn_apply(::Val{:jump}, s, i, b, due, v, gtot, itot) = (v + due, gtot, itot)
@inline function _bsyn_apply(::Union{Val{:current}, Val{:conductance}}, s, i, b, due, v, gtot, itot)
    c = _resolve_coeffs(s.coeffs, b)
    Δg, Δi, newacc = _syn_kinetics(s.model, _read_acc(s.acc, i, b), due, c, v)
    _write_acc!(s.acc, i, b, newacc)
    return (v, gtot + Δg, itot + Δi)
end

# `_binputval` (the per-cell input reader) now lives in src/Stimuli.jl (shared with `ConstantCurrent`). Every
# external input (input / drive / noise) is a stimulus in `integ.stimuli`, applied per cell via the ctx unrolls
# below; the per-column stream is `streams[b]` (the counter-RNG batch axis) so the B instances stay independent
# and reproducible (`streams` all-zero → column b reproduces the scalar batch-0 draw, the bit-exact reference).

# Fused Mode A: per-MEMBER model parameters in the (N,B) megakernel. A `BatchedModel` carries per-member
# override arrays (each length B); the kernel resolves the column's scalar model via `_resolve(m, i, b)`:
# the same seam Heterogeneous uses per-NEURON, here indexed by the batch column `b`. The connectome is the
# single shared CSR, so memory stays O(edges) AND the dynamics are fused (one launch over (N,B)).
struct BatchedModel{M <: AbstractNeuronModel, NT <: NamedTuple} <: AbstractNeuronModel
    base::M
    params::NT          # per-member override arrays, keyed by base field name (each length B)
end
Adapt.@adapt_structure BatchedModel                          # moves the per-member arrays to the device
Base.Broadcast.broadcastable(bm::BatchedModel) = Ref(bm)
statevars(::Type{BatchedModel{M, NT}}) where {M, NT} = statevars(M)
float_type(bm::BatchedModel) = float_type(bm.base)
_resting(bm::BatchedModel) = _resting(bm.base)
@inline _is_hetero(::BatchedModel) = false                   # allowed in batched init; resolved per-column in the kernel

# The leaf (scalar) model type under a `BatchedModel` base: a scalar model IS the leaf; a `Heterogeneous`
# base resolves (per neuron) to its underlying scalar type; `_resolve_member` rebuilds THIS type.
_leaf_model(::Type{M}) where {M <: AbstractNeuronModel} = M
_leaf_model(::Type{Heterogeneous{M, NT}}) where {M, NT} = M

# The per-(neuron, member) scalar model. Start from the base's per-neuron model `_resolve(base, i)` (the
# scalar model itself, or the neuron-`i` model of a `Heterogeneous` base), then override each swept field with
# its per-member value: `params.f[b]` for a length-B override (per member), or `params.f[i, b]` for an N×B
# override (per neuron AND member; e.g. "Δg_K on E only", 0 on I). `@generated` → specialises per (base
# type, override keys, override ranks); inlines to a plain isbits constructor, GPU-kernel-safe.
@generated function _resolve_member(bm::BatchedModel{Base, NT}, i, b) where {Base, NT}
    M = _leaf_model(Base)
    overkeys = NT.parameters[1]
    overtypes = NT.parameters[2].parameters
    args = map(fieldnames(M)) do f
        k = findfirst(==(f), overkeys)
        k === nothing ? :(getfield(base_i, $(QuoteNode(f)))) :
            (overtypes[k] <: AbstractMatrix ? :(@inbounds bm.params.$f[i, b]) : :(@inbounds bm.params.$f[b]))
    end
    return quote
        base_i = _resolve(bm.base, i)
        $(M)($(args...))
    end
end
@inline _resolve(bm::BatchedModel, i, b) = _resolve_member(bm, i, b)
@inline _resolve(bm::BatchedModel, i) = bm.base
# 3-arg seam: a scalar / per-neuron model ignores the batch column (bit-identical to the prior 2-arg path).
@inline _resolve(m::AbstractNeuronModel, i, b) = _resolve(m, i)

# The batched fused dense step: deliver + drive + accumulate + membrane + decay + threshold +
# reset + count for every (neuron,batch), in one launch over a 2-D (N,B) ndrange.
@kernel function _batched_fused_kernel!(V, refrac, spiked, spike_count, itotarr, gtotarr, stimuli, syns, m, dt, n, t, streams, aux)
    I = @index(Global, Cartesian)
    i = I[1]
    b = I[2]
    @inbounds begin
        v = V[i, b]
        r = refrac[i, b]
        z = zero(r)
        gtot = zero(eltype(V))
        str = streams[b]                            # this column's counter-RNG stream (batch axis)
        x0 = stim_ctx(m, v, i, b, n, t, dt, str)    # :current/:kick/:conductance ctx for cell (i,b)
        itot = oftype(gtot, _stim_itot(stimuli, gtot, x0))          # itot base: first :current ASSIGNS (was `_binputval`)
        v, gtot, itot = _bsyn_contribute(syns, i, b, n, v, gtot, itot)
        Δg, Δi = _stim_gtot(stimuli, x0)            # prescribed :conductance g(t) (no-op today; strong-zero identity)
        gtot = _addcond(gtot, Δg)
        itot = _addcond(itot, Δi)
        itotarr[i, b] = itot            # materialise the accumulators (one store each) so Trace(:itot)/:gtot
        gtotarr[i, b] = gtot            # record real values under the batched path (bit-identical to scalar fused)
        v += _stim_kick(stimuli, x0)                # :kick stimuli → v (per-column stream; was `_bdrive_kick`)
        m_i = _resolve(m, i, b)              # per-(neuron, member) model; `= m` for a scalar model
        # subthreshold (V, aux) advance + per-column SDE noise (refractory clamps V, no noise);
        # `aux` is `nothing` for a V-only model -> exactly the prior membrane_step (bit-identical).
        w0 = _aux_read(aux, i, b)
        v_adv, w_adv = _advance_unit(m_i, v, w0, gtot, itot, dt)
        v = ifelse(r > z, reset_value(m_i), v_adv + _stim_noise(stimuli, stim_ctx(m_i, v_adv, i, b, n, t, dt, str)))
        r = max(r - dt, z)
        s = (r ≤ z) & threshold(m_i, v)
        v = ifelse(s, reset_value(m_i), v)
        w_new = _spike_aux(m_i, w_adv, s)
        r = ifelse(s, refractory(m_i), r)
        V[i, b] = v
        refrac[i, b] = r
        _aux_write!(aux, w_new, i, b)
        spiked[i, b] = s
        spike_count[i, b] += s
    end
end

# Batched EDGE-PARALLEL scatter: one thread per (synapse, batch). The shared CSR is broadcast
# across B; the presynaptic source of edge e is read from the materialised `src` field (like the
# scalar kernel in Scatter.jl), and the deposit lands in THIS batch's ring slab
# `slots[post, b, slot]`. Distinct b touch disjoint slabs (no cross-batch contention); within a
# batch the atomic handles same-target collisions.
@kernel function _batched_scatter_edge_kernel!(slots, @Const(spiked), @Const(src), @Const(post), @Const(weight), @Const(delay), now, L)
    I = @index(Global, Cartesian)
    e = I[1]
    b = I[2]
    @inbounds begin
        pre = src[e]
        if spiked[pre, b]
            slot = mod(now + delay[e], L) + 1
            Atomix.@atomic slots[post[e], b, slot] += weight[e]
        end
    end
end

function batched_scatter!(buf::BatchedRing, conn::SparseCSR, spiked, now::Integer; sync::Bool = false)
    backend = get_backend(buf.slots)
    ne = nedges(conn)
    if ne > 0
        _batched_scatter_edge_kernel!(backend)(
            buf.slots, spiked, conn.src, conn.post, conn.weight, conn.delay, Int(now), buf.L;
            ndrange = (ne, size(spiked, 2)),
        )
    end
    sync && applicable(synchronize, backend) && synchronize(backend)
    return nothing
end

# CPU fast path. The batch axis is a NATURAL, contention-free parallelisation axis: distinct
# columns write disjoint slabs `slots[:, b, :]`, so threading over `b` needs NO atomics and each
# column accumulates in a fixed presynaptic order: DETERMINISTIC and bit-identical to a scalar
# serial run regardless of thread count (unlike the GPU atomic scatter, which is order-dependent).
function batched_scatter!(
        buf::BatchedRing{<:Array},
        conn::SparseCSR{<:Array, <:Array, <:Array, <:Array},
        spiked::AbstractArray, now::Integer; sync::Bool = false,
    )
    slots, L = buf.slots, buf.L
    rowptr, post, weight, delay = conn.rowptr, conn.post, conn.weight, conn.delay
    n = Int(now)
    N, B = size(spiked)
    Threads.@threads for b in 1:B
        @inbounds for pre in 1:N
            spiked[pre, b] || continue
            for e in rowptr[pre]:(rowptr[pre + 1] - 1)
                slots[post[e], b, mod(n + delay[e], L) + 1] += weight[e]
            end
        end
    end
    return nothing
end

@inline _bpropagate!(syn::AbstractSynapseState, integ) = (batched_scatter!(syn.buf, syn.conn, integ.spiked, integ.n; sync = false); nothing)
@inline _bpropagate_all!(::Tuple{}, integ) = nothing
@inline _bpropagate_all!(s::Tuple, integ) = (_bpropagate!(first(s), integ); _bpropagate_all!(Base.tail(s), integ))

# Batched compacted scatter (the per-column analogue of src/Compaction.jl)
# Each batch column has its OWN spiking set, so the compacted list is per-column: `active` is
# (npre, B) and `na` is (B,). The 2-level scatter is a 3-D launch over (maxdeg, max_na, B); a
# column with fewer than `max_na` active neurons early-exits its surplus `a` threads. `max_na` (the
# launch bound) is read to the host: the per-step sync, as in the scalar path.
struct BatchedCompactionScratch{AV, NV}
    active::AV    # (npre, B) Int32
    na::NV        # (B,) Int
end
Adapt.@adapt_structure BatchedCompactionScratch
BatchedCompactionScratch(arch, npre::Integer, B::Integer) =
    BatchedCompactionScratch(fill!(allocate(arch, Int32, Int(npre), Int(B)), Int32(0)), fill!(allocate(arch, Int, Int(B)), 0))

@kernel function _batched_compactify_kernel!(active, na, @Const(spiked))
    I = @index(Global, Cartesian)
    i = I[1]
    b = I[2]
    @inbounds if spiked[i, b]
        j = Atomix.@atomic na[b] += 1
        active[j, b] = i % eltype(active)
    end
end

@kernel function _batched_scatter_compacted_kernel!(slots, @Const(active), @Const(na), @Const(rowptr), @Const(post), @Const(weight), @Const(delay), now, L)
    I = @index(Global, Cartesian)
    j = I[1]
    a = I[2]
    b = I[3]
    @inbounds if a ≤ na[b]
        pre = active[a, b]
        rs = rowptr[pre]
        if j ≤ rowptr[pre + 1] - rs
            e = rs + (j - 1)
            slot = mod(now + delay[e], L) + 1
            Atomix.@atomic slots[post[e], b, slot] += weight[e]
        end
    end
end

function batched_compacted_scatter!(buf::BatchedRing, conn::SparseCSR, active, na, max_na::Integer, now::Integer; sync::Bool = false)
    (max_na == 0 || conn.maxdeg == 0) && return nothing
    backend = get_backend(buf.slots)
    _batched_scatter_compacted_kernel!(backend)(
        buf.slots, active, na, conn.rowptr, conn.post, conn.weight, conn.delay, Int(now), buf.L;
        ndrange = (conn.maxdeg, Int(max_na), size(buf.slots, 2)),
    )
    sync && applicable(synchronize, backend) && synchronize(backend)
    return nothing
end

@inline _bcompacted_propagate!(syn::AbstractSynapseState, active, na, max_na, now) =
    (batched_compacted_scatter!(syn.buf, syn.conn, active, na, max_na, now; sync = false); nothing)
@inline _bcompacted_propagate_all!(::Tuple{}, active, na, max_na, now) = nothing
@inline _bcompacted_propagate_all!(s::Tuple, active, na, max_na, now) =
    (_bcompacted_propagate!(first(s), active, na, max_na, now); _bcompacted_propagate_all!(Base.tail(s), active, na, max_na, now))

# Propagate dispatched on the batched compaction scratch (mirrors the scalar `_propagate_step!`).
@inline _batched_propagate_step!(::Nothing, integ) = _bpropagate_all!(integ.syns, integ)
function _batched_propagate_step!(c::BatchedCompactionScratch, integ)
    backend = get_backend(integ.spiked)
    fill!(c.na, 0)
    _batched_compactify_kernel!(backend)(c.active, c.na, integ.spiked; ndrange = size(integ.spiked))
    max_na = maximum(Array(c.na))                 # bulk (B,)-vector DtoH, then host max (no scalar index)
    _bcompacted_propagate_all!(integ.syns, c.active, c.na, max_na, integ.n)
    return nothing
end

# Batched integrator (mutable cache; only n/t mutate)
mutable struct BatchedIntegrator{M, ST, SM, A, T, BL, C, IT, GT, SY, MO, STR, CO, PG}
    const model::M
    const state::ST
    const stimuli::SM            # the unified AbstractStimulus tuple (input/drive/noise); see Stimuli.jl
    const dt::T
    n::Int
    t::T
    const tend::T
    const nsteps::Int            # fixed total step count: the loop bound (== monitor buffer width); see DewdropIntegrator
    const arch::A
    const spiked::BL
    const spike_count::C
    const itot::IT               # (N,B) input-current accumulator, materialised in-kernel (records :itot)
    const gtot::GT               # (N,B) conductance accumulator, materialised in-kernel (records :gtot)
    const syns::SY
    const monitors::MO
    const streams::STR
    const batch::Int
    const sync_every::Int        # periodic device sync cadence (0 = never; host reads still sync)
    const compaction::CO         # BatchedCompactionScratch (scatter = :compacted) or `nothing`
    const progress::PG           # the `progress` kwarg spec (:auto/Bool/String/Int); host-side, read by solve!
end
Adapt.@adapt_structure BatchedIntegrator

# Initial V for the batched state. `nothing`→EL, scalar→clamp, (lo,hi)→per-(neuron,batch) uniform
# drawn on the batch axis (so each column starts independently), vector/matrix→broadcast/copy.
@kernel function _batched_v0_kernel!(V, lo, hi, seed)
    I = @index(Global, Cartesian)
    i = I[1]
    b = I[2]
    @inbounds V[i, b] = lo + (hi - lo) * draw_uniform(eltype(V), seed, 0, i, b)
end
_init_batched_voltage!(V, EL, ::Nothing, ::Type{T}, seed) where {T} = (fill!(V, EL); V)
_init_batched_voltage!(V, EL, v0::Real, ::Type{T}, seed) where {T} = (fill!(V, T(v0)); V)
# a per-neuron vector v0 (length N) is broadcast across the batch: every column gets the same
# initial condition (matching the scalar path). A bare `copyto!(V, v0)` would linear-fill only
# column 1 of the (N,B) state and leave the rest at the allocation zero (a silent wrong v0).
function _init_batched_voltage!(V, EL, v0::AbstractVector, ::Type{T}, seed) where {T}
    size(V, 1) == length(v0) || throw(DimensionMismatch("v0 length $(length(v0)) ≠ N = $(size(V, 1))"))
    col = similar(V, T, size(V, 1))
    copyto!(col, collect(T, v0))                         # host → arch (bulk), per-neuron v0
    V .= reshape(col, :, 1)                              # same v0 in every batch column
    return V
end
# an explicit (N,B) matrix v0 sets each column independently; the shape must match.
function _init_batched_voltage!(V, EL, v0::AbstractMatrix, ::Type{T}, seed) where {T}
    size(V) == size(v0) || throw(DimensionMismatch("v0 size $(size(v0)) ≠ batched state $(size(V))"))
    copyto!(V, v0)
    return V
end
function _init_batched_voltage!(V, EL, v0::Tuple{<:Real, <:Real}, ::Type{T}, seed) where {T}
    backend = get_backend(V)
    _batched_v0_kernel!(backend)(V, T(v0[1]), T(v0[2]), seed % UInt64; ndrange = size(V))
    applicable(synchronize, backend) && synchronize(backend)
    return V
end

function _batched_init(
        prob::DewdropNetwork, alg::FixedStep, B::Int;
        record = nothing, v0 = nothing, v0_seed::Unsigned = 0x05eed00d % UInt64,
        input = nothing, streams = nothing, sync_every::Integer = _DEFAULT_WINDOW,
        scatter::Symbol = :edge, progress = :auto, syn_overrides = nothing, model_overrides = nothing
    )
    arch = prob.arch
    T = float_type(prob.model)
    N = prob.n
    dt = T(alg.dt)
    _check_drive(prob.drive, dt)
    # A single-group `Heterogeneous` model resolves per-NEURON through the `_resolve(m,i,b)` seam the
    # batched megakernel already calls per cell, so it runs in ONE (N,B) launch (the aux/`w` column and
    # `_resolve` delegate through `Heterogeneous` unchanged). Only a `MultiModel` (several groups) needs the
    # per-group launches the single-launch batched kernel cannot express; reject just that.
    prob.model isa MultiModel &&
        throw(ArgumentError("MultiModel (multiple model groups) is not yet supported in batched runs; run separate solves per group, or use block-diagonal batching (`batch([nets…])`)"))
    # STDP in the ensemble batch needs per-column (nedges,B) weights + traces (a follow-on); the
    # shared-CSR batch deliberately keeps one immutable weight set, so reject plastic projections here.
    any(p -> p.plasticity !== nothing, prob.projections) &&
        throw(ArgumentError("STDP (plastic projections) are not yet supported in batched runs; run B sequential plastic solves instead"))
    # `model_overrides` (a NamedTuple of per-member (B) or per-(neuron,member) (N×B) field arrays) wraps the
    # model in a `BatchedModel`, so any neuron-model parameter can vary per member over the shared connectome.
    # The override arrays MUST be uploaded to `arch` first, exactly like `syn_overrides`, `input`, and
    # `streams` below. `@adapt_structure BatchedModel` only rewraps already-device arrays at launch
    # (CuArray → CuDeviceArray); it cannot upload a host array, so a host override would reach the GPU kernel
    # as a non-bitstype argument and fail to compile (`on_architecture` is a no-op on CPU, hence CPU "worked").
    model = (model_overrides === nothing || isempty(model_overrides)) ? prob.model :
        BatchedModel(prob.model, map(a -> on_architecture(arch, a), model_overrides))
    state = batched_population(arch, model, N, B)
    _init_batched_voltage!(state.state.V, T(_resting(model)), v0, T, v0_seed)   # `_resting` handles BatchedModel (no `.EL`)
    spiked = fill!(allocate(arch, Bool, N, B), false)
    spike_count = fill!(allocate(arch, Int, N, B), 0)
    itot = fill!(allocate(arch, T, N, B), zero(T))      # materialised by the kernel so Trace(:itot)/:gtot
    gtot = fill!(allocate(arch, T, N, B), zero(T))      # record real values per (neuron, member)
    # per-projection batched synapse states; `syn_overrides[j]` (a NamedTuple like `(; a = avec)`) makes
    # chosen scalar params of projection j vary per member (the generic synaptic sweep, e.g. `delta`).
    ov(j) = syn_overrides === nothing ? (;) : get(syn_overrides, j, (;))
    syns = ntuple(length(prob.projections)) do j
        p = prob.projections[j]
        _make_batched_synstate(arch, p.synapse, _resolve_delays(p.conn, dt), T, N, B, dt, ov(j))
    end
    inp = input === nothing ? prob.input : on_architecture(arch, to_current(input))
    str = on_architecture(arch, collect(Int, streams === nothing ? (0:(B - 1)) : streams))
    # guard the per-column data shapes against B (the kernel reads streams[b]/input[i,b] @inbounds)
    length(str) == B || throw(ArgumentError("streams has length $(length(str)) but batch B = $B"))
    inp isa AbstractMatrix && size(inp) != (N, B) && throw(ArgumentError("input matrix $(size(inp)) ≠ (N, B) = ($N, $B)"))
    inp isa AbstractVector && length(inp) != N && throw(ArgumentError("input vector length $(length(inp)) ≠ N = $N"))
    nsteps = round(Int, (prob.tspan[2] - prob.tspan[1]) / dt)
    monitors = _make_batched_monitors(record, arch, T, N, B, nsteps, dt)
    compaction = scatter === :compacted ? BatchedCompactionScratch(arch, N, B) :
        scatter === :edge ? nothing :
        throw(ArgumentError("scatter must be :edge or :compacted (got :$scatter)"))
    _validate_stimuli(prob.stimuli, N, nsteps, dt)             # shape-check the `stimuli =` extras vs (N, nsteps, dt)
    stimuli = _assemble_stimuli(inp, prob.drive, prob.noise, prob.stimuli)    # ConstantCurrent(inp) + drive? + noise? + extras
    return BatchedIntegrator(
        model, state, stimuli, dt, 0, prob.tspan[1], prob.tspan[2], nsteps, arch,
        spiked, spike_count, itot, gtot, syns, monitors, str, B, Int(sync_every), compaction, progress,
    )
end

function _batched_step!(integ::BatchedIntegrator)
    _batched_synprestep_all!(integ.syns, integ)         # streaming drives → fill the (N,B) ring (shared across columns)
    st = integ.state.state
    V = st.V
    backend = get_backend(V)
    _batched_fused_kernel!(backend)(
        V, st.refrac, integ.spiked, integ.spike_count, integ.itot, integ.gtot, integ.stimuli,
        integ.syns, integ.model, integ.dt, integ.n, _step_time(integ), integ.streams, _aux_col(st, integ.model);
        ndrange = size(V),
    )
    _batched_propagate_step!(integ.compaction, integ)   # edge-parallel, or compacted per column
    _record_all!(integ.monitors, integ)
    return nothing
end

function CommonSolve.step!(integ::BatchedIntegrator)
    _batched_step!(integ)
    integ.n += 1
    integ.t += integ.dt
    _maybe_sync!(integ)
    return integ
end

function CommonSolve.solve!(integ::BatchedIntegrator)
    rep = _progress_reporter(integ.progress, _progress_total(integ))   # nothing ⇒ every hook no-ops
    _progress_start!(rep)
    while integ.n < integ.nsteps               # integer count: exactly fills the monitor buffer (no float-time drift)
        step!(integ)
        _progress_step!(rep, integ.n)
    end
    _progress_finish!(rep)
    _finalize_all!(integ.monitors)
    return BatchedSolution(integ)
end

"""
    BatchedSolution

The result of a batched run: final `(N,B)` SoA state, `(N,B)` per-cell `spike_count`, the batch
size, and `record` (a NamedTuple of batched monitor results). Column `b` is the b-th ensemble
instance. See [`firing_rate`](@ref).
"""
struct BatchedSolution{ST, C, T, R}
    state::ST
    spike_count::C
    nsteps::Int
    dt::T
    tspan::Tuple{T, T}
    batch::Int
    record::R
end
function BatchedSolution(integ::BatchedIntegrator)
    return BatchedSolution(
        integ.state, integ.spike_count, integ.n, integ.dt,
        # tspan from the fixed window (tend, nsteps), NOT the drifted Float32 `integ.t` (see DewdropSolution).
        (integ.tend - integ.nsteps * integ.dt, integ.tend), integ.batch, map(_result, integ.monitors),
    )
end
export BatchedSolution

duration(sol::BatchedSolution) = sol.nsteps * sol.dt
firing_rate(sol::BatchedSolution) = sol.spike_count ./ duration(sol)   # (N,B) per-cell rate

# Batched recording
# A batched monitor inserts B as a MIDDLE axis with TIME trailing: per-unit window/store are
# (n_out, B, Wcols)/(n_out, B, ncols), aggregate are (B, Wcols)/(B, ncols). Because time stays
# the LAST axis the windowed device→host flush is the same rank-generic bulk copy as the scalar
# path. The spec types (Trace/Spikes/Aggregate), source descriptors, `_read`, `_resolve_idx`,
# and `RecordResult` are reused verbatim; only the buffer rank and the (n,b) selection differ.

@inline _lead(::AbstractArray{<:Any, Nd}) where {Nd} = ntuple(_ -> Colon(), Nd - 1)   # all axes but time
@inline _timecol(A, c) = @inbounds view(A, _lead(A)..., c)
@inline _timerange(A, r) = @inbounds view(A, _lead(A)..., r)

mutable struct BatchedWindow{W <: AbstractArray, H <: AbstractArray}
    const window::W      # arch-resident staging (…, Wcols), time trailing
    const store::H        # host-resident result (…, ncols)
    const Wcols::Int
    filled::Int
    flushed::Int
end
# adapt ONLY the device window; the host store stays a host Array (the device-window/host-store split)
Adapt.adapt_structure(to, wb::BatchedWindow) =
    BatchedWindow(adapt(to, wb.window), wb.store, wb.Wcols, wb.filled, wb.flushed)
function BatchedWindow(arch, ::Type{E}, leaddims::Dims, ncols::Integer) where {E}
    Wcols = min(Int(ncols), _DEFAULT_WINDOW)
    window = fill!(allocate(arch, E, leaddims..., Wcols), zero(E))
    store = fill!(Array{E}(undef, leaddims..., Int(ncols)), zero(E))
    return BatchedWindow(window, store, Wcols, 0, 0)
end
@inline _bwcol(wb::BatchedWindow) = wb.filled + 1
@inline _bwindowslice(window::Array, filled) = _timerange(window, 1:filled)
function _bwindowslice(window, filled)                       # device → host, one bulk copy of the full window
    h = Array{eltype(window)}(undef, size(window))
    copyto!(h, window)
    return _timerange(h, 1:filled)
end
function flush!(wb::BatchedWindow)
    wb.filled == 0 && return wb
    @inbounds copyto!(_timerange(wb.store, (wb.flushed + 1):(wb.flushed + wb.filled)), _bwindowslice(wb.window, wb.filled))
    wb.flushed += wb.filled
    wb.filled = 0
    return wb
end
@inline function _badvance!(wb::BatchedWindow)
    wb.filled += 1
    wb.filled == wb.Wcols && flush!(wb)
    return nothing
end
@inline _bdue(m, integ) = (integ.n % m.every == 0) && (m.buf.flushed + m.buf.filled < size(m.buf.store)[end])

# unit selection on a batched (N,B) source: all rows, or a row subset (batch axis kept)
@inline _bselect(arr, ::Colon) = arr
@inline _bselect(arr, idx) = @inbounds @view arr[idx, :]

struct BatchedPerUnit{S, I, W <: BatchedWindow}
    src::S
    idx::I
    buf::W
    every::Int
end
Adapt.@adapt_structure BatchedPerUnit
struct BatchedAgg{S, I, W <: BatchedWindow, R}
    src::S
    idx::I
    buf::W
    every::Int
    n::Int
end
function Adapt.adapt_structure(to, m::BatchedAgg{S, I, W, R}) where {S, I, W, R}
    src, idx, buf = adapt(to, m.src), adapt(to, m.idx), adapt(to, m.buf)
    return BatchedAgg{typeof(src), typeof(idx), typeof(buf), R}(src, idx, buf, m.every, m.n)
end

@inline function record!(m::BatchedPerUnit, integ)
    _bdue(m, integ) || return nothing
    @inbounds _timecol(m.buf.window, _bwcol(m.buf)) .= _bselect(_read(m.src, integ), m.idx)
    _badvance!(m.buf)
    return nothing
end

# aggregate: reduce the selected units to one scalar PER BATCH (one thread per column b)
@kernel function _bagg_kernel!(window, col, @Const(vals), domean, ncells)
    b = @index(Global)
    acc = zero(eltype(window))
    @inbounds for k in 1:size(vals, 1)
        acc += vals[k, b]
    end
    @inbounds window[b, col] = domean ? acc / ncells : acc
end
function _baggregate!(buf::BatchedWindow, vals, ::BatchedAgg{S, I, W, R}, col, ncells) where {S, I, W, R}
    backend = get_backend(buf.window)
    _bagg_kernel!(backend)(buf.window, col, vals, R === :mean, ncells; ndrange = size(buf.window, 1))
    return nothing
end
@inline function record!(m::BatchedAgg, integ)
    _bdue(m, integ) || return nothing
    _baggregate!(m.buf, _bselect(_read(m.src, integ), m.idx), m, _bwcol(m.buf), m.n)
    _badvance!(m.buf)
    return nothing
end

# spec → batched monitor (mirrors scalar `_materialize`, reusing _srcof/_resolve_idx/_nsel/_ncols). `dt`
# (the solve step, ms) is threaded so time-aware monitors (Welch) can resolve a sampling rate; the
# per-unit/aggregate monitors ignore it.
function _bmaterialize(spec::Trace, arch, ::Type{T}, N, B, nsteps, dt) where {T}
    idx = _resolve_idx(arch, spec.of)   # :itot/:gtot now materialised per (neuron, member) in the batched kernel
    buf = BatchedWindow(arch, T, (_nsel(N, idx), Int(B)), _ncols(nsteps, spec.every))
    return BatchedPerUnit(_srcof(spec), idx, buf, spec.every)
end
function _bmaterialize(spec::Spikes, arch, ::Type{T}, N, B, nsteps, dt) where {T}
    idx = _resolve_idx(arch, spec.of)
    buf = BatchedWindow(arch, Bool, (_nsel(N, idx), Int(B)), _ncols(nsteps, spec.every))
    return BatchedPerUnit(SpikeSrc(), idx, buf, spec.every)
end
function _bmaterialize(spec::Aggregate, arch, ::Type{T}, N, B, nsteps, dt) where {T}
    idx = _resolve_idx(arch, spec.inner.of)
    buf = BatchedWindow(arch, T, (Int(B),), _ncols(nsteps, spec.every))
    src = spec.inner isa Spikes ? SpikeSrc() : _srcof(spec.inner)
    return BatchedAgg{typeof(src), typeof(idx), typeof(buf), spec.reducer}(src, idx, buf, spec.every, _nsel(N, idx))
end
_bmaterialize(::Probe, arch, ::Type{T}, N, B, nsteps, dt) where {T} =
    error("Probe is not yet supported in batched runs (needs an (n,B) batched layout); use Trace/Spikes/Aggregate")

_result(m::BatchedPerUnit{<:SpikeSrc}) = RecordResult(m.buf.store, m.idx, m.every, :spikes)
_result(m::BatchedPerUnit) = RecordResult(m.buf.store, m.idx, m.every, :trace)
_result(m::BatchedAgg) = RecordResult(m.buf.store, nothing, m.every, :aggregate)

# Batched streaming temporal monitors (MADev / Welch). Fold each step's selected (n_out, B) slice into a
# streaming reducer (TemporalReducers.jl): no window/store, so memory is O(maxlag)/O(nfft), not O(nsteps).
# Skip the first `start` recorded steps (transient), then record every `every`-th step, indexing recorded
# samples 1, 2, …. `update!`/`result` run host-orchestrated (kernel launch / batched rfft + broadcasts), so
# the monitor is never passed into a kernel; its reducer arrays are already arch-resident (adapt is identity).
_srcof_var(var::Symbol) = var in (:gtot, :itot) ? AccumSrc{var}() : StateSrc{var}()

mutable struct BatchedTemporalMonitor{R, S, I}
    reducer::R
    src::S
    idx::I
    start::Int       # transient: ignore steps with integ.n ≤ start
    every::Int
    k::Int           # recorded-sample counter (1-based index handed to the reducer)
    kind::Symbol     # :madev | :welch
end

function _bmaterialize(spec::MADev, arch, ::Type{T}, N, B, nsteps, dt) where {T}
    idx = _resolve_idx(arch, spec.of)
    red = StreamingMADev(arch, T, _nsel(N, idx), Int(B), spec.lags)
    return BatchedTemporalMonitor(red, _srcof_var(spec.var), idx, spec.transient, spec.every, 0, :madev)
end
function _bmaterialize(spec::Welch, arch, ::Type{T}, N, B, nsteps, dt) where {T}
    idx = _resolve_idx(arch, spec.of)
    fs = 1 / (float(dt) * spec.every)        # recorded sampling rate
    red = StreamingWelch(arch, T, _nsel(N, idx), Int(B), fs, spec.f_min)
    return BatchedTemporalMonitor(red, _srcof_var(spec.var), idx, spec.transient, spec.every, 0, :welch)
end
# SpikeRate / Fano consume the (N,B) spike mask (`SpikeSrc`); their recorded rate is fs = 1/(every·dt).
function _bmaterialize(spec::SpikeRate, arch, ::Type{T}, N, B, nsteps, dt) where {T}
    idx = _resolve_idx(arch, spec.of)
    red = StreamingRate(arch, T, _nsel(N, idx), Int(B), float(dt) * spec.every)
    return BatchedTemporalMonitor(red, SpikeSrc(), idx, spec.transient, spec.every, 0, :rate)
end
function _bmaterialize(spec::Fano, arch, ::Type{T}, N, B, nsteps, dt) where {T}
    idx = _resolve_idx(arch, spec.of)
    red = StreamingFano(arch, T, _nsel(N, idx), Int(B), spec.taus, float(dt) * spec.every, fld(Int(nsteps), spec.every) + 2)
    return BatchedTemporalMonitor(red, SpikeSrc(), idx, spec.transient, spec.every, 0, :fano)
end

@inline function record!(m::BatchedTemporalMonitor, integ)
    # `integ.n` is 0-based here (incremented after `_record_all!`), matching the Trace monitor's `_bdue`:
    # record n = start, start+every, … (`start` = transient samples skipped; start = 0 records every sample).
    (integ.n >= m.start && (integ.n - m.start) % m.every == 0) || return nothing
    m.k += 1
    update!(m.reducer, _bselect(_read(m.src, integ), m.idx), m.k)
    return nothing
end
@inline _finalize!(::BatchedTemporalMonitor) = nothing      # streaming reducer: nothing to flush

function _result(m::BatchedTemporalMonitor)
    r = result(m.reducer, m.k)
    data = m.kind === :welch ? permutedims(r, (2, 3, 1)) : r   # welch (nfreq,n_out,B) → (n_out,B,nfreq); madev already (n_out,B,nlags)
    return RecordResult(data, m.idx, m.every, m.kind)
end

_make_batched_monitors(::Nothing, arch, ::Type{T}, N, B, nsteps, dt) where {T} = (;)
_make_batched_monitors(record::NamedTuple, arch, ::Type{T}, N, B, nsteps, dt) where {T} =
    map(spec -> _bmaterialize(spec, arch, T, N, B, nsteps, dt), record)
