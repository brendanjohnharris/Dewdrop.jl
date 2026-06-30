# * Compacted scatter (opt-in device fast path for the sparse-firing regime)
#
# The edge-parallel scatter (Scatter.jl) launches one thread per synapse and lets the idle ones
# (whose source did not spike) exit early. At realistic firing (rate*dt ≈ 0.1-1% of neurons per
# step) that is mostly wasted threads. The compacted scatter instead processes ONLY active
# synapses, in two phases:
#   1. compactify --- build a dense list `active` of the spiking neurons (atomic-append), with the
#      count in `na`. One pass over N; cheap at sparse firing.
#   2. a 2-LEVEL scatter --- launch (maxdeg × na) threads, thread (j, a) handling the j-th edge of
#      active neuron `active[a]` (rows shorter than maxdeg early-exit). Every launched group maps
#      to a real spiking neuron, so the device does work proportional to ACTIVE synapses, not all
#      synapses --- measured up to ~30× over edge-parallel at high degree + sparse firing.
#
# Cost: the launch needs the active count `na` on the HOST (to size the 2-level grid), i.e. one
# device→host read per step. That reintroduces a per-step sync (which the fused device step removed)
# --- negligible when the step is scatter-bound (the regime where compaction is used), but a net
# loss in the launch-bound small-N regime, so this path is OPT-IN (`solve(...; scatter = :compacted)`)
# and the performance advisor only suggests it when the regime fits.

# Per-projection compaction is shared across a population's projections (they scatter the same
# spike mask), so the scratch lives on the integrator and `active`/`na` are filled ONCE per step.
struct CompactionScratch{AV, NV}
    active::AV    # (npre,) device Int32: compacted spiking-neuron indices (only 1:na are valid)
    na::NV        # (1,) device Int: number of spiking neurons this step
end
Adapt.@adapt_structure CompactionScratch

function CompactionScratch(arch, npre::Integer)
    return CompactionScratch(fill!(allocate(arch, Int32, Int(npre)), Int32(0)), fill!(allocate(arch, Int, 1), 0))
end

# Atomic-append the spiking neurons into `active`; `na[1]` holds the running (and final) count.
# `na` must be zeroed before launch. The atomic returns the post-increment value → a 1-based slot.
@kernel function _compactify_kernel!(active, na, @Const(spiked))
    i = @index(Global)
    @inbounds if spiked[i]
        j = Atomix.@atomic na[1] += 1
        active[j] = i % eltype(active)
    end
end

# 2-level compacted scatter: thread (j, a) deposits the j-th edge of spiking neuron active[a].
@kernel function _scatter_compacted_kernel!(
        slots, @Const(active), @Const(rowptr), @Const(post), @Const(weight), @Const(delay), now, L
    )
    I = @index(Global, Cartesian)
    j = I[1]
    a = I[2]
    @inbounds begin
        pre = active[a]
        rs = rowptr[pre]
        if j ≤ rowptr[pre + 1] - rs                  # row may be shorter than maxdeg
            e = rs + (j - 1)
            slot = mod(now + delay[e], L) + 1
            Atomix.@atomic slots[post[e], slot] += weight[e]
        end
    end
end

# Read the active count to the host WITHOUT scalar indexing (a 1-element bulk DtoH, then host index
# --- `na[1]` directly would scalar-index, forbidden under CUDA / JLArrays allowscalar(false)).
@inline _read_na(na) = @inbounds Array(na)[1]

# Scatter only the active synapses into the ring (the compacted analogue of `scatter!`).
function compacted_scatter!(buf::DelayBuffer, conn::SparseCSR, active, na::Integer, now::Integer; sync::Bool = false)
    (na == 0 || conn.maxdeg == 0) && return nothing
    backend = get_backend(buf.slots)
    _scatter_compacted_kernel!(backend)(
        buf.slots, active, conn.rowptr, conn.post, conn.weight, conn.delay, Int(now), buf.L;
        ndrange = (conn.maxdeg, Int(na)),
    )
    sync && applicable(synchronize, backend) && synchronize(backend)
    return nothing
end

@inline _compacted_propagate!(syn::AbstractSynapseState, active, na, now) =
    (compacted_scatter!(syn.buf, syn.conn, active, na, now; sync = false); nothing)
@inline _compacted_propagate_all!(::Tuple{}, active, na, now) = nothing
@inline _compacted_propagate_all!(s::Tuple, active, na, now) =
    (_compacted_propagate!(first(s), active, na, now); _compacted_propagate_all!(Base.tail(s), active, na, now))

# The propagate step, dispatched on the integrator's compaction scratch: `nothing` → the
# edge-parallel scatter (no per-step sync); a `CompactionScratch` → compactify once, read `na`
# (the per-step device→host sync), then the 2-level compacted scatter for every projection.
@inline _propagate_step!(::Nothing, integ) = _propagate_all_nosync!(integ.syns, integ)
function _propagate_step!(c::CompactionScratch, integ)
    backend = get_backend(integ.spiked)
    fill!(c.na, 0)
    _compactify_kernel!(backend)(c.active, c.na, integ.spiked; ndrange = length(integ.spiked))
    na = _read_na(c.na)
    _compacted_propagate_all!(integ.syns, c.active, na, integ.n)
    return nothing
end
