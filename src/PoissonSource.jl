# * Generic streaming Poisson drive: `PoissonSource{S}` wraps any synapse model `S` and turns it into an
# external drive of virtual Poisson sources (rate `rate` Hz), wired to post-neurons by `extconn`. Each step
# it generates the sources' Poisson spikes and scatters them (through the same `scatter!` + delay-buffer +
# `_deliver!` pipeline the network uses for real spikes) into the wrapped synapse's state. So the
# postsynaptic kinetics are exactly `S`'s (delta / CUBA / COBA / dual-exp / …): the source is the input
# statistics + wiring, the synapse is the response. Streaming (O(N) state, no precomputed conductance
# matrix), and (because it routes through `scatter!`) CPU and device alike.
#
# The state delegates `_deliver!`/`_accumulate!`/`_decay!`/`_syn_one` to the wrapped synapse's state
# (`inner`); only `_synprestep!` is new (generate + scatter the Poisson events). The outer `conn` is an empty
# CSR, so the per-state network scatter (`scatter!(syn.buf, syn.conn, …)`) is a no-op; the real wiring
# lives in `extconn`, used only by the once-per-step generator.

"""
    PoissonSource(synapse, extconn; rate, seed)

An external drive of virtual Poisson sources wired to the network by `extconn` (an `n_ext × N` CSR carrying
per-edge weights and delays), delivering through any postsynaptic `synapse` model. Each step generates the
sources' Poisson spikes and scatters them into `synapse`'s conductance state via the standard delay-buffer
pipeline: no precomputed conductance matrix. The postsynaptic kinetics are exactly `synapse`'s, so the same drive composes with `DeltaSynapse`,
`CurrentSynapse`, `ConductanceSynapse`, `DualExpSynapse`, …. Add to a network as
`Projection(PoissonSource(...), empty_csr)`.

!!! note "`rate` here is in Hz"
    This is the one deliberate exception to the coherent unit system: `rate` is in **Hz**, converted
    internally (`rate·dt/1000`, with `dt` in ms), because an external drive is conventionally quoted in
    Hz. [`PoissonDrive`](@ref) and [`InhomogeneousPoisson`](@ref) take the canonical per-`dt`-unit rate
    (kHz) instead, so 20 Hz is `rate = 20.0` here and `rate = 0.02` there.
"""
struct PoissonSource{S <: AbstractSynapseModel, C, T} <: AbstractSynapseModel
    synapse::S       # inner postsynaptic synapse model (its kinetics define the response)
    extconn::C       # n_ext virtual sources → post indices (per-edge weights + delays)
    rate::T          # Poisson rate (Hz) per source
    seed::UInt64
end
# the firing stream is separated from the wiring builders, so `seed` may be reused across both
PoissonSource(synapse::AbstractSynapseModel, extconn; rate, seed = 0x9e3779b97f4a7c15) =
    PoissonSource(synapse, extconn, Float64(rate), domain_seed(seed, DOMAIN_POISSON))
export PoissonSource

# A plastic wrapper (PlasticState/BatchedSTDState) would hide the event generator from
# `_synprestep!` and wrap the empty outer CSR, silently killing the drive (src/Plasticity.jl).
is_source_synapse(::PoissonSource) = true

# Per-step firing probability. Above 1 the uniform draw always succeeds and the source degenerates to
# a deterministic every-step spike train with no Poisson statistics, so refuse it. (`PoissonDrive` has
# the same guard in `_check_drive`, against its own sampler's underflow cliff.)
function _poisson_p(rate, dt, ::Type{T}) where {T}
    p = rate * dt / 1000
    p <= 1 || throw(ArgumentError(
        "PoissonSource rate = $rate Hz at dt = $dt ms gives a per-step spike probability of $p > 1: " *
            "every source would fire every step. Use rate <= $(1000 / dt) Hz, or spread the drive over more sources."))
    return T(p)
end

# State: the wrapped synapse's real state (`inner`) + the Poisson generator (extconn, firing mask, p_spike,
# seed). `conn`/`buf` are exposed so the engine's per-state network scatter finds them: `conn` is empty
# (a no-op), `buf` aliases `inner.buf` (the deposit target shared with the delegated `_deliver!`).
struct PoissonSourceState{IS <: AbstractSynapseState, EC, CC, BUF, MV, T} <: AbstractSynapseState
    inner::IS
    extconn::EC
    conn::CC         # empty CSR → network scatter no-op (interface satisfaction)
    buf::BUF         # === inner.buf: events land here via `_synprestep!`, are read by `_deliver!(inner, …)`
    spiked::MV       # preallocated per-source firing mask (length n_ext)
    p_spike::T
    seed::UInt64
end
Adapt.@adapt_structure PoissonSourceState

function _make_synstate(arch, syn::PoissonSource, conn, ::Type{T}, N, dt) where {T}
    ext = _resolve_delays(syn.extconn, dt)                    # ms → steps at the solve dt
    inner = _make_synstate(arch, syn.synapse, ext, T, N, dt)  # buf sized by ext's delays (inner.conn = ext, unused)
    spiked = fill!(allocate(arch, Bool, npre(ext)), false)
    return PoissonSourceState(inner, ext, conn, inner.buf, spiked, _poisson_p(syn.rate, dt, T), syn.seed)
end

# Batched (N,B) drive state (see `BatchedPoissonSourceState` in src/Batch.jl): build the batched inner synapse
# (its (N,B,L) ring receives the deposits), the (n_ext, B) firing mask, and the (n_ext,) source-row index.
# Keying the draw on the source alone makes it shared across the B columns: the same drive realization per
# member, the right default for a parameter sweep at a fixed connectome.
function _make_batched_synstate(arch, syn::PoissonSource, conn, ::Type{T}, N, B, dt, over) where {T}
    ext = _resolve_delays(syn.extconn, dt)
    inner = _make_batched_synstate(arch, syn.synapse, ext, T, N, B, dt, over)   # recurse the sweep into the inner synapse
    next = npre(ext)
    spiked = fill!(allocate(arch, Bool, next, Int(B)), false)
    srcidx = on_architecture(arch, collect(1:next))   # (n_ext,), broadcast across the B columns
    return BatchedPoissonSourceState(inner, ext, conn, inner.buf, spiked, srcidx, _poisson_p(syn.rate, dt, T), syn.seed)
end

# Once-per-step: which virtual sources fire (counter RNG keyed by (seed, step, source)), then scatter their
# events through `extconn` into the wrapped synapse's buffer: the same `scatter!` that delivers network
# spikes (so the CPU/device paths and per-edge delays are shared, not re-implemented).
@inline function _synprestep!(s::PoissonSourceState, integ)
    n = integ.n
    idx = eachindex(s.spiked)
    @. s.spiked = draw_uniform(Float64, s.seed, n, idx) < s.p_spike
    # `sync = false`: this scatter and the next read of `s.buf` run on the same device stream, so ordering
    # already guarantees visibility. Synchronising here instead would drain the pipeline every step (twice
    # over for a two-drive net) and leave the GPU step launch-bound.
    scatter!(s.buf, s.extconn, s.spiked, n; sync = false)
    return nothing
end

# Deliver / accumulate / decay / fused-`_syn_one`: delegate to the wrapped synapse's state. The buffer is
# already populated by `_synprestep!` (the network scatter through the empty outer `conn` is a no-op).
@inline _deliver!(s::PoissonSourceState, integ) = _deliver!(s.inner, integ)
@inline _deliver_jump!(s::PoissonSourceState, integ) = _deliver_jump!(s.inner, integ)
@inline _accumulate!(s::PoissonSourceState, gtot, itot, V) = _accumulate!(s.inner, gtot, itot, V)
@inline _decay!(s::PoissonSourceState) = _decay!(s.inner)
@inline _syn_one(s::PoissonSourceState, i, n, v0, vj, gtot, itot) = _syn_one(s.inner, i, n, v0, vj, gtot, itot)

# * SpikeSourceArray: the deterministic sibling of `PoissonSource`. Instead of drawing Poisson spikes each
# step, it replays a precomputed pattern `spikes` (n_ext × nsteps, source × step) through the identical
# scatter → delay-buffer → deliver pipeline, delegating the postsynaptic response to the wrapped synapse. Only
# `_synprestep!` differs (a table read replaces the RNG draw); everything else is shared with PoissonSource.

"""
    SpikeSourceArray(synapse, extconn, spikes)

An external drive that replays a fixed spike pattern (rather than drawing them): `spikes` is an `n_ext × nsteps`
boolean matrix (virtual source × step), scattered each step through `extconn` (an `n_ext × N` CSR of per-edge
weights + delays) into the wrapped `synapse`'s state. Deterministic (no RNG); the postsynaptic kinetics are
exactly `synapse`'s, so it composes with any synapse family + per-edge delays. Add to a network as
`Projection(SpikeSourceArray(...), Dewdrop._empty_csr(arch, N))`; batched runs share the pattern across columns.
"""
struct SpikeSourceArray{S <: AbstractSynapseModel, C, SP} <: AbstractSynapseModel
    synapse::S       # inner postsynaptic synapse model (its kinetics define the response)
    extconn::C       # n_ext virtual sources → post indices (per-edge weights + delays)
    spikes::SP       # n_ext × nsteps replay pattern (source × step, boolean)
end
export SpikeSourceArray

is_source_synapse(::SpikeSourceArray) = true

# `_synprestep!` reads column `n + 1` every step, so a pattern shorter than the run is a BoundsError
# deep in the loop. Say so at init instead, as `stim_validate` does for TimedArray.
function _check_synapse(s::SpikeSourceArray, nsteps)
    L = size(s.spikes, 2)
    L ≥ nsteps || throw(
        ArgumentError(
            "SpikeSourceArray pattern has $L steps but the run is $nsteps steps " *
                "(need one column per step; it is replayed as `spikes[:, n + 1]`)"
        )
    )
    return nothing
end

struct SpikeSourceArrayState{IS <: AbstractSynapseState, EC, CC, BUF, MV, SP} <: AbstractSynapseState
    inner::IS
    extconn::EC
    conn::CC         # empty outer CSR → the per-state network scatter is a no-op
    buf::BUF         # === inner.buf: events land here via `_synprestep!`, read by `_deliver!(inner, …)`
    spiked::MV       # per-source firing-mask scratch (length n_ext), filled from `spikes` each step
    spikes::SP       # device n_ext × nsteps replay pattern
end
Adapt.@adapt_structure SpikeSourceArrayState

function _make_synstate(arch, syn::SpikeSourceArray, conn, ::Type{T}, N, dt) where {T}
    ext = _resolve_delays(syn.extconn, dt)
    inner = _make_synstate(arch, syn.synapse, ext, T, N, dt)
    spiked = fill!(allocate(arch, Bool, npre(ext)), false)
    return SpikeSourceArrayState(inner, ext, conn, inner.buf, spiked, on_architecture(arch, syn.spikes))
end

# Batched (N,B) replay state (struct `BatchedSpikeSourceArrayState` in src/Batch.jl); the pattern is shared
# across the B columns, mirroring the batched Poisson source's shared draw.
function _make_batched_synstate(arch, syn::SpikeSourceArray, conn, ::Type{T}, N, B, dt, over) where {T}
    ext = _resolve_delays(syn.extconn, dt)
    inner = _make_batched_synstate(arch, syn.synapse, ext, T, N, B, dt, over)
    spiked = fill!(allocate(arch, Bool, npre(ext), Int(B)), false)
    return BatchedSpikeSourceArrayState(inner, ext, conn, inner.buf, spiked, on_architecture(arch, syn.spikes))
end

# Once per step: copy this step's precomputed firing column (`n` 0-based → column n+1) into the scratch mask
# and scatter it through `extconn` into the wrapped synapse's buffer (the same scatter path as real spikes;
# copying into the scratch, rather than passing a device view to `scatter!`, keeps the device path allocation-
# and scalar-index-free; `sync = false` pipelines it on the device stream, like the Poisson source).
@inline function _synprestep!(s::SpikeSourceArrayState, integ)
    n = integ.n
    s.spiked .= view(s.spikes, :, n + 1)
    scatter!(s.buf, s.extconn, s.spiked, n; sync = false)
    return nothing
end
@inline _deliver!(s::SpikeSourceArrayState, integ) = _deliver!(s.inner, integ)
@inline _deliver_jump!(s::SpikeSourceArrayState, integ) = _deliver_jump!(s.inner, integ)
@inline _accumulate!(s::SpikeSourceArrayState, gtot, itot, V) = _accumulate!(s.inner, gtot, itot, V)
@inline _decay!(s::SpikeSourceArrayState) = _decay!(s.inner)
@inline _syn_one(s::SpikeSourceArrayState, i, n, v0, vj, gtot, itot) = _syn_one(s.inner, i, n, v0, vj, gtot, itot)
