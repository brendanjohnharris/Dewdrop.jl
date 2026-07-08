# * Generic streaming Poisson drive: `PoissonSource{S}` wraps ANY synapse model `S` and turns it into an
# external drive of virtual Poisson sources (rate `rate` Hz), wired to post-neurons by `extconn`. Each step
# it generates the sources' Poisson spikes and scatters them (through the SAME `scatter!` + delay-buffer +
# `_deliver!` pipeline the network uses for real spikes) into the wrapped synapse's state. So the
# postsynaptic kinetics are exactly `S`'s (delta / CUBA / COBA / dual-exp / …): the source is the input
# statistics + wiring, the synapse is the response. Streaming (O(N) state, no precomputed conductance
# matrix), and (because it routes through `scatter!`) CPU and device alike.
#
# The state DELEGATES `_deliver!`/`_accumulate!`/`_decay!`/`_syn_one` to the wrapped synapse's state
# (`inner`); only `_synprestep!` is new (generate + scatter the Poisson events). The outer `conn` is an empty
# CSR, so the per-state network scatter (`scatter!(syn.buf, syn.conn, …)`) is a no-op; the real wiring
# lives in `extconn`, used only by the once-per-step generator. The generic replacement for the bespoke
# `PoissonDualExpDrive`: that one hard-codes the dual-exp deliver/accumulate/decay bodies; this delegates.

"""
    PoissonSource(synapse, extconn; rate, seed)

An external drive of virtual Poisson sources wired to the network by `extconn` (an `n_ext × N` CSR carrying
per-edge weights and delays), delivering through any postsynaptic `synapse` model. Each step generates the
sources' Poisson spikes (rate `rate` Hz; per-source per-step probability `rate·dt/1000`) and scatters them
into `synapse`'s conductance state via the standard delay-buffer pipeline: no precomputed conductance
matrix. The postsynaptic kinetics are exactly `synapse`'s, so the same drive composes with `DeltaSynapse`,
`CurrentSynapse`, `ConductanceSynapse`, `DualExpSynapse`, …. Add to a network as
`Projection(PoissonSource(...), empty_csr)`.
"""
struct PoissonSource{S <: AbstractSynapseModel, C, T} <: AbstractSynapseModel
    synapse::S       # inner postsynaptic synapse model (its kinetics define the response)
    extconn::C       # n_ext virtual sources → post indices (per-edge weights + delays)
    rate::T          # Poisson rate (Hz) per source
    seed::UInt64
end
PoissonSource(synapse::AbstractSynapseModel, extconn; rate, seed = 0x9e3779b97f4a7c15) =
    PoissonSource(synapse, extconn, Float64(rate), seed % UInt64)
export PoissonSource

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
    return PoissonSourceState(inner, ext, conn, inner.buf, spiked, T(syn.rate * dt / 1000), syn.seed)
end

# Batched (N,B) drive state (see `BatchedPoissonSourceState` in src/Batch.jl): build the batched inner synapse
# (its (N,B,L) ring receives the deposits), the (n_ext, B) firing mask, and the constant (n_ext, B) source-row
# index that makes the per-step counter-RNG draw SHARED across the B columns: same drive realization per
# member, the right default for a parameter sweep at a fixed connectome.
function _make_batched_synstate(arch, syn::PoissonSource, conn, ::Type{T}, N, B, dt, over) where {T}
    ext = _resolve_delays(syn.extconn, dt)
    inner = _make_batched_synstate(arch, syn.synapse, ext, T, N, B, dt, over)   # recurse the sweep into the inner synapse
    next = npre(ext)
    spiked = fill!(allocate(arch, Bool, next, Int(B)), false)
    srcidx = on_architecture(arch, repeat(1:next, 1, Int(B)))
    return BatchedPoissonSourceState(inner, ext, conn, inner.buf, spiked, srcidx, T(syn.rate * dt / 1000), syn.seed)
end

# Once-per-step: which virtual sources fire (counter RNG keyed by (seed, step, source)), then scatter their
# events through `extconn` into the wrapped synapse's buffer: the SAME `scatter!` that delivers network
# spikes (so the CPU/device paths and per-edge delays are shared, not re-implemented).
@inline function _synprestep!(s::PoissonSourceState, integ)
    n = integ.n
    idx = eachindex(s.spiked)
    @. s.spiked = draw_uniform(Float64, s.seed, n, idx) < s.p_spike
    # `sync = false`: this scatter and the next read of `s.buf` (the fused step's inline deliver) run on the SAME
    # device stream, so ordering already guarantees visibility: no host sync needed. The `scatter!` default
    # `sync = true` would `synchronize()` the device EVERY step (and twice over for a 2-drive net),
    # draining the pipeline the fused path builds (Fused.jl:197 passes `sync = false` for the same reason) and
    # leaving the GPU step launch-bound / slower than CPU. Behaviour-identical; the buffer contents are the same.
    scatter!(s.buf, s.extconn, s.spiked, n; sync = false)
    return nothing
end

# Deliver / accumulate / decay / fused-`_syn_one`: delegate to the wrapped synapse's state. The buffer is
# already populated by `_synprestep!` (the network scatter through the empty outer `conn` is a no-op).
@inline _deliver!(s::PoissonSourceState, integ) = _deliver!(s.inner, integ)
@inline _accumulate!(s::PoissonSourceState, gtot, itot, V) = _accumulate!(s.inner, gtot, itot, V)
@inline _decay!(s::PoissonSourceState) = _decay!(s.inner)
@inline _syn_one(s::PoissonSourceState, i, n, v, gtot, itot) = _syn_one(s.inner, i, n, v, gtot, itot)

# * SpikeSourceArray: the DETERMINISTIC sibling of `PoissonSource`. Instead of drawing Poisson spikes each
# step, it REPLAYS a precomputed pattern `spikes` (n_ext × nsteps, source × step) through the identical
# scatter → delay-buffer → deliver pipeline, delegating the postsynaptic response to the wrapped synapse. Only
# `_synprestep!` differs (a table read replaces the RNG draw); everything else is shared with PoissonSource.

"""
    SpikeSourceArray(synapse, extconn, spikes)

An external drive that REPLAYS a fixed spike pattern (rather than drawing them): `spikes` is an `n_ext × nsteps`
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
# and scatter it through `extconn` into the wrapped synapse's buffer (the SAME scatter path as real spikes;
# copying into the scratch, rather than passing a device view to `scatter!`, keeps the device path allocation-
# and scalar-index-free; `sync = false` pipelines it on the device stream, like the Poisson source).
@inline function _synprestep!(s::SpikeSourceArrayState, integ)
    n = integ.n
    s.spiked .= view(s.spikes, :, n + 1)
    scatter!(s.buf, s.extconn, s.spiked, n; sync = false)
    return nothing
end
@inline _deliver!(s::SpikeSourceArrayState, integ) = _deliver!(s.inner, integ)
@inline _accumulate!(s::SpikeSourceArrayState, gtot, itot, V) = _accumulate!(s.inner, gtot, itot, V)
@inline _decay!(s::SpikeSourceArrayState) = _decay!(s.inner)
@inline _syn_one(s::SpikeSourceArrayState, i, n, v, gtot, itot) = _syn_one(s.inner, i, n, v, gtot, itot)
