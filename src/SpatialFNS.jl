# * Native spatial FNS "working-regime" network --- the BrainPy `WRCircuit.jl` `Spatial` model
# (conductance-adaptation FNS E/I on a 2D periodic box, distance-dependent COBA connectivity, plus an
# external Poisson drive) re-expressed as a single high-level constructor over Dewdrop primitives. No
# Python/BrainPy: the connectome is sampled with the native Gumbel-top-k `distance_fixed_count`, weights
# with the generic in-degree-scaled `correlate_weights!` (a reusable Dewdrop primitive), the recurrent paths use the
# BrainPy-faithful `FrozenDualExpSynapse`, and the external drive is replayed as a `PrescribedCOBA`
# conductance generated from a native Poisson raster.
#
# The structure (connectome / weights / initial V / external spikes) is sampled from Dewdrop's
# counter-based RNG rather than JAX, so a `spatial_fns` network is a *statistical* (not bit-for-bit) match
# to a same-parameter BrainPy `Spatial` run --- the two RNGs cannot produce identical realisations. The
# integration scheme, kernels, and parameter→network map are matched exactly (see test/spatialfns.jl and
# the bit-for-bit ingest reproduction in test/simulator_comparisons/wrcircuit).

# Derive an independent sub-seed per random structure (golden-ratio mix; the native analogue of a
# `jax.random.split` chain off the master seed).
@inline _subseed(seed::Unsigned, tag::Integer) = (seed % UInt64) ⊻ ((tag % UInt64) * 0x9e3779b97f4a7c15)

# Cell-centred grid over [0, dx]^2 with ne×ne points (BrainPy `GridPositions`: spacing dx/ne, points at
# dx/ne·(i-0.5)) --- the first-class `grid_positions(…; centered=true)`. E tiles [0,dx]^2; I is uniform on it.
_grid_centered(ne::Integer, dx::Real) = grid_positions(ne, ne; spacing = dx / ne, centered = true)

# In-degree-scaled recurrent weights are the generic `correlate_weights!` (see Connectivity.jl) --- the
# WRCircuit `utils.correlate_weights` port now lives there as a reusable building block, called below with
# the per-path `targets`/`seed`.

# Physical synaptic latency `delay_ms` → Dewdrop stored delay (steps). BrainPy's delay `D=round(delay/dt)`
# onsets the postsynaptic conductance at spike+D; Dewdrop's ring buffer + dual-exp onsets at spike+stored+1,
# so a stored delay of `D-1` reproduces the same physical latency (the −1 calibrated bit-for-bit in the
# ingest harness). Clamped to ≥ 1 (the fixed-step engine cannot deliver within the same step).
@inline _delay_steps(delay_ms::Real, dt::Real) = max(1, round(Int, delay_ms / dt) - 1)

# Out-edges of each presynaptic source as `(post_local, weight)` lists (host-side gather over a CSR).
function _out_edges(conn::SparseCSR)
    out = [Tuple{Int, Float64}[] for _ in 1:(conn.npre)]
    @inbounds for pre in 1:(conn.npre)
        for e in conn.rowptr[pre]:(conn.rowptr[pre + 1] - 1)
            push!(out[pre], (Int(conn.post[e]), Float64(conn.weight[e])))
        end
    end
    return out
end

# Native external Poisson drive → a replayed `(N, nsteps)` dual-exp COBA conductance matrix. `N_ext`
# Poisson sources at `nu` Hz connect to E and I via `FixedProb(p_ext)` connectomes (weights set by
# `correlate_weights`); each source spike deposits, after the conduction delay, into the target's dual-exp
# conductance via the SAME deliver→read→decay recurrence as the recurrent synapses (Erev=0). The conductance
# read at step `n` is `a·(g_decay − g_rise)` BEFORE that step's deposit (the deposit cancels in the
# difference), so it lines up with how `PrescribedCOBA` reads the column. Sampling all of (sources × steps)
# of the connectome/raster is host-side and reproducible from the counter RNG.
function _external_gext(N, NE, NI, N_ext, p_ext, J_ee, J_ei, nu, dt, nsteps, τr, τd, de, seed)
    conn_e = fixed_prob(CPU(), N_ext, NE, p_ext; weight = 1.0, delay = steps(de),
        seed = _subseed(seed, 10), sources = 1:N_ext, targets = 1:NE)
    correlate_weights!(conn_e, J_ee; targets = 1:NE, seed = _subseed(seed, 12))
    conn_i = fixed_prob(CPU(), N_ext, NI, p_ext; weight = 1.0, delay = steps(de),
        seed = _subseed(seed, 11), sources = 1:N_ext, targets = 1:NI)
    correlate_weights!(conn_i, J_ei; targets = 1:NI, seed = _subseed(seed, 13))
    outE = _out_edges(conn_e)
    outI = _out_edges(conn_i)

    a = _dualexp_a(τr, τd)
    dr = exp(-dt / τr)
    dd = exp(-dt / τd)
    p_spike = nu * dt / 1000                              # BrainPy PoissonGroup: P(spike)=freqs·dt/1000
    sraster = _subseed(seed, 14)

    # Schedule deposits by arrival iteration: a source-j spike at step s arrives (is deposited) at
    # iteration s+de, so onset is at step s+de+1 = s + round(delay/dt) (matches BrainPy).
    deliveries = [Tuple{Int, Int, Float64}[] for _ in 1:nsteps]   # (pop ∈ {0=E,1=I}, post_local, weight)
    for s in 0:(nsteps - 1)
        for j in 1:N_ext
            draw_uniform(Float64, sraster, s, j) < p_spike || continue
            arr = s + de
            arr ≤ nsteps - 1 || continue
            for (post, w) in outE[j]
                push!(deliveries[arr + 1], (0, post, w))
            end
            for (post, w) in outI[j]
                push!(deliveries[arr + 1], (1, post, w))
            end
        end
    end

    grE = zeros(NE); gdE = zeros(NE); grI = zeros(NI); gdI = zeros(NI)
    G = zeros(N, nsteps)
    for n in 0:(nsteps - 1)
        @inbounds for j in 1:NE
            G[j, n + 1] = a * (gdE[j] - grE[j])
        end
        @inbounds for j in 1:NI
            G[NE + j, n + 1] = a * (gdI[j] - grI[j])
        end
        @inbounds for (pop, post, w) in deliveries[n + 1]
            if pop == 0
                grE[post] += w; gdE[post] += w
            else
                grI[post] += w; gdI[post] += w
            end
        end
        grE .*= dr; gdE .*= dd; grI .*= dr; gdI .*= dd
    end
    return G
end

# --- Streaming external Poisson drive: now the generic `PoissonSource{FrozenDualExpSynapse}` (see
# PoissonSource.jl). This was a bespoke `PoissonDualExpDrive` synapse with hand-copied dual-exp
# deliver/accumulate/decay bodies; it is now `PoissonSource` wrapping the frozen dual-exp synapse ---
# bit-identical (same (seed, step, source) RNG keying, same scatter order, same frozen-current kinetics),
# but generalised to drive ANY synapse and to route Poisson events through the shared `scatter!`.

# Build the merged external connectome (ext2E + ext2I, weights via `correlate_weights`) as one
# `n_ext × N` CSR --- shared by the streaming drive and (via `_external_gext`) the prescribed drive,
# using the SAME sub-seeds so the two external-drive paths realise the identical connectome.
function _build_extconn(arch, N, NE, NI, N_ext, p_ext, J_ee, J_ei, de, seed)
    ce = fixed_prob(CPU(), N_ext, NE, p_ext; weight = 1.0, delay = steps(de),
        seed = _subseed(seed, 10), sources = 1:N_ext, targets = 1:NE)
    correlate_weights!(ce, J_ee; targets = 1:NE, seed = _subseed(seed, 12))
    ci = fixed_prob(CPU(), N_ext, NI, p_ext; weight = 1.0, delay = steps(de),
        seed = _subseed(seed, 11), sources = 1:N_ext, targets = 1:NI)
    correlate_weights!(ci, J_ei; targets = 1:NI, seed = _subseed(seed, 13))
    edges = Tuple{Int, Int, Float64, Int}[]
    sizehint!(edges, length(ce.post) + length(ci.post))
    @inbounds for e in eachindex(ce.post)
        push!(edges, (Int(ce.src[e]), Int(ce.post[e]), Float64(ce.weight[e]), Int(ce.delay[e])))
    end
    @inbounds for e in eachindex(ci.post)               # I targets live at global indices NE+1 … N
        push!(edges, (Int(ci.src[e]), Int(ci.post[e]) + NE, Float64(ci.weight[e]), Int(ci.delay[e])))
    end
    return SparseCSR(arch, edges; npre = N_ext, npost = N)
end

# Move a host-built CSR onto `arch` (no-op for CPU; the construction loops are inherently host-side).
_csr_on_arch(::CPU, c::SparseCSR) = c
_csr_on_arch(arch::AbstractArchitecture, c::SparseCSR) = SparseCSR(
    on_architecture(arch, c.rowptr), on_architecture(arch, c.post), on_architecture(arch, c.weight),
    on_architecture(arch, c.delay), on_architecture(arch, c.src), c.maxdeg, c.npre, c.npost)

"""
    spatial_fns(; rho=20000, dx=0.5, gamma=4, sigma_ee=0.06, sigma_ei=0.07, sigma_ie=0.14, sigma_ii=0.14,
                K_ee=260, K_ei=340, K_ie=225, K_ii=290, delta=4.0, J_ee=0.00105, J_ei=0.00145,
                nu=10.0, n_ext=100, tau_r_e=1.0, tau_r_i=2.0, tau_d_e=5.0, tau_d_i=4.5,
                V_rev_e=0.0, V_rev_i=-80.0, e_delay=1.5, i_delay=1.5, Delta_g_K=0.002, tau_K=40.0,
                tspan, dt=0.1, seed=0x5fd, synapse=:frozen, arch=CPU())

Assemble the spatial FNS E/I "working-regime" network natively (the BrainPy `WRCircuit.jl` `Spatial`
model), returning a [`DewdropNetwork`](@ref) with `:E`/`:I` subpopulations registered. Geometry:
`ne = round(√(rho)·dx)`, `NE = ne²` excitatory neurons on a cell-centred grid over `[0,dx]²`, `NI =
round(NE/gamma)` inhibitory neurons at uniform-random positions; the box is periodic. Each of the four
recurrent paths is a distance-dependent fixed-count connectome (`distance_fixed_count`, exponential
kernel of length scale `sigma_xx`) with mean in-degree `K_xx` (`num_connections = K_xx·N_post`), weights
set by the in-degree-scaled `correlate_weights` (`J_ie = J_ee·delta`, `J_ii = J_ei·delta`; inhibition is
carried by `V_rev_i`, not by sign). An external population of `N_ext = round(√(n_ext·NE))` Poisson sources
at `nu` Hz drives E and I (`FixedProb(p_ext=√(n_ext/NE))`), replayed as a `PrescribedCOBA` conductance.

`synapse` selects the COBA integration scheme: `:frozen` ([`FrozenDualExpSynapse`](@ref), the BrainPy
`sum_current_inputs` scheme --- use to reproduce WRCircuit) or `:exact` ([`DualExpSynapse`](@ref), the
exact propagator). `external` selects the drive: `:streaming` ([`PoissonSource`](@ref) over a frozen dual-exp synapse, generated
on the fly --- O(N) memory, the default, scales to long/large runs, CPU-only) or `:prescribed` (a dense
`(N, nsteps)` conductance replayed via [`PrescribedCOBA`](@ref) --- backend-agnostic, for bit-for-bit
replay/validation at small scale). Both realise the identical external connectome + Poisson raster for a
given seed (so `:streaming` ≡ `:prescribed` numerically). `dt` sets every connectome's integer delays
(and the drive's per-step Poisson probability), so it MUST equal the step passed to
`solve(prob, FixedStep(dt))`. Canonical initial condition:
`solve(prob, FixedStep(dt); v0 = (-70.0, -50.0))` (uniform `V₀`).
"""
function spatial_fns(; rho = 20000, dx = 0.5, gamma = 4,
        sigma_ee = 0.06, sigma_ei = 0.07, sigma_ie = 0.14, sigma_ii = 0.14,
        K_ee = 260, K_ei = 340, K_ie = 225, K_ii = 290,
        delta = 4.0, J_ee = 0.00105, J_ei = 0.00145,
        nu = 10.0, n_ext = 100,
        tau_r_e = 1.0, tau_r_i = 2.0, tau_d_e = 5.0, tau_d_i = 4.5,
        V_rev_e = 0.0, V_rev_i = -80.0, e_delay = 1.5, i_delay = 1.5,
        Delta_g_K = 0.002, tau_K = 40.0,
        tspan, dt = 0.1, seed = 0x5fd, synapse::Symbol = :frozen, external::Symbol = :streaming,
        arch::AbstractArchitecture = CPU())
    seed = seed % UInt64

    # --- geometry ---
    ne = round(Int, sqrt(rho) * dx)
    ne ≥ 1 || throw(ArgumentError("rho·dx² too small: ne = round(√rho·dx) = $ne < 1"))
    NE = ne^2
    NI = round(Int, NE / gamma)
    NI ≥ 1 || throw(ArgumentError("NE/gamma too small: NI = $NI < 1"))
    N = NE + NI
    Erange = 1:NE
    Irange = (NE + 1):N

    posE = _grid_centered(ne, dx)
    posI = random_positions(NI, (dx, dx); seed = _subseed(seed, 1))
    positions = vcat(posE, posI)

    # --- neurons (E adapts, I does not; merged into a Heterogeneous FNS by `wrcircuit`) ---
    E = FNSNeuron(; C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -70.0,
        tref = 4.0, τK = tau_K, ΔgK = Delta_g_K)
    I = FNSNeuron(; C = 0.25, gL = 0.025, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -70.0,
        tref = 4.0, τK = tau_K, ΔgK = 0.0)

    # --- recurrent connectomes (Gumbel top-k fixed count; periodic box; E2E/I2I include the diagonal) ---
    de = _delay_steps(e_delay, dt)
    di = _delay_steps(i_delay, dt)
    period = (Float64(dx), Float64(dx))
    cee = distance_fixed_count(CPU(), positions; kernel = exponential_kernel(sigma_ee),
        count = K_ee * NE, weight = 1.0, delay = steps(de), seed = _subseed(seed, 2),
        allow_self = true, period = period, sources = Erange, targets = Erange)
    cei = distance_fixed_count(CPU(), positions; kernel = exponential_kernel(sigma_ei),
        count = K_ei * NI, weight = 1.0, delay = steps(de), seed = _subseed(seed, 3),
        allow_self = false, period = period, sources = Erange, targets = Irange)
    cie = distance_fixed_count(CPU(), positions; kernel = exponential_kernel(sigma_ie),
        count = K_ie * NE, weight = 1.0, delay = steps(di), seed = _subseed(seed, 4),
        allow_self = false, period = period, sources = Irange, targets = Erange)
    cii = distance_fixed_count(CPU(), positions; kernel = exponential_kernel(sigma_ii),
        count = K_ii * NI, weight = 1.0, delay = steps(di), seed = _subseed(seed, 5),
        allow_self = true, period = period, sources = Irange, targets = Irange)

    # --- weights (in-degree-scaled; I weights are δ-amplified, sign carried by reversal potential) ---
    J_ie = J_ee * delta
    J_ii = J_ei * delta
    correlate_weights!(cee, J_ee; targets = Erange, seed = _subseed(seed, 6))
    correlate_weights!(cei, J_ei; targets = Irange, seed = _subseed(seed, 7))
    correlate_weights!(cie, J_ie; targets = Erange, seed = _subseed(seed, 8))
    correlate_weights!(cii, J_ii; targets = Irange, seed = _subseed(seed, 9))

    synmodel(τr, τd, Erev) = synapse === :frozen ? FrozenDualExpSynapse(; τr = τr, τd = τd, Erev = Erev) :
                             synapse === :exact ? DualExpSynapse(; τr = τr, τd = τd, Erev = Erev) :
                             throw(ArgumentError("synapse must be :frozen or :exact (got :$synapse)"))
    projections = [
        Projection(synmodel(tau_r_e, tau_d_e, V_rev_e), _csr_on_arch(arch, cee)),   # E→E
        Projection(synmodel(tau_r_e, tau_d_e, V_rev_e), _csr_on_arch(arch, cei)),   # E→I
        Projection(synmodel(tau_r_i, tau_d_i, V_rev_i), _csr_on_arch(arch, cie)),   # I→E
        Projection(synmodel(tau_r_i, tau_d_i, V_rev_i), _csr_on_arch(arch, cii)),   # I→I
    ]

    # --- external Poisson drive: streaming (scalable, default) or the dense prescribed gext ---
    p_ext = sqrt(n_ext / NE)
    N_ext = round(Int, sqrt(n_ext * NE))
    if external === :streaming
        extconn = _build_extconn(arch, N, NE, NI, N_ext, p_ext, J_ee, J_ei, de, seed)
        drive = Projection(
            PoissonSource(FrozenDualExpSynapse(; τr = tau_r_e, τd = tau_d_e, Erev = V_rev_e),
                extconn; rate = nu, seed = _subseed(seed, 14)),
            _empty_csr(arch, N))
        return wrcircuit(; NE = NE, NI = NI, E = E, I = I, projections = vcat(projections, [drive]),
            gext = nothing, positions = positions, tspan = tspan, arch = arch)
    elseif external === :prescribed
        nsteps = round(Int, (tspan[2] - tspan[1]) / dt)
        nsteps ≥ 1 || throw(ArgumentError("tspan/dt gives nsteps = $nsteps < 1"))
        gext = _external_gext(N, NE, NI, N_ext, p_ext, J_ee, J_ei, nu, dt, nsteps,
            tau_r_e, tau_d_e, de, seed)
        return wrcircuit(; NE = NE, NI = NI, E = E, I = I, projections = projections,
            gext = gext, positions = positions, tspan = tspan, arch = arch)
    else
        throw(ArgumentError("external must be :streaming or :prescribed (got :$external)"))
    end
end
export spatial_fns
