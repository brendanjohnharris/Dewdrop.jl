using Dewdrop
using Test

# WRCircuit end-to-end --- the spatial E/I "working-regime" circuit assembled in pure Dewdrop from
# the pieces built across the WRCircuit phases: FNS conductance-adaptation neurons (E adapts, I does
# not), dual-exponential COBA synapses (excitatory Erev=0 / inhibitory Erev<0), distance-dependent
# fixed-count connectivity with exponential kernels + periodic boundaries (broad inhibition), E on a
# grid + I at random positions, all wired through the fluent named-subpopulation builder. This is a
# small-scale run check (it executes, both populations are active, adaptation engages, no NaN) --- a
# statistical match to the BrainPy backend on an identical connectome is the separate validation step.

@testset "WRCircuit-style spatial E/I network (end-to-end)" begin
    Lx = Ly = 16.0
    NE, NI = 256, 64                                  # 16×16 E grid + 64 random I (4:1)
    posE = grid_positions(16, 16)
    posI = random_positions(NI, (Lx, Ly); seed = UInt64(11))

    # one FNS model type, E adapts (ΔgK>0) and I does not (ΔgK=0) → the builder merges them into a
    # single Heterogeneous FNS (block per-neuron ΔgK).
    fnsE = FNSNeuron(; ΔgK = 0.01)
    fnsI = FNSNeuron(; ΔgK = 0.0)

    exc() = DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    inh() = DualExpSynapse(; τr = 1.0, τd = 8.0, Erev = -85.0)

    nb = network(; tspan = (0.0, 300.0))
    population!(nb, :E, fnsE, NE; input = 0.5, positions = posE)     # input > FNS rheobase (≈0.334)
    population!(nb, :I, fnsI, NI; input = 0.5, positions = posI)
    # four distance-dependent fixed-count projections; broad inhibition (larger σ), periodic box
    project!(nb, :E => :E, exc(); kernel = exponential_kernel(2.0), count = 6 * NE, weight = 0.002,
        delay = 2, seed = UInt64(1), period = (Lx, Ly))
    project!(nb, :E => :I, exc(); kernel = exponential_kernel(2.5), count = 6 * NI, weight = 0.002,
        delay = 2, seed = UInt64(2), period = (Lx, Ly))
    project!(nb, :I => :E, inh(); kernel = exponential_kernel(4.5), count = 6 * NE, weight = 0.004,
        delay = 2, seed = UInt64(3), period = (Lx, Ly))
    project!(nb, :I => :I, inh(); kernel = exponential_kernel(4.5), count = 6 * NI, weight = 0.004,
        delay = 2, seed = UInt64(4), period = (Lx, Ly))
    prob = build(nb)

    @test prob.model isa Heterogeneous                # same FNS type, per-neuron ΔgK
    @test prob.n == NE + NI
    @test length(prob.projections) == 4
    @test prob.subpops.E == 1:NE && prob.subpops.I == (NE + 1):(NE + NI)
    @test Dewdrop.nedges(prob.projections[1].conn) == 6 * NE   # exact fixed count

    # randomised initial V breaks synchrony; run and inspect via the named-subpop reference API
    sol = solve(prob, FixedStep(0.1); v0 = (-70.0, -50.0), record = (spikes = Spikes(),))
    @test all(isfinite, sol.state.state.V)            # no NaN from the conductance dynamics
    @test sum(sol.spike_count[1:NE]) > 0              # excitatory population active
    @test sum(firing_rate(sol, :I)) > 0              # inhibitory population active (via the addressor)
    @test maximum(sol.state.state.w[1:NE]) > 0        # adaptation conductance gK accumulated in E
    t_E, _ = raster(sol; of = :E)                    # per-subpop raster works on the spatial net
    @test !isempty(t_E)

    # spatial metadata flows through to the solution (for radial-AC / spatial measures)
    @test length(sol.positions) == NE + NI
    @test sol[:E].positions == posE                  # grid coordinates of the E subpopulation
    @test sol[:I].positions == sol.positions[(NE + 1):end]
end

# --- BrainPy-faithful reproduction machinery (FrozenDualExpSynapse, PrescribedCOBA, wrcircuit) ---
# These pieces reproduce the WRCircuit's BrainPy integration scheme bit-for-bit (the explicit/frozen
# COBA current g·(Erev−V) at the pre-step V, not folded into the membrane leak). The end-to-end
# cross-simulator validation lives in test/simulator_comparisons/wrcircuit (needs the BrainPy env);
# here we pin the new building blocks against independent in-test references.

# the exact COBA propagator (mirrors Dewdrop's internal `_coba_step`): the reference both tests check.
_coba(V, EL, R, τ, gtot, itot, dt) = (d = 1 + R * gtot; V∞ = (EL + R * itot) / d; V∞ + (V - V∞) * exp(-dt * d / τ))

@testset "PrescribedCOBA frozen current ≡ analytic reference" begin
    dt, nsteps, N = 0.1, 600, 1
    EL, R, τ, Erev = -65.0, 1.0, 10.0, 0.0
    m = LIF(; τ = τ, EL = EL, Vθ = 1.0e9, Vr = -70.0, R = R, tref = 0.0)   # Vθ huge → purely subthreshold
    g = reshape([0.06 * exp(-((n - 150) / 60)^2) for n in 1:nsteps], 1, nsteps)   # a smooth conductance bump
    prob = DewdropNetwork(m, N; input = 0.0, tspan = (0.0, nsteps * dt),
        projections = (Projection(PrescribedCOBA(g, Erev), Dewdrop._empty_csr(Dewdrop.CPU(), N)),))
    v0 = -60.0
    sol = solve(prob, FixedStep(dt); v0 = [v0], record = (v = Trace(:V),))
    Vdd = vec(sol.record.v.data)                                   # V_1 … V_nsteps
    Vref = Float64[];
    V = v0
    for n in 1:nsteps                                              # frozen current: Isyn = g·(Erev − V_pre), gtot = 0
        V = _coba(V, EL, R, τ, 0.0, g[1, n] * (Erev - V), dt)
        push!(Vref, V)
    end
    @test Vdd ≈ Vref rtol = 1.0e-12
    @test maximum(abs.(Vdd .- Vref)) < 1.0e-9
end

@testset "FrozenDualExpSynapse ≡ replayed reference (2-neuron)" begin
    dt, nsteps = 0.1, 800
    τr, τd, Erev, w, d = 1.0, 5.0, 0.0, 2.0, 5
    pre = LIF(; τ = 10.0, EL = -65.0, Vθ = -50.0, Vr = -70.0, R = 1.0, tref = 2.0)
    post = LIF(; τ = 10.0, EL = -65.0, Vθ = 1.0e9, Vr = -70.0, R = 1.0, tref = 0.0)   # post never spikes
    model = Dewdrop._combine_models(Any[pre, post], Int[1, 1])     # two-group Heterogeneous LIF
    conn = Dewdrop.SparseCSR(Dewdrop.CPU(), [(1, 2, w, d)]; npre = 2, npost = 2)
    proj = Projection(FrozenDualExpSynapse(; τr = τr, τd = τd, Erev = Erev), conn)
    prob = DewdropNetwork(model, 2; input = [25.0, 0.0], tspan = (0.0, nsteps * dt), projections = (proj,))
    sol = solve(prob, FixedStep(dt); v0 = [-65.0, -60.0], record = (v = Trace(:V), spikes = Spikes()))
    Vpost = sol.record.v.data[2, :]                               # post membrane, V_1 … V_nsteps
    tpre, ids = raster(sol)
    prespk = sort([round(Int, tpre[k] / dt) - 1 for k in eachindex(tpre) if ids[k] == 1])   # 0-based pre steps
    @test !isempty(prespk)                                        # the driven pre neuron actually fires
    # replay the dual-exp COBA conductance (same deliver→read→decay recurrence + frozen current)
    a = Dewdrop._dualexp_a(τr, τd);
    dr, dd = exp(-dt / τr), exp(-dt / τd)
    deposit = zeros(nsteps + d + 2)
    for s in prespk
        s + d + 1 ≤ length(deposit) && (deposit[s + d + 1] += w)  # ring: spike at s deposits at s+d
    end
    gr = 0.0;
    gd = 0.0;
    V = -60.0;
    Vref = Float64[]
    for n in 0:(nsteps - 1)
        g = a * (gd - gr)                                         # conductance used at step n (pre-deposit)
        V = _coba(V, -65.0, 1.0, 10.0, 0.0, g * (Erev - V), dt)  # frozen current
        push!(Vref, V)
        gr += deposit[n + 1];
        gd += deposit[n + 1]
        gr *= dr;
        gd *= dd
    end
    @test Vpost ≈ Vref rtol = 1.0e-10
    # and the frozen scheme genuinely differs from the implicit DualExpSynapse (conductance in the leak)
    proj2 = Projection(DualExpSynapse(; τr = τr, τd = τd, Erev = Erev), conn)
    prob2 = DewdropNetwork(model, 2; input = [25.0, 0.0], tspan = (0.0, nsteps * dt), projections = (proj2,))
    sol2 = solve(prob2, FixedStep(dt); v0 = [-65.0, -60.0], record = (v = Trace(:V),))
    @test maximum(abs.(sol2.record.v.data[2, :] .- Vref)) > 1.0e-2   # implicit ≠ frozen for finite conductance
end

@testset "wrcircuit builder: Heterogeneous FNS E/I + replayed external drive" begin
    NE, NI = 64, 16
    N = NE + NI
    E = FNSNeuron(; gL = 0.0167, ΔgK = 0.005)
    I = FNSNeuron(; gL = 0.025, ΔgK = 0.0)                        # I does not adapt → Heterogeneous merge
    posE = grid_positions(8, 8);
    posI = random_positions(NI, (8.0, 8.0); seed = UInt64(7))
    pos = vcat(posE, posI)
    exc(p) = FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    inh(p) = FrozenDualExpSynapse(; τr = 2.0, τd = 4.5, Erev = -80.0)
    cee = distance_fixed_count(Dewdrop.CPU(), pos; kernel = exponential_kernel(2.0), count = 6NE,
        weight = 0.002, delay = 14, seed = UInt64(1), period = (8.0, 8.0), sources = 1:NE, targets = 1:NE)
    cie = distance_fixed_count(Dewdrop.CPU(), pos; kernel = exponential_kernel(4.0), count = 6NE,
        weight = 0.004, delay = 14, seed = UInt64(3), period = (8.0, 8.0), sources = (NE + 1):N, targets = 1:NE)
    projs = [Projection(exc(0), cee), Projection(inh(0), cie)]
    nsteps = 2000
    gext = fill(0.05, N, nsteps)                                  # a constant external conductance (Erev=0)
    prob = wrcircuit(; NE = NE, NI = NI, E = E, I = I, projections = projs, gext = gext,
        positions = pos, tspan = (0.0, nsteps * 0.1))
    @test prob.model isa Dewdrop.Heterogeneous                   # E adapts, I does not
    @test prob.subpops.E == 1:NE && prob.subpops.I == (NE + 1):N
    @test length(prob.projections) == 3                          # 2 recurrent + the prescribed external

    v0 = collect(range(-70.0, -50.0; length = N))
    sol = solve(prob, FixedStep(0.1); v0 = v0, record = (spikes = Spikes(),))
    sol2 = solve(prob, FixedStep(0.1); v0 = v0, record = (spikes = Spikes(),))
    @test all(isfinite, sol.state.state.V)                       # conductance dynamics stay finite
    @test sol.spike_count == sol2.spike_count                    # deterministic
    @test sum(sol.spike_count[1:NE]) > 0                         # excitatory population active
    @test maximum(sol.state.state.w[1:NE]) > 0                   # E adaptation conductance accumulates

    # the external drive is what keeps it alive: a stronger prescribed conductance → more spikes
    probhi = wrcircuit(; NE = NE, NI = NI, E = E, I = I, projections = projs, gext = 2 .* gext,
        positions = pos, tspan = (0.0, nsteps * 0.1))
    solhi = solve(probhi, FixedStep(0.1); v0 = v0, record = (spikes = Spikes(),))
    @test sum(solhi.spike_count) > sum(sol.spike_count)
end
