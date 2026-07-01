using Dewdrop
using Test
using Statistics
using CairoMakie
using Fathom
include(joinpath(@__DIR__, "brunel_analysis.jl"))

set_theme!(fathom())

# The Brunel (2000) sparse balanced E/I network, the canonical SNN-simulator benchmark,
# reproduced across its four classical dynamical regimes (Brunel Fig. 8): SR (synchronous
# regular), AI (asynchronous irregular), SI-fast and SI-slow (synchronous irregular, fast/slow
# global oscillation). Each is a scaled instance (N=2500-10000) of E+I LIF neurons with signed
# delta-synapse weights (J_E=+J, J_I=-g*J), a fixed 1.5 ms delay, and an external Poisson drive;
# each is validated by its statistical signature -- regularity (CV-ISI) and the population-rate
# power spectrum (peak frequency + prominence) -- and reproduced as the classical raster +
# population-rate + spectrum figure. Parameters were tuned + adversarially verified at reduced
# scale (see .claude/docs/m2-validation-scenarios.md); the (g, eta) coordinates are the canonical
# Brunel points, with drive/J adjusted for the finite-size operating point.

# Run one scaled-Brunel regime through the builder API; return everything the assertions + the
# figure need.
function run_regime(p)
    N = p.NE + p.NI
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    nb = network(m, p.NE, p.NI; arch = Dewdrop.CPU(), tspan = (0.0, p.T))
    project!(nb, :E, DeltaSynapse(); p = 0.1, weight = p.J, delay = steps(p.D), seed = UInt64(1))
    project!(nb, :I, DeltaSynapse(); p = 0.1, weight = -p.g * p.J, delay = steps(p.D), seed = UInt64(0x0101))
    drive!(nb, PoissonDrive(; rate = p.drive_rate, weight = p.drive_weight, seed = UInt64(0x0201)))
    sol = solve(build(nb), FixedStep(0.1); record = (spikes = Spikes(),), v0 = p.v0)
    times, ids = raster(sol)
    _, A = pop_activity(times, p.transient, p.T, 1.0)
    freqs, power = pop_spectrum(A, 1.0)
    fpk, prom = dominant_peak(freqs, power)
    st = (
        rate = mean_rate(times, p.transient, p.T, N), cv = mean_cv_isi(times, ids, N),
        fpeak = fpk, prominence = prom, sync = synchrony_index(A),
        frac_silent = count(==(0), sol.spike_count) / N,
    )
    return (; p, times, ids, A, freqs, power, st, N)
end

# Brunel Fig. 8-style composite: one column per regime, rows = raster / population rate /
# population-rate power spectrum.
function brunel_figure(rs)
    fig = Figure(; size = (1100, 720))
    for (col, r) in enumerate(rs)
        win = (r.p.T - 300.0, r.p.T)
        keep = (r.ids .≤ 150) .& (r.times .≥ win[1]) .& (r.times .≤ win[2])
        cwin, Awin = pop_activity(r.times, win[1], win[2], 1.0)
        Arate = Awin .* (1000.0 / r.N)
        ax1 = Axis(fig[1, col]; title = r.p.title, ylabel = col == 1 ? "Neuron" : "")
        scatter!(ax1, r.times[keep], r.ids[keep]; markersize = 2)
        ax2 = Axis(fig[2, col]; ylabel = col == 1 ? "Rate (Hz)" : "", xlabel = "Time (ms)")
        lines!(ax2, cwin, Arate)
        ax3 = Axis(fig[3, col]; ylabel = col == 1 ? "Power" : "", xlabel = "Freq. (Hz)")
        lines!(ax3, r.freqs, r.power)
        scatter!(ax3, [r.st.fpeak], [maximum(r.power)]; color = Fathom.bermejo, markersize = 7)
        linkxaxes!(ax1, ax2)
    end
    rowsize!(fig.layout, 1, Relative(0.45))
    path = joinpath(@__DIR__, "plots", "m2_brunel_regimes.png")
    mkpath(dirname(path))
    save(path, fig)
    return path
end

@testset "Brunel regimes (Fig. 8: SR / AI / SI-fast / SI-slow)" begin
    # ν_thr (threshold external rate) = θ / (J_E · C_E · τ_m) -- a simulation-independent identity
    # for the full-scale network (θ=20 mV, J=0.1 mV, C_E=1000, τ_m=20 ms → 10 Hz).
    @test 20.0 / (0.1 * 1000 * 20.0) * 1000 ≈ 10.0

    rs = NamedTuple[]

    @testset "SR (synchronous regular, g=3)" begin
        r = run_regime(
            (;
                title = "SR (g = 3)", NE = 2000, NI = 500, g = 3.0, J = 0.224,
                drive_rate = 300.0, drive_weight = 0.224, D = 15, T = 1000.0, transient = 200.0, v0 = nothing,
            )
        )
        push!(rs, r)
        st = r.st
        @test st.cv < 0.1                # regular firing (near clock-like)
        @test st.rate > 300              # high rate (near 1/tref ceiling)
        @test st.prominence > 50         # strong global oscillation
        @test st.sync > 0.2              # well above the finite-size floor
        @test 50 < st.fpeak < 140        # collective lockstep band
        @test st.frac_silent < 0.02
    end

    # AI (asynchronous-irregular). Uses the canonical Brunel drive (external rate = η·θ/(J·τ_m)
    # = 20 events/ms at η=2, weight J=0.1) with randomized initial V. NOTE on fidelity: a fully
    # flat-spectrum, CV≈1 AI state is a large-N phenomenon (the asynchronous region of Brunel's
    # phase diagram shrinks at finite N). At reduced scale the network is either mean-driven
    # (this point: moderate CV, flattest spectrum) or, at lower drive, drops into the
    # synchronous SI-slow regime; a clean asynchronous state in between needs N≫10^4. We
    # therefore validate AI as the *most asynchronous* of the four regimes (flattest spectrum,
    # no synchronous bands) and demonstrate a genuine high-CV AI state with conductance synapses
    # in the Vogels-Abbott COBA test (CV ≈ 1.4).
    @testset "AI (asynchronous-irregular, finite-size-limited, g=5)" begin
        r = run_regime(
            (;
                title = "AI (g = 5)", NE = 2000, NI = 500, g = 5.0, J = 0.1,
                drive_rate = 20.0, drive_weight = 0.1, D = 15, T = 600.0, transient = 200.0, v0 = (10.0, 20.0),
            )
        )
        push!(rs, r)
        st = r.st
        @test 50 < st.rate < 100         # sustained balanced activity at the canonical η=2 drive
        @test st.cv > 0.1                # irregular (finite-size mean-driven; full CV≈1 needs large N)
        @test st.prominence < 40         # flattest spectrum of the four (SR ~680, SI-slow ~255, SI-fast ~37)
        @test st.sync < 0.4              # closest to the finite-size noise floor
        @test st.frac_silent < 0.05      # broadly active
    end

    @testset "SI-fast (synchronous irregular, fast, g=6)" begin
        r = run_regime(
            (;
                title = "SI fast (g = 6)", NE = 2000, NI = 500, g = 6.0, J = 0.4,
                drive_rate = 11.0, drive_weight = 0.4, D = 15, T = 1000.0, transient = 200.0, v0 = nothing,
            )
        )
        push!(rs, r)
        st = r.st
        @test 0.7 < st.cv < 1.05         # irregular single cells
        @test 140 < st.fpeak < 220       # fast delay-controlled oscillation
        @test st.prominence > 8          # clear peak
        @test st.rate < st.fpeak         # sparse: single-cell rate below oscillation freq
        @test st.frac_silent < 0.2
    end

    @testset "SI-slow (synchronous irregular, slow, g=4.5)" begin
        r = run_regime(
            (;
                title = "SI slow (g = 4.5)", NE = 2000, NI = 500, g = 4.5, J = 0.224,
                drive_rate = 3.8, drive_weight = 0.224, D = 15, T = 1000.0, transient = 200.0, v0 = nothing,
            )
        )
        push!(rs, r)
        st = r.st
        @test 3.5 < st.rate < 9.0        # low single-cell rate
        @test 0.4 < st.cv < 0.85         # intermediate-irregular
        @test 8 < st.fpeak < 50          # slow oscillation, well below SI-fast
        @test st.prominence > 40         # clear low-frequency peak
        @test st.frac_silent < 0.1
    end

    path = brunel_figure(rs)
    @test isfile(path) && filesize(path) > 0
end
