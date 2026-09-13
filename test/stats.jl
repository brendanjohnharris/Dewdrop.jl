using Dewdrop
using Test

# Statistical observables. The spectral measures transform through FFTW, so what is anchored here is
# Dewdrop's own part: the frequency axis and the way each measure is assembled, the latter against a
# direct O(N²) DFT. The measures themselves are checked on hand-constructed rasters with known values,
# plus the sol-level wrappers on a small recorded network. (Exact agreement with the reference
# stats.py/numpy is the separate cross-validation in test/simulator_comparisons/stats_validation/.)

# the O(N²) definition, as an independent reference for the assembled spectrum
_dft_ref(x) = [sum(x[j + 1] * cis(-2π * k * j / length(x)) for j in 0:(length(x) - 1)) for k in 0:(length(x) - 1)]

@testset "frequency axis and the assembled spectrum" begin
    # `_fftfreq` follows numpy: the second argument is a sample spacing, not a sampling rate
    @test Dewdrop._fftfreq(8, 1.0) ≈ [0, 1, 2, 3, -4, -3, -2, -1] ./ 8
    @test Dewdrop._fftfreq(5, 0.5) ≈ [0, 1, 2, -2, -1] ./ (5 * 0.5)
    # the spectrum against a direct DFT of the same rows, at a power-of-two and an awkward length
    for T in (16, 30)
        S = [sinpi(0.3k + 0.1i) + 0.2cospi(0.07k) for i in 1:3, k in 0:(T - 1)]
        psd, freqs = power_spectrum(S; n_segments = 1, dt = 0.5)
        ref = zeros(Float64, T)
        for i in 1:3
            ref .+= abs2.(_dft_ref(Float64.(S[i, :]))) ./ T
        end
        @test psd ≈ ref ./ 3
        @test freqs ≈ Dewdrop._fftfreq(T, 0.5)
    end
end

@testset "coarsegrain (time binning)" begin
    S = Bool[
        1 0 1 1 0 1
        0 0 1 0 1 1
    ]
    @test Dewdrop.coarsegrain(S, 2; dims = 2) == [1 2 1; 0 1 2]          # sum each 2-step bin per neuron
    @test Dewdrop.coarsegrain(S, 3; dims = 2) == [2 2; 1 2]
    @test size(Dewdrop.coarsegrain(S, 4; dims = 2), 2) == 1             # remainder discarded
end

@testset "susceptibility (population synchrony variance)" begin
    # fully synchronous, half on / half off → χ = ⟨ρ²⟩ − ⟨ρ⟩² = 0.5 − 0.25 = 0.25
    Ssync = Bool[1 0 1 0; 1 0 1 0; 1 0 1 0; 1 0 1 0]
    @test susceptibility(Ssync) ≈ 0.25
    # perfectly asynchronous: exactly one of four active each step → ρ ≡ 0.25 → χ = 0
    Sasync = Bool[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]
    @test susceptibility(Sasync) ≈ 0.0 atol = 1.0e-12
end

@testset "mua / temporal_average / grand_distribution / cv_isi" begin
    S = Bool[1 0 1; 1 1 0; 0 1 1]
    @test mua(S) == [2, 2, 2]                                   # population count per time step
    M = Float64[1 3; 2 6; 0 0]
    @test temporal_average(M) == [2.0, 4.0, 0.0]               # mean over time per neuron

    counts, centers = grand_distribution([0.0, 1.0, 2.0, 3.0], 2)
    @test counts == [2, 2]
    @test centers ≈ [0.75, 2.25]

    @test cv_isi([1.0, 2.0, 3.0, 4.0]) ≈ 0.0                   # regular train → CV 0
    @test cv_isi([0.0]) |> isnan                               # < 2 spikes
    @test cv_isi([1.0, 2.0, 4.0, 7.0]) ≈ sqrt(2 / 3) / 2      # isis [1,2,3]: popstd/mean
end

@testset "power_spectrum peaks at the driving frequency" begin
    dt = 1.0
    T = 256
    f0 = 0.1                                                    # cycles per unit time
    sig = reshape(Float64[cos(2π * f0 * (t - 1) * dt) for t in 1:T], 1, T)
    psd, freqs = power_spectrum(sig; n_segments = 1, dt = dt)
    @test length(psd) == T
    kpos = 2:(T ÷ 2)                                            # positive non-DC frequencies
    kpk = kpos[argmax(psd[kpos])]
    @test freqs[kpk] ≈ f0 atol = 1 / T                         # spectral peak at f0
end

@testset "efficiency (spatial coding)" begin
    # 4 neurons, 2 spatial bins of 2 neurons; uniform activity over both bins → maximal entropy
    S = Bool[1 1; 1 1; 1 1; 1 1]                                # all fire both steps
    bin_indices = reshape([[1, 2], [3, 4]], 2, 1)              # 2×1 spatial bins
    η = efficiency(S, bin_indices, 1.0; dt = 1.0)
    @test all(isfinite, η)
    @test length(η) == 2
end

@testset "radial autocorrelation on a grid" begin
    pos = grid_positions(8, 8)                                  # 64 sites, evenly spaced
    # two frames of structured + random activity
    S = Bool[(i + t) % 3 == 0 for i in 1:64, t in 1:6]
    g_r, r_bins = radial_autocorrelation(S, pos; dr = 1.0)
    @test length(g_r) == length(r_bins)
    @test g_r[1] ≈ 1.0 atol = 1.0e-8                           # zero-lag autocorrelation normalised to 1
    @test all(isfinite, g_r)
    @test all(abs.(g_r) .≤ 1.0 + 1.0e-8)                       # normalised correlation magnitude ≤ 1
    # grid validation
    @test_throws Exception radial_autocorrelation(S, pos[1:60]; dr = 1.0)
end

@testset "sol-level wrappers (addressor + positions)" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    pos = grid_positions(8, 8)
    prob = DewdropNetwork(
        m, 64; input = 30.0, tspan = (0.0, 200.0),
        subpops = (E = 1:32, I = 33:64), positions = pos
    )
    sol = solve(prob, FixedStep(0.1); v0 = (5.0, 19.0), record = (spikes = Spikes(),))

    @test susceptibility(sol) isa Float64
    @test susceptibility(sol; of = :E) == susceptibility(Array(sol.record.spikes.data)[1:32, :])
    @test mua(sol; of = :I) == vec(sum(Array(sol.record.spikes.data)[33:64, :]; dims = 1))
    @test mua(sol; bin = 5.0) |> length == sol.nsteps ÷ 50     # bin=5ms / dt=0.1 → 50-step bins
    psd, freqs = power_spectrum(sol; n_segments = 2)
    @test length(psd) == length(freqs)
    @test cv_isi(sol) ≈ 0 atol = 1.0e-12        # constant drive → perfectly regular firing
    g_r, r_bins = radial_autocorrelation(sol; dr = 1.0)        # uses sol.positions
    @test g_r[1] ≈ 1.0 atol = 1.0e-8
end

# `temporal_average(sol, var)` used to stop at the FIRST trace in the record regardless of `var`, so
# asking for one variable silently returned another.
@testset "temporal_average selects the requested variable" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    sol = solve(
        DewdropNetwork(m, 8; input = 15.0, tspan = (0.0, 50.0)), FixedStep(0.1);
        record = (V = Trace(:V), it = Trace(:itot)), progress = false
    )
    @test all(≈(15.0), temporal_average(sol, :itot))          # itot is the constant input
    @test all(<(15.0), temporal_average(sol, :V))             # V sits below it, and they differ
    @test temporal_average(sol) == temporal_average(sol, :V)  # the default is still :V
    # asking for something that was not recorded says so, and lists what was
    @test_throws ErrorException temporal_average(sol, :refrac)
end

# `cv_isi(sol)` grouped spikes with `t[id .== n]` inside a loop over neurons, rescanning the whole
# raster once per neuron. The grouped form must give exactly the same number.
@testset "cv_isi groups spikes in one pass" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    het = Heterogeneous(m; R = collect(range(1.0, 2.0; length = 40)))   # a spread of rates
    sol = solve(
        DewdropNetwork(het, 40; input = 15.0, tspan = (0.0, 400.0)), FixedStep(0.1);
        record = (spikes = Spikes(),), progress = false
    )
    t, id = raster(sol)
    @test !isempty(id)
    reference = let cvs = Float64[]                       # the original O(neurons × spikes) form
        for n in unique(id)
            ts = sort(t[id .== n])
            length(ts) ≥ 2 && push!(cvs, cv_isi(ts))
        end
        sum(cvs) / length(cvs)
    end
    @test cv_isi(sol) == reference
end

@testset "coarsegrain is unchanged by the traversal order" begin
    S = reshape(collect(1.0:60.0), 5, 12)
    reference = let bs = 4, nb = 3, out = zeros(5, 3)      # the original neuron-outer form
        for b in 1:nb, i in 1:5
            acc = 0.0
            for t in ((b - 1) * bs + 1):(b * bs)
                acc += S[i, t]
            end
            out[i, b] = acc
        end
        out
    end
    @test Dewdrop.coarsegrain(S, 4) == reference
    @test Dewdrop.coarsegrain(S .> 30, 4) == [count(>(30.0), S[i, ((b - 1) * 4 + 1):(b * 4)]) for i in 1:5, b in 1:3]
    @test eltype(Dewdrop.coarsegrain(S .> 30, 4)) === Int
    @test size(Dewdrop.coarsegrain(S, 5)) == (5, 2)        # the trailing remainder is discarded
end
