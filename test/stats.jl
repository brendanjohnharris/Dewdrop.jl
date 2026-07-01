using Dewdrop
using Test

# Statistical observables. The spectral measures rest on the internal
# FFT (FFT.jl), so the FFT is anchored against a direct DFT reference for arbitrary lengths; the
# measures themselves are checked on hand-constructed rasters with known values, plus the sol-level
# wrappers on a small recorded network. (Exact agreement with the reference stats.py/numpy is the
# separate cross-validation step in test/simulator_comparisons/stats_validation/.)

@testset "internal FFT ≡ direct DFT (radix-2 + Bluestein, any length)" begin
    for n in (1, 2, 3, 4, 5, 7, 8, 12, 16, 17, 31, 64)
        x = ComplexF64[cospi(0.3k) + im * sinpi(0.11k + 0.2) for k in 0:(n - 1)]
        @test Dewdrop._fft(x) ≈ Dewdrop._dft(x, -1)             # forward matches the O(N²) reference
        @test Dewdrop._ifft(Dewdrop._fft(x)) ≈ x               # round-trip
    end
    # real input, power of 2 and not
    xr = Float64[sinpi(0.25k) for k in 0:9]
    @test Dewdrop._fft(xr) ≈ Dewdrop._dft(xr, -1)
    # fftfreq matches the numpy convention
    @test Dewdrop._fftfreq(8, 1.0) ≈ [0, 1, 2, 3, -4, -3, -2, -1] ./ 8
    @test Dewdrop._fftfreq(5, 0.5) ≈ [0, 1, 2, -2, -1] ./ (5 * 0.5)
    # 2-D round trip
    A = reshape(Float64.(1:12), 3, 4)
    @test real.(Dewdrop._ifft2(Dewdrop._fft2(A))) ≈ A
end

@testset "coarsegrain (time binning)" begin
    S = Bool[
        1 0 1 1 0 1
        0 0 1 0 1 1
    ]
    @test coarsegrain(S, 2; dims = 2) == [1 2 1; 0 1 2]          # sum each 2-step bin per neuron
    @test coarsegrain(S, 3; dims = 2) == [2 2; 1 2]
    @test size(coarsegrain(S, 4; dims = 2), 2) == 1             # remainder discarded
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
    @test cv_isi(sol) ≥ 0
    g_r, r_bins = radial_autocorrelation(sol; dr = 1.0)        # uses sol.positions
    @test g_r[1] ≈ 1.0 atol = 1.0e-8
end
