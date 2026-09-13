using Dewdrop
using Test
using Random

# StreamingFano is the on-device (batched) Fano-factor reducer (TemporalReducers.jl). Its window-boundary
# schedule is precomputed once per (τ, step) and folded per neuron/member; this checks it against the
# textbook Fano factor over non-overlapping width-τ windows. Integer step-widths make the binning exact, so
# the only difference is Float32 vs Float64 rounding in the final ratio.

# Textbook Fano over nb = ⌊(nseen−1)/k⌋ windows of k steps: window m = spikes in steps (m−1)k+1 … mk.
function fano_reference(spikes::AbstractArray{<:Real, 3}, kwidths::Vector{Int}, nseen::Int)
    n, B, _ = size(spikes)
    out = fill(NaN, n, B, length(kwidths))
    for (t, k) in enumerate(kwidths)
        nb = fld(nseen - 1, k)
        for b in 1:B, i in 1:n
            sc = 0.0; sc2 = 0.0
            for m in 1:nb
                c = 0.0
                for s in ((m - 1) * k + 1):(m * k)
                    c += spikes[i, b, s]
                end
                sc += c; sc2 += c * c
            end
            out[i, b, t] = (nb < 2 || sc == 0) ? NaN : (nb * sc2 - sc^2) / ((nb - 1) * sc)
        end
    end
    return out
end

@testset "StreamingFano matches the textbook Fano factor" begin
    Random.seed!(20260709)
    n, B, nseen = 64, 4, 1500
    dt = 1.0                                              # exact in Float32 ⇒ window boundaries land on steps
    kwidths = [3, 5, 8, 13, 21, 34, 55, 89, 144]          # integer step-widths (τ = k·dt = k)
    taus = Float64.(kwidths)
    spikes = Float32.(rand(n, B, nseen) .< 0.3)
    ref = fano_reference(spikes, kwidths, nseen)

    m = Dewdrop.StreamingFano(Dewdrop.CPU(), Float32, n, B, taus, dt, nseen + 2)
    for s in 1:nseen
        Dewdrop.update!(m, @view(spikes[:, :, s]), s)     # scalar CPU path; `s` is the recorded-sample index
    end
    got = Dewdrop.result(m, nseen)

    @test size(got) == (n, B, length(kwidths))
    @test isnan.(got) == isnan.(ref)                      # same NaN pattern (silent units / < 2 windows)
    fin = isfinite.(ref)
    @test Float64.(got[fin]) ≈ ref[fin] rtol = 1.0e-4       # Float32 reducer vs Float64 reference
    @test count(fin) > 0.5 * length(ref)                  # sanity: most entries are real Fano factors
end

# StreamingWelch accumulates raw moments (Σr, Σr²) so the global mean can be removed analytically in one
# pass. Accumulated in the STATE float type, `Σr² − (Σr)²/n` catastrophically cancels for a trace whose
# mean dwarfs its fluctuation: the ordinary membrane regime (−65 mV ± a fraction of a mV). The difference
# then goes NEGATIVE, and `result` scales the whole spectrum by it.
@testset "StreamingWelch: moments survive a large DC offset" begin
    fs, f_min, n = 1000.0, 4.0, 60_000
    ω = 0.3                                        # rad/sample → f = ω·fs/2π ≈ 47.7 Hz
    fsig = ω * fs / 2π
    for T in (Float32, Float64)
        w = Dewdrop.StreamingWelch(Dewdrop.CPU(), T, 1, 1, fs, f_min)
        for k in 1:n
            Dewdrop.update!(w, reshape([T(-65 + 0.05 * sin(ω * k))], 1, 1))
        end
        psd = vec(Dewdrop.result(w, n))
        @test all(isfinite, psd)
        @test all(≥(0), psd)                       # power is non-negative
        # the estimator still resolves the tone on top of the offset, in the right bin
        freqs = (0:(length(psd) - 1)) .* (fs / w.nfft)
        @test abs(freqs[argmax(psd)] - fsig) ≤ fs / w.nfft
    end
end

# `_welch_nfft` admits nfft = 2 at f_min = fs/2, but the symmetric Hann window `0.5 - 0.5cos(2πj/(nfft-1))`
# is then exactly [0, 0], so the window power A = 0 and every normalisation divides by zero.
@testset "StreamingWelch: a degenerate window length is refused" begin
    @test_throws ArgumentError Dewdrop._welch_nfft(10.0, 5.0)     # would give nfft = 2
    @test Dewdrop._welch_nfft(10.0, 2.5) == 4                     # the smallest usable length
    w = Dewdrop.StreamingWelch(Dewdrop.CPU(), Float64, 1, 1, 10.0, 2.5)
    @test w.A > 0
end

# The streaming reducers fold an (n_out x B) sample per step, so they exist only on the batched path.
# A scalar `solve` should say that rather than report a MethodError on an internal function.
@testset "streaming reducers report the batched-only requirement" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    prob = DewdropNetwork(m, 4; input = 0.5, tspan = (0.0, 50.0))
    for spec in (Welch(:V; f_min = 20.0), MADev(:V; lags = [1, 2]), SpikeRate(), Fano(; taus = [1.0]))
        @test_throws ArgumentError solve(prob, FixedStep(0.1); record = (r = spec,))
    end
end

@testset "StreamingWelch: segment transform is in place and correct" begin
    fs, f_min, f0, n = 100.0, 5.0, 20.0, 512
    w = Dewdrop.StreamingWelch(Dewdrop.CPU(), Float64, 1, 1, fs, f_min)
    for s in 1:n
        Dewdrop.update!(w, reshape([100.0 + sin(2π * f0 * (s - 1) / fs)], 1, 1), s)   # big DC + a tone
    end
    P = Dewdrop.result(w, n)
    @test !any(isnan, P)
    @test (argmax(vec(P[:, 1, 1])) - 1) * (fs / w.nfft) ≈ f0        # the tone survives the rewrite

    # a flat trace has no variance and no spectrum; the normalisation is 0/0 there
    wf = Dewdrop.StreamingWelch(Dewdrop.CPU(), Float64, 1, 1, fs, f_min)
    for s in 1:n
        Dewdrop.update!(wf, reshape([-70.0], 1, 1), s)
    end
    Pf = Dewdrop.result(wf, n)
    @test !any(isnan, Pf)
    @test all(iszero, Pf)

    # the segment scratch is preallocated, so closing a segment allocates no buffers
    wa = Dewdrop.StreamingWelch(Dewdrop.CPU(), Float64, 4, 2, fs, f_min)
    xt = fill(1.0, 4, 2)
    for s in 1:(2 * wa.nfft)
        Dewdrop.update!(wa, xt, s)
    end
    a = @allocated for s in 1:(wa.nfft)
        Dewdrop.update!(wa, xt, s)
    end
    @test a < 4096
end
