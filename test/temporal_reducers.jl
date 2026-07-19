using Dewdrop
using Test
using Random

# StreamingFano is the on-device (batched) Fano-factor reducer (TemporalReducers.jl). Its window-boundary
# schedule is precomputed once per (τ, step) and folded per neuron/member; this checks it against the
# textbook Fano factor over non-overlapping width-τ windows. Integer step-widths make the binning exact, so
# the only difference is Float32 vs Float64 rounding in the final ratio.

# Textbook Fano over nb = ⌊(nseen−1)/k⌋ windows of k steps: window m = spikes in steps (m−1)k+1 … mk.
function fano_reference(spikes::AbstractArray{<:Real,3}, kwidths::Vector{Int}, nseen::Int)
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
    @test Float64.(got[fin]) ≈ ref[fin] rtol = 1e-4       # Float32 reducer vs Float64 reference
    @test count(fin) > 0.5 * length(ref)                  # sanity: most entries are real Fano factors
end
