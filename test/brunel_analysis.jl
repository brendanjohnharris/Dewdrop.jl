using Statistics

# Shared analysis for the Brunel regime tests + the parameter-sweep probe, so both compute
# identical statistics. Dependency-light (Statistics only): a manual periodogram avoids an FFT
# dependency and is cheap at these sizes (~10^3 bins x ~300 freqs).

# Per-neuron CV-ISI averaged over neurons with >= minspikes spikes. ~1 => Poisson-irregular;
# ~0 => regular/clock-like (mean-driven).
function mean_cv_isi(times, ids, N; minspikes = 4)
    cvs = Float64[]
    for i in 1:N
        ts = sort(times[ids .== i])
        length(ts) < minspikes && continue
        isi = diff(ts)
        m = mean(isi)
        m > 0 && push!(cvs, std(isi) / m)
    end
    return isempty(cvs) ? NaN : mean(cvs)
end

# Binned population spike count A(t) over [t0, t1) in Δ-ms bins. Returns (centers_ms, counts).
function pop_activity(times, t0, t1, Δ)
    nb = max(1, floor(Int, (t1 - t0) / Δ))
    A = zeros(Float64, nb)
    @inbounds for t in times
        (t0 ≤ t < t0 + nb * Δ) || continue
        A[floor(Int, (t - t0) / Δ) + 1] += 1
    end
    centers = t0 .+ Δ .* (0:(nb - 1)) .+ Δ / 2
    return centers, A
end

# Periodogram of the mean-subtracted activity over a Hz grid (Δ in ms). Returns (freqs, power).
function pop_spectrum(A, Δ; fmin = 2.0, fmax = 300.0, df = 1.0)
    a = A .- mean(A)
    Δs = Δ / 1000
    K = length(a)
    freqs = collect(fmin:df:fmax)
    power = similar(freqs)
    @inbounds for (j, f) in enumerate(freqs)
        s = zero(ComplexF64)
        for k in 1:K
            s += a[k] * cis(-2π * f * (k - 1) * Δs)
        end
        power[j] = abs2(s) / K
    end
    return freqs, power
end

# Dominant spectral peak (Hz) + its prominence (peak power / median power). Prominence >> 1
# means a genuine global oscillation rising above the finite-size noise floor.
function dominant_peak(freqs, power)
    j = argmax(power)
    return freqs[j], power[j] / median(power)
end

# Global synchrony index: CV of the binned population activity. For an asynchronous Poisson
# population this sits at the finite-size floor 1/sqrt(mean count per bin); a synchronous state
# rises well above it.
synchrony_index(A) = std(A) / mean(A)

# Mean firing rate (Hz) over [t0, t1] computed directly from the raster (transient excluded).
function mean_rate(times, t0, t1, N)
    return 1000 * count(t -> t0 ≤ t ≤ t1, times) / (N * (t1 - t0))
end
