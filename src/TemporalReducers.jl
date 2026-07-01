# * Streaming temporal reducers (M-stream).
# Consume ONE (n_out × B) sample per step and accumulate a reduced temporal statistic on-device, so a
# long per-(neuron, member) trace is NEVER materialised --- not on the host, not on the device. This is
# the streaming analogue of running a windowed/lag estimator over a fully recorded trace; the only memory
# that scales with the run is a fixed ring/segment buffer (O(max lag) or O(nfft)), not O(nsteps). The
# `update!`/`result` pair is backend-generic (KernelAbstractions + AbstractFFTs), so the same code runs on
# CPU and GPU; only the small reduced result is brought to the host.

using KernelAbstractions: @kernel, @index, @Const, get_backend
using FFTW: rfft, plan_rfft

# --- StreamingMADev: p=1 mean absolute displacement at integer lags ------------------------------------
# The streaming form of `TimeseriesTools.madev(x, lags; p = 1)` = `mean_t |x(t) − x(t+k)|` over the `n−k`
# valid pairs, per lag `k`. A ring of the last `maxlag + 1` samples per (n_out, B) lets each new sample
# x(s) accumulate `|x(s) − x(s−k)|` for every lag `k < s`; `result` divides by the pair count `nseen − k`.
struct StreamingMADev{RB, AC, LV}
    ring::RB     # (n_out, B, Lring), Lring = maxlag + 1: the last Lring samples, ring-indexed by step
    acc::AC      # (n_out, B, nlags): running Σ |x(s) − x(s−k)|
    lags::LV     # device vector of sorted positive Int lags
    Lring::Int
end
Adapt.@adapt_structure StreamingMADev

function StreamingMADev(arch, ::Type{T}, n_out::Integer, B::Integer, lags) where {T}
    lagv = collect(Int, lags)
    all(>(0), lagv) || throw(ArgumentError("madev lags must be positive integers (got $lagv)"))
    issorted(lagv) || sort!(lagv)
    Lring = maximum(lagv) + 1
    ring = fill!(allocate(arch, T, Int(n_out), Int(B), Lring), zero(T))
    acc = fill!(allocate(arch, T, Int(n_out), Int(B), length(lagv)), zero(T))
    return StreamingMADev(ring, acc, on_architecture(arch, lagv), Lring)
end

# One thread per (neuron, member): store the current sample into its ring slot, then accumulate the
# absolute increment against each lag's earlier sample. Each thread owns its own ring/acc rows → no races.
@kernel function _madev_update_kernel!(ring, acc, @Const(xt), @Const(lags), Lring, s)
    i, b = @index(Global, NTuple)
    @inbounds begin
        x = xt[i, b]
        ring[i, b, mod(s - 1, Lring) + 1] = x          # store x(s); k ≤ maxlag < Lring ⇒ never aliases x(s−k)
        for li in eachindex(lags)
            k = lags[li]
            if s > k
                acc[i, b, li] += abs(x - ring[i, b, mod(s - k - 1, Lring) + 1])
            end
        end
    end
end

# Feed the s-th (1-based) recorded sample `xt` (an (n_out × B) view of the engine's selected state).
function update!(m::StreamingMADev, xt::AbstractMatrix, s::Integer)
    _madev_update_kernel!(get_backend(m.acc))(m.ring, m.acc, xt, m.lags, m.Lring, Int(s); ndrange = size(xt))
    return m
end

# Finalise to a host `(n_out, B, nlags)` array: each lag's running sum / its pair count `nseen − k`
# (0 where the lag never had a valid pair). Small (no nsteps axis), so the device→host copy is cheap.
function result(m::StreamingMADev, nseen::Integer)
    lagv = Array(m.lags)
    out = Array(m.acc)
    @inbounds for li in eachindex(lagv)
        np = nseen - lagv[li]
        @views out[:, :, li] .*= (np > 0 ? inv(float(eltype(out))(np)) : zero(eltype(out)))
    end
    return out
end

# --- StreamingWelch: averaged Welch periodogram (power spectrum) ---------------------------------------
# The streaming form of `TimeseriesTools.spectrum(x .- mean(x), f_min)` (a Hann-windowed, 50%-overlap Welch
# periodogram, Parseval-normalised). Segments complete at the same step for every (neuron, member), so each
# completed segment is transformed with ONE batched `rfft` along the time axis (FFTW on CPU, CUFFT on GPU).
#
# Global mean removal is NOT one-pass directly, but the rFFT is linear: with raw segment `r` and window `h`,
# `rfft((r − μ)·h) = rfft(r·h) − μ·rfft(h) = Yr − μ·H`. So we accumulate the RAW windowed-segment transforms
# (`ΣYr` complex and `Σ|Yr|²` real) plus the raw moments `Σr, Σr²`, and apply the exact mean (μ = Σr/n) and
# Parseval normalisation analytically in `result`. This keeps the pass single and the result bit-equal to the
# batch reference (to FFT round-off). nfft / overlap / window / normalisation mirror `_periodogram` (padding 0).
mutable struct StreamingWelch{BUF, AP2, AP1, SX, HW, HV, PL, T}
    const buf::BUF       # (nfft, n_out, B) sliding raw segment buffer (no mean removal)
    const accP2::AP2     # (nfreq, n_out, B) real:    Σ_seg |Yr|²  (Yr = rfft(raw_seg .* hann))
    const accP1::AP1     # (nfreq, n_out, B) complex: Σ_seg  Yr
    const sumr::SX       # (n_out, B): Σ_t r(t)
    const sumr2::SX      # (n_out, B): Σ_t r(t)²
    const hann::HW       # (nfft,) device window
    const H::HV          # (nfreq,) HOST complex = rfft(hann); used for the mean correction in `result`
    const plan::PL       # batched rfft plan along dim 1 of an (nfft, n_out, B) array
    const nfft::Int
    const overlap::Int
    const nfreq::Int
    const A::T           # Σ hann²  (window power)
    const fs::T          # sampling rate (= 1/dt)
    fill::Int            # samples currently staged in `buf`
    nseg::Int            # completed segments transformed so far
end

# nfft from (fs, f_min), matching `_periodogram`: ceil(fs/f_min), forced even (padding = 0 here).
function _welch_nfft(fs::Real, f_min::Real)
    f_min > 0 || throw(ArgumentError("Welch f_min must be > 0 (got $f_min)"))
    nfft = ceil(Int, float(fs) / float(f_min))
    isodd(nfft) && (nfft += 1)
    nfft ≥ 2 || throw(ArgumentError("Welch nfft = $nfft < 2; lower f_min or raise fs"))
    return nfft
end

function StreamingWelch(arch, ::Type{T}, n_out::Integer, B::Integer, fs::Real, f_min::Real) where {T}
    nfft = _welch_nfft(fs, f_min)
    overlap = nfft ÷ 2
    nfreq = nfft ÷ 2 + 1
    hann_h = T[0.5 - 0.5 * cos(2 * π * j / (nfft - 1)) for j in 0:(nfft - 1)]   # _periodogram window
    A = sum(abs2, hann_h)
    H = rfft(hann_h)                                                            # (nfreq,) host complex
    buf = fill!(allocate(arch, T, nfft, Int(n_out), Int(B)), zero(T))
    accP2 = fill!(allocate(arch, T, nfreq, Int(n_out), Int(B)), zero(T))
    accP1 = fill!(allocate(arch, Complex{T}, nfreq, Int(n_out), Int(B)), zero(Complex{T}))
    sumr = fill!(allocate(arch, T, Int(n_out), Int(B)), zero(T))
    sumr2 = fill!(allocate(arch, T, Int(n_out), Int(B)), zero(T))
    hann = on_architecture(arch, hann_h)
    plan = plan_rfft(buf, 1)
    return StreamingWelch{
        typeof(buf), typeof(accP2), typeof(accP1), typeof(sumr), typeof(hann),
        typeof(H), typeof(plan), T,
    }(
        buf, accP2, accP1, sumr, sumr2, hann, H, plan, nfft, overlap, nfreq, T(A), T(fs), 0, 0
    )
end

# Feed the s-th recorded sample `xt` ((n_out × B) view of the engine's selected state). Accumulate raw
# moments every step; on a full segment, batch-rFFT it and accumulate the windowed transform, then slide.
function update!(m::StreamingWelch, xt::AbstractMatrix, ::Integer = 0)
    m.sumr .+= xt
    m.sumr2 .+= abs2.(xt)
    m.fill += 1
    @inbounds @views m.buf[m.fill, :, :] .= xt
    if m.fill == m.nfft
        seg = m.buf .* reshape(m.hann, m.nfft, 1, 1)     # Hann-windowed raw segment
        Yr = m.plan * seg                                # (nfreq, n_out, B) complex
        m.accP2 .+= abs2.(Yr)
        m.accP1 .+= Yr
        m.nseg += 1
        ov = m.overlap
        @inbounds @views m.buf[1:ov, :, :] .= m.buf[(m.nfft - ov + 1):m.nfft, :, :]   # slide: keep last `overlap`
        m.fill = ov
    end
    return m
end

# Finalise to a host `(nfreq, n_out, B)` power spectrum, applying the analytic global-mean correction and the
# `_periodogram`/`_powerspectrum`/`powerspectrum` Parseval normalisation. All host-side over small arrays.
function result(m::StreamingWelch, nseen::Integer)
    m.nseg ≥ 1 || throw(ArgumentError("StreamingWelch: no complete segments (nseen = $nseen < nfft = $(m.nfft))"))
    P2 = Array(m.accP2) ./ m.nseg                       # (nfreq, n_out, B)
    P1 = Array(m.accP1) ./ m.nseg                       # complex
    H = m.H                                             # (nfreq,) complex
    sr = Array(m.sumr); sr2 = Array(m.sumr2)            # (n_out, B)
    nfft, A, fs, nfreq = m.nfft, m.A, m.fs, m.nfreq
    df = fs / nfft
    duration = (nseen - 1) / fs                         # (last − first) time, = (n−1)·dt
    norm = nfft^2 * A
    n_out, B = size(sr)
    out = Array{eltype(P2)}(undef, nfreq, n_out, B)
    meanS = Vector{eltype(P2)}(undef, nfreq)
    @inbounds for b in 1:B, i in 1:n_out
        μ = sr[i, b] / nseen
        sumx2 = sr2[i, b] - sr[i, b]^2 / nseen          # Σ (r − μ)²
        for f in 1:nfreq
            meanS[f] = (P2[f, i, b] - 2μ * real(P1[f, i, b] * conj(H[f])) + μ^2 * abs2(H[f])) / norm
        end
        scalar1 = (sum(meanS) - 0.5 * meanS[1]) * df
        fac = 0.5 * (sumx2 / fs) / (scalar1 * duration)
        for f in 1:nfreq
            out[f, i, b] = fac * meanS[f]
        end
    end
    return out
end

# --- StreamingRate: per-(neuron, member) mean firing rate -----------------------------------------------
# Total spikes / observation time: a single (n_out, B) cumulative count, finalised as `count / (nseen·dt)`
# (the per-cell `firing_rate` convention, duration = nsteps·dt). `xt` is the per-step Bool spike slice.
struct StreamingRate{C, T}
    count::C     # (n_out, B): Σ_t spike(t)
    dt::T
end
Adapt.@adapt_structure StreamingRate

function StreamingRate(arch, ::Type{T}, n_out::Integer, B::Integer, dt::Real) where {T}
    count = fill!(allocate(arch, T, Int(n_out), Int(B)), zero(T))
    return StreamingRate(count, T(dt))
end

update!(m::StreamingRate, xt::AbstractMatrix, ::Integer = 0) = (m.count .+= xt; m)
result(m::StreamingRate, nseen::Integer) = Array(m.count) ./ (nseen * m.dt)   # (n_out, B) rate

# --- StreamingFano: per-(neuron, member) Fano factor curve over count windows ---------------------------
# The streaming form of `TimeseriesTools.fano_factor(spikes, τs)` = `var(counts; mean=m)/m` (Bessel-corrected)
# of spike counts in width-`τ` windows, per timescale `τ`. Windows are a FIXED grid aligned to the recording
# start (origin 0): window j of τ is `[jτ, (j+1)τ)`. We keep a cumulative per-neuron count `cum`; when a
# sample's time crosses into a new window for τ (`floor(t/τ)` increments, ≤ 1 crossing/step since τ ≥ dt), the
# just-completed window's count `cum − cum_last` folds into Σcount / Σcount², and `cum_last ← cum`. Only
# complete windows are counted (the trailing partial is dropped), and `result` forms `(nb·Σc² − (Σc)²)/((nb−1)·Σc)`
# with `nb = floor(t_max/τ)`. (Origin 0 vs TimeseriesTools' per-neuron min-spike origin → agrees up to bin
# alignment, not bit-identical.)
struct StreamingFano{CU, AC, TV, T}
    cum::CU      # (n_out, B): cumulative Σ spike(t) up to (not incl.) the current step
    cum_last::AC # (n_out, B, ntau): cum at each τ's last window boundary
    sumc::AC     # (n_out, B, ntau): Σ window counts
    sumc2::AC    # (n_out, B, ntau): Σ window counts²
    taus::TV     # device vector of timescales (same time units as dt)
    dt::T
end
Adapt.@adapt_structure StreamingFano

function StreamingFano(arch, ::Type{T}, n_out::Integer, B::Integer, taus, dt::Real) where {T}
    tv = collect(T, taus)
    all(>(0), tv) || throw(ArgumentError("Fano timescales must be positive"))
    ntau = length(tv)
    z3() = fill!(allocate(arch, T, Int(n_out), Int(B), ntau), zero(T))
    return StreamingFano(
        fill!(allocate(arch, T, Int(n_out), Int(B)), zero(T)), z3(), z3(), z3(),
        on_architecture(arch, tv), T(dt)
    )
end

# One thread per (neuron, member, timescale): on a window boundary (this step's time entered a new τ-window),
# fold the just-completed window's count into the per-τ moment sums. `cum` holds counts through step s−1.
@kernel function _fano_step_kernel!(@Const(cum), cum_last, sumc, sumc2, @Const(taus), t, dt, s)
    i, b, k = @index(Global, NTuple)
    @inbounds if s >= 2
        τ = taus[k]
        if floor(t / τ) > floor((t - dt) / τ)                # crossed into a new window ⇒ previous one is complete
            c = cum[i, b] - cum_last[i, b, k]
            sumc[i, b, k] += c
            sumc2[i, b, k] += c * c
            cum_last[i, b, k] = cum[i, b]
        end
    end
end

function update!(m::StreamingFano, xt::AbstractMatrix, s::Integer)
    t = (Int(s) - 1) * m.dt                                   # time of sample s (0-based: 0, dt, 2dt, …)
    _fano_step_kernel!(get_backend(m.cum))(m.cum, m.cum_last, m.sumc, m.sumc2, m.taus, t, m.dt, Int(s); ndrange = size(m.sumc))
    m.cum .+= xt                                              # add step s AFTER the flush (cum was through s−1)
    return m
end

function result(m::StreamingFano, nseen::Integer)
    taus = Array(m.taus); dt = m.dt
    sumc = Array(m.sumc); sumc2 = Array(m.sumc2)
    n_out, B, ntau = size(sumc)
    tmax = (nseen - 1) * dt
    out = Array{eltype(sumc)}(undef, n_out, B, ntau)
    nan = eltype(out)(NaN)
    @inbounds for k in 1:ntau
        nb = floor(Int, tmax / taus[k])                       # number of complete windows
        for b in 1:B, i in 1:n_out
            sc = sumc[i, b, k]; sc2 = sumc2[i, b, k]
            out[i, b, k] = (nb < 2 || sc == 0) ? nan : (nb * sc2 - sc^2) / ((nb - 1) * sc)
        end
    end
    return out
end

# --- User-facing temporal-monitor specs (materialised into batched monitors; BATCHED solve path only) ---
# These hook the per-step recording loop and fold each sample into a streaming reducer, so the long trace is
# never stored. Parameters are in RECORDED-SAMPLE units (lags, transient = step counts; the caller converts
# physical time → steps), except `Welch.f_min`, a frequency resolved against the recorded rate fs = 1/(every·dt).
"""
    MADev(var; of=:all, lags, transient=0, every=1)

Stream the p=1 mean-absolute-displacement of `var` (`:itot`, `:V`, …) at integer step `lags`, skipping the
first `transient` recorded steps. On-device analogue of `TimeseriesTools.madev`; result is `(n_out, B, nlags)`.
"""
struct MADev
    var::Symbol
    of::Any
    lags::Vector{Int}
    transient::Int
    every::Int
end
MADev(var::Symbol; of = :all, lags, transient::Integer = 0, every::Integer = 1) =
    MADev(var, of, collect(Int, lags), Int(transient), Int(every))

"""
    Welch(var; of=:all, f_min, transient=0, every=1)

Stream the Hann-windowed, 50%-overlap Welch power spectrum of `var` (mean removed), with minimum resolved
frequency `f_min` (window `nfft = ceil(fs/f_min)`, `fs = 1/(every·dt)`), skipping the first `transient`
recorded steps. On-device analogue of `TimeseriesTools.spectrum(x .- mean(x), f_min)`; result is `(n_out, B, nfreq)`.
"""
struct Welch
    var::Symbol
    of::Any
    f_min::Float64
    transient::Int
    every::Int
end
Welch(var::Symbol; of = :all, f_min, transient::Integer = 0, every::Integer = 1) =
    Welch(var, of, Float64(f_min), Int(transient), Int(every))

"""
    SpikeRate(; of=:all, transient=0, every=1)

Stream the per-(neuron, member) mean firing rate (total spikes / observation time) of the selected units,
skipping the first `transient` recorded steps. Result is `(n_out, B)`.
"""
struct SpikeRate
    of::Any
    transient::Int
    every::Int
end
SpikeRate(; of = :all, transient::Integer = 0, every::Integer = 1) = SpikeRate(of, Int(transient), Int(every))

"""
    Fano(; of=:all, taus, transient=0, every=1)

Stream the per-(neuron, member) Fano factor curve (variance/mean of spike counts in width-`τ` windows) at
timescales `taus` (same time units as `dt`), skipping the first `transient` recorded steps. On-device analogue
of `TimeseriesTools.fano_factor(spikes, taus)` (fixed grid aligned to the recording start). Result is `(n_out, B, ntau)`.
"""
struct Fano
    of::Any
    taus::Vector{Float64}
    transient::Int
    every::Int
end
Fano(; of = :all, taus, transient::Integer = 0, every::Integer = 1) =
    Fano(of, collect(Float64, taus), Int(transient), Int(every))

export MADev, Welch, SpikeRate, Fano
