# * Internal FFT (host-side analysis only; NO external dependency). Implements the standard DFT
#       X[k] = Σ_n x[n] · exp(-2πi k n / N)
# (numpy / `jnp.fft` convention) via recursive radix-2 Cooley--Tukey for power-of-2 lengths and
# Bluestein's chirp-z transform for arbitrary lengths, so the spectral observables (`power_spectrum`,
# `radial_autocorrelation` in Stats.jl) match a numpy reference for ANY length. A direct O(N²) `_dft`
# is kept as the correctness reference (the tests cross-check `_fft` against it). This is a deliberately
# self-contained, dependency-free transform; it runs on the host at analysis time, not in the hot loop,
# so the O(N log N) pure-Julia implementation is more than adequate. (FFTW is a package dependency and
# could back these instead; see the breaking-change report for the trade-offs.)

# Direct DFT / inverse DFT (the O(N²) reference). `sign = -1` forward, `+1` inverse (unnormalised).
function _dft(x::AbstractVector{<:Number}, sign::Int = -1)
    n = length(x)
    out = zeros(ComplexF64, n)
    @inbounds for k in 0:(n - 1)
        s = zero(ComplexF64)
        for j in 0:(n - 1)
            s += x[j + 1] * cis(sign * 2π * k * j / n)
        end
        out[k + 1] = s
    end
    return out
end

"""
    _fft(x) -> Vector{ComplexF64}

Forward DFT of `x` (numpy convention), radix-2 for power-of-2 length, Bluestein otherwise.
"""
function _fft(x::AbstractVector{<:Number})
    n = length(x)
    n ≤ 1 && return ComplexF64.(collect(x))
    return ispow2(n) ? _fft_radix2(ComplexF64.(collect(x))) : _bluestein(ComplexF64.(collect(x)))
end

# Inverse DFT (normalised): `ifft(x) = conj(fft(conj(x))) / N`.
_ifft(x::AbstractVector{<:Number}) = (n = length(x); conj.(_fft(conj.(ComplexF64.(collect(x))))) ./ n)

# Recursive radix-2 Cooley--Tukey (decimation in time), power-of-2 length.
function _fft_radix2(x::Vector{ComplexF64})
    n = length(x)
    n == 1 && return x
    even = _fft_radix2(x[1:2:end])
    odd = _fft_radix2(x[2:2:end])
    out = Vector{ComplexF64}(undef, n)
    half = n ÷ 2
    @inbounds for k in 0:(half - 1)
        t = cis(-2π * k / n) * odd[k + 1]
        out[k + 1] = even[k + 1] + t
        out[k + half + 1] = even[k + 1] - t
    end
    return out
end

# Bluestein's algorithm: an arbitrary-length DFT as a convolution evaluated by a power-of-2 FFT.
# The chirp angle uses `k² mod 2n` so it stays exact for large `k` (the angle is 2π-periodic).
function _bluestein(x::Vector{ComplexF64})
    n = length(x)
    m = nextpow(2, 2n - 1)
    w = ComplexF64[cis(-π * mod(k * k, 2n) / n) for k in 0:(n - 1)]   # chirp
    a = zeros(ComplexF64, m)
    @inbounds for k in 0:(n - 1)
        a[k + 1] = x[k + 1] * w[k + 1]
    end
    b = zeros(ComplexF64, m)
    @inbounds b[1] = conj(w[1])
    @inbounds for k in 1:(n - 1)
        b[k + 1] = conj(w[k + 1])
        b[m - k + 1] = conj(w[k + 1])                                # b is symmetric (b[-k] = b[k])
    end
    fc = _fft_radix2(a) .* _fft_radix2(b)
    c = conj.(_fft_radix2(conj.(fc))) ./ m                           # inverse FFT (size m, power of 2)
    return ComplexF64[w[k + 1] * c[k + 1] for k in 0:(n - 1)]
end

# 2-D forward / inverse DFT (transform rows, then columns), for the radial autocorrelation.
function _fft2(A::AbstractMatrix{<:Number})
    M = ComplexF64.(A)
    for i in 1:size(M, 1)
        @inbounds M[i, :] = _fft(M[i, :])
    end
    for j in 1:size(M, 2)
        @inbounds M[:, j] = _fft(M[:, j])
    end
    return M
end
function _ifft2(A::AbstractMatrix{<:Number})
    M = ComplexF64.(A)
    for i in 1:size(M, 1)
        @inbounds M[i, :] = _ifft(M[i, :])
    end
    for j in 1:size(M, 2)
        @inbounds M[:, j] = _ifft(M[:, j])
    end
    return M
end

# numpy `fftfreq(n, d)`: [0, 1, …, ⌈n/2⌉-1, -⌊n/2⌋, …, -1] / (n·d).
function _fftfreq(n::Integer, d::Real = 1.0)
    f = Vector{Float64}(undef, n)
    half = (n - 1) ÷ 2 + 1                       # number of non-negative frequencies
    @inbounds for k in 0:(half - 1)
        f[k + 1] = k / (n * d)
    end
    @inbounds for (i, k) in enumerate(-(n ÷ 2):-1)
        f[half + i] = k / (n * d)
    end
    return f
end
