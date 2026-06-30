# * Statistical observables (analysis layer) --- spatial-network measures,
# operating on the recorded traces of a `DewdropSolution`. Two layers: matrix-core functions over a
# Neuron×Time raster `S` (Dewdrop's recording orientation), and
# `sol`-based wrappers that pull the `Spikes`/`Trace` data, the time step, positions, and the
# named-subpopulation registry (`of = :E`) from the solution. Spectral measures use the internal FFT
# (FFT.jl). These are host-side analysis, not engine work --- pure functions of recorded output.

# population mean / variance without a Statistics dependency (population variance, ÷N, the
# `mean(x²) − mean(x)²` form used by the susceptibility measure).
@inline _mean(x) = sum(x) / length(x)
@inline _popvar(x) = (μ = _mean(x); _mean(abs2.(x .- μ)))

# extract a Neuron×Time spike raster (Bool) from the solution's Spikes monitor, restricted to subpop
# `of`. Requires a full `Spikes()` recording (idx = Colon) when slicing a named subpop.
function _spike_raster(sol::DewdropSolution; of = :all, name = nothing)
    res = _find_spikes(sol.record, name)
    res === nothing && error("no spikes recorded --- pass `record = (spikes = Spikes(),)` to `solve`")
    data = res.data                                   # (neuron × time), over the recorded set res.idx
    of === :all && return data
    r = _subrange(sol.subpops, of)
    res.idx isa Colon || error("subpop slicing (`of = :$of`) needs a full `Spikes()` recording (idx = :all)")
    return @view data[r, :]
end

"""
    coarsegrain(S, binsize; dims=2) -> Matrix

Sum a matrix into non-overlapping bins of width `binsize` along `dims` (default the time axis, dim 2
for a Neuron×Time raster); the trailing remainder is discarded.
"""
function coarsegrain(S::AbstractMatrix, binsize::Integer; dims::Integer = 2)
    bs = Int(binsize)
    bs ≥ 1 || throw(ArgumentError("binsize must be ≥ 1 (got $bs)"))
    T = eltype(S) === Bool ? Int : eltype(S)
    if dims == 2
        nb = size(S, 2) ÷ bs
        out = zeros(T, size(S, 1), nb)
        @inbounds for b in 1:nb, i in 1:size(S, 1)
            acc = zero(T)
            for t in ((b - 1) * bs + 1):(b * bs)
                acc += S[i, t]
            end
            out[i, b] = acc
        end
        return out
    elseif dims == 1
        nb = size(S, 1) ÷ bs
        out = zeros(T, nb, size(S, 2))
        @inbounds for j in 1:size(S, 2), b in 1:nb
            acc = zero(T)
            for i in ((b - 1) * bs + 1):(b * bs)
                acc += S[i, j]
            end
            out[b, j] = acc
        end
        return out
    end
    throw(ArgumentError("dims must be 1 or 2 (got $dims)"))
end
export coarsegrain

"""
    susceptibility(S) -> χ
    susceptibility(sol; bin=nothing, of=:all) -> χ

Population susceptibility `χ = ⟨ρ²⟩_t − ⟨ρ⟩_t²`, where `ρ(t)` is the fraction of active neurons at
time `t` of a Neuron×Time raster `S` (the variance of the population activity fraction --- large in a
synchronous regime, small in an asynchronous one). `bin` (in time units) coarse-grains time first.
"""
function susceptibility(S::AbstractMatrix)
    nN, nt = size(S)
    ρ = Vector{Float64}(undef, nt)
    @inbounds for t in 1:nt
        c = 0
        for i in 1:nN
            c += (S[i, t] > 0)
        end
        ρ[t] = c / nN
    end
    return _popvar(ρ)
end
function susceptibility(sol::DewdropSolution; bin = nothing, of = :all, name = nothing)
    S = _spike_raster(sol; of = of, name = name)
    bin === nothing || (S = coarsegrain(S, round(Int, bin / sol.dt); dims = 2))
    return susceptibility(S)
end
export susceptibility

"""
    mua(S) -> Vector
    mua(sol; bin=nothing, of=:all) -> Vector

Multi-unit activity: the population spike count per time bin (sum over neurons of a Neuron×Time
raster). `bin` (in time units) coarse-grains time first.
"""
mua(S::AbstractMatrix) = vec(sum(S; dims = 1))
function mua(sol::DewdropSolution; bin = nothing, of = :all, name = nothing)
    S = _spike_raster(sol; of = of, name = name)
    bin === nothing || (S = coarsegrain(S, round(Int, bin / sol.dt); dims = 2))
    return mua(S)
end
export mua

"""
    temporal_average(M) -> Vector
    temporal_average(sol, var=:V; of=:all) -> Vector

Per-unit average over time of a Neuron×Time trace `M` (e.g. the mean membrane potential). The
`sol` form reads the recorded `var` trace (`Trace(var)`).
"""
temporal_average(M::AbstractMatrix) = vec(sum(M; dims = 2)) ./ size(M, 2)
function temporal_average(sol::DewdropSolution, var::Symbol = :V; of = :all)
    res = nothing
    for r in values(sol.record)
        r.kind === :trace && (res = r; break)
    end
    res === nothing && error("no trace recorded --- pass e.g. `record = (V = Trace(:V),)` to `solve`")
    data = res.data
    of === :all && return temporal_average(data)
    r = _subrange(sol.subpops, of)
    res.idx isa Colon || error("subpop slicing needs a full `Trace($var)` recording")
    return temporal_average(@view data[r, :])
end
export temporal_average

"""
    grand_distribution(X, nbins) -> (counts, centers)

Histogram of all values in `X` into `nbins` equal-width bins spanning `[min, max]`; returns the bin
counts and bin centres.
"""
function grand_distribution(X, nbins::Integer)
    nbins ≥ 1 || throw(ArgumentError("nbins must be ≥ 1"))
    lo = float(minimum(X))
    hi = float(maximum(X))
    bw = (hi - lo) / nbins
    edges = range(lo, hi; length = nbins + 1)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in 1:nbins]
    counts = zeros(Int, nbins)
    for x in X
        idx = bw == 0 ? 1 : clamp(floor(Int, (x - lo) / bw) + 1, 1, nbins)
        counts[idx] += 1
    end
    return counts, centers   # `centers` is already a Vector from the comprehension
end
export grand_distribution

"""
    cv_isi(times) -> Float64
    cv_isi(sol; of=:all) -> Float64

Coefficient of variation of inter-spike intervals (population std ÷ mean of the ISIs). The `times`
form takes one neuron's sorted spike times; the `sol` form averages the per-neuron CV-ISI over
neurons in subpopulation `of` (neurons with < 2 spikes are skipped). NaN if undefined.
"""
function cv_isi(times::AbstractVector)
    length(times) < 2 && return NaN
    isis = diff(sort(times))
    μ = _mean(isis)
    μ == 0 && return NaN
    return sqrt(_popvar(isis)) / μ
end
function cv_isi(sol::DewdropSolution; of = :all, name = nothing)
    t, id = raster(sol; of = of, name = name)
    isempty(id) && return NaN
    cvs = Float64[]
    for n in unique(id)
        ts = sort(t[id .== n])
        length(ts) ≥ 2 && push!(cvs, cv_isi(ts))
    end
    return isempty(cvs) ? NaN : _mean(cvs)
end
export cv_isi

"""
    power_spectrum(S; n_segments=1, dt=1.0) -> (psd, freqs)
    power_spectrum(sol; n_segments=1, of=:all) -> (psd, freqs)

Bartlett-averaged power spectral density of a Neuron×Time raster: split time into `n_segments`
segments, form the periodogram `|FFT|²/seg` of each, average over segments and neurons. Returns the
PSD and the `fftfreq` frequency axis (from the segment length and `dt`).
"""
function power_spectrum(S::AbstractMatrix; n_segments::Integer = 1, dt::Real = 1.0)
    nN, Tn = size(S)
    seg = Tn ÷ Int(n_segments)
    seg ≥ 1 || throw(ArgumentError("n_segments = $n_segments too large for $Tn time steps"))
    psd = zeros(Float64, seg)
    @inbounds for i in 1:nN, s in 1:n_segments
        F = _fft(Float64.(@view S[i, ((s - 1) * seg + 1):(s * seg)]))
        for k in 1:seg
            psd[k] += abs2(F[k]) / seg
        end
    end
    psd ./= (Int(n_segments) * nN)
    return psd, _fftfreq(seg, dt)
end
function power_spectrum(sol::DewdropSolution; n_segments::Integer = 1, of = :all, name = nothing)
    S = _spike_raster(sol; of = of, name = name)
    return power_spectrum(S; n_segments = n_segments, dt = sol.dt)
end
export power_spectrum

"""
    efficiency(S, bin_indices, tau; dt=1.0) -> Vector

Spatial coding efficiency per time bin: coarse-grain time by `tau`, group neurons into spatial bins
(`bin_indices[i,j]` is a vector of neuron indices in spatial bin `(i,j)`), form the spatial spike
distribution per time bin, its entropy `H`, and the energy cost `C` (total spikes); `η = n·H/C`
(`n` = neuron count).
"""
function efficiency(S::AbstractMatrix, bin_indices::AbstractMatrix{<:AbstractVector{<:Integer}}, tau; dt::Real = 1.0)
    tbinned = coarsegrain(S, round(Int, tau / dt); dims = 2)        # neuron × n_tbins
    nbx, nby = size(bin_indices)
    ntb = size(tbinned, 2)
    xbinned = zeros(Float64, nbx, nby, ntb)
    @inbounds for i in 1:nbx, j in 1:nby, t in 1:ntb
        acc = 0.0
        for n in bin_indices[i, j]
            acc += tbinned[n, t]
        end
        xbinned[i, j, t] = acc
    end
    H = zeros(Float64, ntb)
    @inbounds for t in 1:ntb
        for j in 1:nby                                              # normalise over the first spatial axis (i)
            col = 0.0
            for i in 1:nbx
                col += xbinned[i, j, t]
            end
            for i in 1:nbx
                p = col == 0 ? 0.0 : xbinned[i, j, t] / col
                H[t] -= p * log(p + 1.0e-10)
            end
        end
    end
    C = vec(sum(tbinned; dims = 1))                                 # energy cost: total spikes per time bin
    n = size(S, 1)
    return n .* H ./ C
end
export efficiency

"""
    radial_autocorrelation(S, positions; dr=0.05) -> (g_r, r_bins)
    radial_autocorrelation(sol; dr=0.05, of=:all) -> (g_r, r_bins)

Radially-averaged spatial autocorrelation of a Neuron×Time raster on a rectangular lattice
(`positions` a flat list of `(x, y)` site coordinates). Each time frame is reshaped to the grid, its
2-D autocorrelation computed by FFT (`C = ifft2(|fft2(f₀)|²)/(XY)`, normalised to `C(0)=1`),
radially averaged into shells of width `dr`, then averaged over the variance-nonzero frames. Returns
the radial profile `g_r` (with `g_r[1] ≈ 1`) and the shell left-edges. Requires a full rectangular, evenly-spaced grid.
"""
function radial_autocorrelation(S::AbstractMatrix, positions; dr::Real = 0.05)
    xs = sort(unique(first.(positions)))
    ys = sort(unique(last.(positions)))
    X, Y = length(xs), length(ys)
    X * Y == length(positions) || throw(ArgumentError("positions do not form a full rectangular grid"))
    dx = X > 1 ? xs[2] - xs[1] : 1.0
    dy = Y > 1 ? ys[2] - ys[1] : 1.0
    (X > 1 && !all(≈(dx), diff(xs))) && throw(ArgumentError("x coordinates are not evenly spaced"))
    (Y > 1 && !all(≈(dy), diff(ys))) && throw(ArgumentError("y coordinates are not evenly spaced"))
    xpos = Dict(x => i for (i, x) in enumerate(xs))
    ypos = Dict(y => j for (j, y) in enumerate(ys))
    ix = [xpos[p[1]] for p in positions]
    iy = [ypos[p[2]] for p in positions]

    # lag-space radial distance grid (fftfreq wrap-around), and the shell each lag belongs to
    lagx = _fftfreq(X) .* (X * dx)
    lagy = _fftfreq(Y) .* (Y * dy)
    dist = [sqrt(lagx[i]^2 + lagy[j]^2) for i in 1:X, j in 1:Y]
    flat_dist = vec(dist)
    maxr = maximum(flat_dist)
    r_bins = collect(0.0:dr:(maxr + dr))
    shell = [searchsortedlast(r_bins, d) for d in flat_dist]        # 1-based shell index (digitize)
    counts = zeros(Int, length(r_bins))
    for s in shell
        (1 ≤ s ≤ length(r_bins)) && (counts[s] += 1)
    end

    nN, Tn = size(S)
    nN == length(positions) || throw(ArgumentError("raster has $nN neurons but $(length(positions)) positions"))
    profiles = Vector{Vector{Float64}}()
    for t in 1:Tn
        grid = zeros(Float64, X, Y)
        @inbounds for k in 1:nN
            grid[ix[k], iy[k]] = S[k, t]
        end
        f0 = grid .- _mean(grid)
        var = _mean(abs2.(f0))
        var > 0 || continue                                         # skip empty / variance-zero frames
        F = _fft2(f0)
        C = real.(_ifft2(abs2.(F))) ./ (X * Y)
        Cn = C ./ C[1, 1]                                           # normalise C(0) = 1
        flatC = vec(Cn)
        sums = zeros(Float64, length(r_bins))
        @inbounds for e in 1:length(flatC)
            s = shell[e]
            (1 ≤ s ≤ length(r_bins)) && (sums[s] += flatC[e])
        end
        push!(profiles, sums ./ max.(counts, 1))
    end
    isempty(profiles) && error("all frames were variance-zero --- nothing to average")
    g_r = [_mean(getindex.(profiles, k)) for k in 1:length(r_bins)]
    return g_r, r_bins
end
function radial_autocorrelation(sol::DewdropSolution; dr::Real = 0.05, of = :all, name = nothing)
    sol.positions === nothing && error("the solution carries no positions --- build with `positions = …`")
    S = _spike_raster(sol; of = of, name = name)
    pos = of === :all ? sol.positions : sol.positions[_subrange(sol.subpops, of)]
    return radial_autocorrelation(S, pos; dr = dr)
end
export radial_autocorrelation
