#! /bin/bash
#=
exec julia +1.12 --project="$(dirname "${BASH_SOURCE[0]}")/../.." "${BASH_SOURCE[0]}" "$@"
=#
# Compare the Dewdrop WRCircuit reproduction (dewdrop/out) against the BrainPy reference (brainpy/out):
# membrane traces, spike times, firing rates, adaptation conductance, and the reconstructed synaptic
# input current. Prints a metrics table and (unless `metrics-only`) writes a Fathom comparison figure.
# Defines `compare_wrcircuit(; plot)` returning the metrics NamedTuple (used by the permanent test).
using DelimitedFiles
using Statistics

const HERE = @__DIR__
const BP = joinpath(HERE, "brainpy", "out")
const DD = joinpath(HERE, "dewdrop", "out")

readmat(dir, f) = readdlm(joinpath(dir, f), ',')
function read_raster(dir, f)                         # → Set{(step, id)} 0-based
    isfile(joinpath(dir, f)) || return Set{Tuple{Int, Int}}()
    raw = readmat(dir, f)
    isempty(raw) && return Set{Tuple{Int, Int}}()
    return Set((Int(raw[k, 1]), Int(raw[k, 2])) for k in axes(raw, 1))
end

# spike-time agreement: of BrainPy's spikes, fraction with a same-neuron Dewdrop spike within ±tol steps.
function spikematch(bp::Set, dd::Set, tol::Int)
    tol == 0 && return length(intersect(bp, dd))
    return count(p -> any(q -> q[2] == p[2] && abs(q[1] - p[1]) ≤ tol, dd), bp)
end

function metrics(name, vbp, vdd, sbp, sdd, T)
    n = min(size(vbp, 1), size(vdd, 1))
    err = abs.(@view(vbp[1:n, :]) .- @view(vdd[1:n, :]))
    nrn = size(vbp, 2)
    return (name = name, nspk_bp = length(sbp), nspk_dd = length(sdd),
        rate_bp = length(sbp) / (nrn * T / 1000), rate_dd = length(sdd) / (nrn * T / 1000),
        vmean = mean(err), vmax = maximum(err), vcor = cor(vec(@view vbp[1:n, :]), vec(@view vdd[1:n, :])),
        exact = spikematch(sbp, sdd, 0), within1 = spikematch(sbp, sdd, 1), tot = length(sbp))
end

function compare_wrcircuit(; plot::Bool = true)
    scal = Dict(string(r[1]) => r[2] for r in eachrow(readmat(BP, "scalars.csv")))
    T = Float64(scal["T"]);
    dt = Float64(scal["dt"]);
    NE = Int(scal["NE"]);
    NI = Int(scal["NI"]);
    nsteps = Int(scal["nsteps"])

    vEbp = readmat(BP, "vE.csv");
    vEdd = readmat(DD, "vE.csv")
    vIbp = readmat(BP, "vI.csv");
    vIdd = readmat(DD, "vI.csv")
    sEbp = read_raster(BP, "spikesE.csv");
    sEdd = read_raster(DD, "spikesE.csv")
    sIbp = read_raster(BP, "spikesI.csv");
    sIdd = read_raster(DD, "spikesI.csv")

    mE = metrics("E", vEbp, vEdd, sEbp, sEdd, T)
    mI = metrics("I", vIbp, vIdd, sIbp, sIdd, T)

    println("\n  WRCircuit: Dewdrop vs BrainPy (N = $(NE+NI), $(round(T))ms, dt=$dt)\n")
    println(rpad("pop", 5), rpad("spikes bp/dd", 16), rpad("rate bp/dd (Hz)", 20),
        rpad("mean|ΔV| (mV)", 16), rpad("V cor", 10), "spike-time match (exact / ±1)")
    for m in (mE, mI)
        println(rpad(m.name, 5), rpad("$(m.nspk_bp) / $(m.nspk_dd)", 16),
            rpad("$(round(m.rate_bp,digits=2)) / $(round(m.rate_dd,digits=2))", 20),
            rpad(string(round(m.vmean, digits = 5)), 16), rpad(string(round(m.vcor, digits = 6)), 10),
            "$(m.exact)/$(m.tot)  ($(m.within1)/$(m.tot) within ±1)")
    end

    if plot
        make_figure(vEbp, vEdd, vIbp, vIdd, sEbp, sEdd, sIbp, sIdd, NE, NI, nsteps, dt, T)
    end
    return (E = mE, I = mI, NE = NE, NI = NI, T = T)
end

function make_figure(args...)
    @eval Main begin
        using CairoMakie
        using Fathom
    end
    Base.invokelatest(_make_figure, args...)         # run in the post-`using` world
end

function _make_figure(vEbp, vEdd, vIbp, vIdd, sEbp, sEdd, sIbp, sIdd, NE, NI, nsteps, dt, T)
    CairoMakie = Main.CairoMakie                       # resolved here (new world via invokelatest)
    Fathom = Main.Fathom
    CairoMakie.set_theme!(Fathom.fathom())
    bpc = Fathom.colororder[1]      # baikal (blue)  = BrainPy
    ddc = Fathom.colororder[2]      # bermejo (red)  = Dewdrop
    n = min(size(vEbp, 1), size(vEdd, 1))
    ts = ((1:n) .* dt)
    fig = Fathom.SixPanel()

    # (1) one E membrane trace, overlaid
    j = argmax(vec(sum(abs.(diff(vEbp; dims = 1)); dims = 1)))   # a lively neuron
    ax1 = CairoMakie.Axis(fig[1, 1]; xlabel = "Time (ms)", ylabel = "Membrane potential (mV)",
        title = "Excitatory trace (neuron $j)")
    CairoMakie.lines!(ax1, ts, vEbp[1:n, j]; color = bpc, label = "BrainPy")
    CairoMakie.lines!(ax1, ts, vEdd[1:n, j]; color = ddc, linestyle = :dash, label = "Dewdrop")
    CairoMakie.axislegend(ax1; position = :rb, framevisible = false)

    # (2) E raster overlay (BrainPy ● vs Dewdrop ×)
    ax2 = CairoMakie.Axis(fig[1, 2]; xlabel = "Time (ms)", ylabel = "Excitatory neuron",
        title = "Excitatory raster")
    rb = collect(sEbp);
    rd = collect(sEdd)
    CairoMakie.scatter!(ax2, [p[1] * dt for p in rb], [p[2] for p in rb]; color = bpc, markersize = 6)
    CairoMakie.scatter!(ax2, [p[1] * dt for p in rd], [p[2] for p in rd]; color = ddc, markersize = 4, marker = :xcross)

    # (3) mean |ΔV| over time
    ax3 = CairoMakie.Axis(fig[1, 3]; xlabel = "Time (ms)", ylabel = "Mean |ΔV| (mV)",
        title = "Membrane error", yscale = log10)
    e = vec(mean(abs.(vEbp[1:n, :] .- vEdd[1:n, :]); dims = 2))
    CairoMakie.lines!(ax3, ts, max.(e, 1e-8); color = Fathom.colororder[3])

    # (4) per-neuron firing-rate scatter (E and I)
    ax4 = CairoMakie.Axis(fig[2, 1]; xlabel = "BrainPy rate (Hz)", ylabel = "Dewdrop rate (Hz)",
        title = "Per-neuron firing rate")
    rateE_bp = [count(p -> p[2] == k, sEbp) for k in 0:(NE - 1)] ./ (T / 1000)
    rateE_dd = [count(p -> p[2] == k, sEdd) for k in 0:(NE - 1)] ./ (T / 1000)
    rmax = max(maximum(rateE_bp; init = 1.0), maximum(rateE_dd; init = 1.0))
    CairoMakie.lines!(ax4, [0, rmax], [0, rmax]; color = Fathom.colororder[6], linestyle = :dash)
    CairoMakie.scatter!(ax4, rateE_bp, rateE_dd; color = bpc, markersize = 7)

    # (5) adaptation conductance gK (one E neuron)
    ax5 = CairoMakie.Axis(fig[2, 2]; xlabel = "Time (ms)", ylabel = "Adaptation gK (µS)",
        title = "Adaptation conductance")
    gkbp = readmat(BP, "gKE.csv");
    gkdd = readmat(DD, "gKE.csv")
    ng = min(size(gkbp, 1), size(gkdd, 1))
    CairoMakie.lines!(ax5, (1:ng) .* dt, gkbp[1:ng, j]; color = bpc, label = "BrainPy")
    CairoMakie.lines!(ax5, (1:ng) .* dt, gkdd[1:ng, j]; color = ddc, linestyle = :dash, label = "Dewdrop")
    CairoMakie.axislegend(ax5; position = :rt, framevisible = false)

    # (6) spike-time difference distribution (ziggurat preferred over hist)
    ax6 = CairoMakie.Axis(fig[2, 3]; xlabel = "Spike-time difference (steps)", ylabel = "Count",
        title = "Spike-time agreement")
    diffs = Int[]
    for (s, id) in sEbp
        cand = [q[1] - s for q in sEdd if q[2] == id]
        isempty(cand) || push!(diffs, cand[argmin(abs.(cand))])
    end
    if !isempty(diffs)
        Fathom.ziggurat!(ax6, diffs; bins = (minimum(diffs) - 0.5):1:(maximum(diffs) + 0.5), color = ddc)
    end

    Fathom.addlabels!(fig)
    out = joinpath(HERE, "wrcircuit_comparison.png")
    CairoMakie.save(out, fig; px_per_unit = 2)
    CairoMakie.save(replace(out, ".png" => ".pdf"), fig)
    println("\n  wrote figure → $out")
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_wrcircuit(; plot = !("metrics-only" in ARGS))
end
