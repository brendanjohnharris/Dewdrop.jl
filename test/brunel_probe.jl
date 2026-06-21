#! /usr/bin/env julia
# Parameter-sweep probe for the Brunel (2000) regimes. Runs scaled Brunel instances and, for
# each, prints a single JSON `RESULT {...}` line of statistics + saves a 3-panel diagnostic
# figure (raster / population rate / spectrum). Used by the regime parameter sweep; NOT part of
# the test suite (runtests.jl includes files explicitly).
#
# Single run:
#   julia --project=test test/brunel_probe.jl NE=2000 NI=500 g=5 J=0.224 \
#       drive_rate=6 drive_weight=0.224 D=15 T=1000 transient=200 tag=ai seed=1
#
# Batch (one Julia load, many solves --- amortises startup):
#   julia --project=test test/brunel_probe.jl batch=/tmp/cands.txt
#   where each non-blank line of cands.txt is one run's `key=value key=value ...` arg list.
#
# Builds the network through Dewdrop's builder API (dogfooding it): two delta-synapse
# projections (E with +J, I with -g*J) plus a Poisson background drive.

using Dewdrop
using CairoMakie
using Fathom
include(joinpath(@__DIR__, "brunel_analysis.jl"))

set_theme!(fathom())

const DEFAULTS = Dict{String, Float64}(
    "NE" => 2000, "NI" => 500, "g" => 5.0, "J" => 0.1 * sqrt(5),
    "drive_rate" => 6.0, "drive_weight" => 0.1 * sqrt(5), "D" => 15,
    "T" => 1000.0, "transient" => 200.0, "seed" => 1, "dt" => 0.1,
    "v0lo" => 0.0, "v0hi" => 0.0,   # if v0hi>v0lo, randomise initial V uniformly in [v0lo,v0hi)
)

function parse_args(tokens)
    P = copy(DEFAULTS)
    tag = "probe"
    for a in tokens
        isempty(strip(a)) && continue
        k, v = split(a, "="; limit = 2)
        if k == "tag"
            tag = v
        else
            P[k] = parse(Float64, v)
        end
    end
    return P, tag
end

js(x) = isfinite(x) ? string(round(x; digits = 4)) : "null"

function probe(P, tag; savefig = true)
    NE, NI = Int(P["NE"]), Int(P["NI"])
    N = NE + NI
    g, J, D = P["g"], P["J"], Int(P["D"])
    T, transient, dt = P["T"], P["transient"], P["dt"]
    seed = UInt64(Int(P["seed"]))

    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    nb = network(m, NE, NI; arch = Dewdrop.CPU(), tspan = (0.0, T))
    project!(nb, :E, DeltaSynapse(); p = 0.1, weight = J, delay = D, seed = seed)
    project!(nb, :I, DeltaSynapse(); p = 0.1, weight = -g * J, delay = D, seed = seed + 0x100)
    drive!(nb, PoissonDrive(; rate = P["drive_rate"], weight = P["drive_weight"], seed = seed + 0x200))

    v0 = P["v0hi"] > P["v0lo"] ? (P["v0lo"], P["v0hi"]) : nothing
    t_run = @elapsed sol = solve(build(nb), FixedStep(dt); record = (spikes = Spikes(),), v0 = v0)
    times, ids = raster(sol)

    t0, t1, Δ = transient, T, 1.0
    rate = mean_rate(times, t0, t1, N)
    cv = mean_cv_isi(times, ids, N)
    _, A = pop_activity(times, t0, t1, Δ)
    freqs, power = pop_spectrum(A, Δ)
    fpk, prom = dominant_peak(freqs, power)
    sync = synchrony_index(A)
    frac_silent = count(==(0), sol.spike_count) / N

    println("RESULT {",
        "\"tag\":\"", tag, "\",\"rate\":", js(rate), ",\"cv\":", js(cv),
        ",\"fpeak\":", js(fpk), ",\"prominence\":", js(prom), ",\"sync\":", js(sync),
        ",\"frac_silent\":", js(frac_silent), ",\"runtime_s\":", js(t_run),
        ",\"NE\":", NE, ",\"NI\":", NI, ",\"g\":", js(g), ",\"J\":", js(J),
        ",\"drive_rate\":", js(P["drive_rate"]), ",\"drive_weight\":", js(P["drive_weight"]),
        ",\"D\":", D, ",\"T\":", js(T), "}")

    savefig || return nothing
    win = (max(t0, T - 300.0), T)
    keep = (ids .≤ 200) .& (times .≥ win[1]) .& (times .≤ win[2])
    cwin, Awin = pop_activity(times, win[1], win[2], Δ)
    Arate = Awin .* (1000.0 / N / Δ)
    fig = Figure(; size = (700, 720))
    ax1 = Axis(fig[1, 1]; ylabel = "Neuron",
        title = "Brunel $(tag): $(round(rate; digits = 1)) Hz, CV $(round(cv; digits = 2)), f $(round(fpk; digits = 0)) Hz (prom $(round(prom; digits = 1)))")
    scatter!(ax1, times[keep], ids[keep]; markersize = 2)
    ax2 = Axis(fig[2, 1]; xlabel = "Time (ms)", ylabel = "Pop. rate (Hz)")
    lines!(ax2, cwin, Arate)
    linkxaxes!(ax1, ax2)
    ax3 = Axis(fig[3, 1]; xlabel = "Frequency (Hz)", ylabel = "Power")
    lines!(ax3, freqs, power)
    scatter!(ax3, [fpk], [power[argmax(power)]]; color = Fathom.bermejo, markersize = 8)
    rowsize!(fig.layout, 1, Relative(0.5))
    path = joinpath(@__DIR__, "plots", "_probe_$(tag).png")
    mkpath(dirname(path))
    save(path, fig)
    println("FIGURE ", path)
    return nothing
end

# ---- entry point: batch=<file> runs many, otherwise one run from ARGS -------------------------
let batchidx = findfirst(a -> startswith(a, "batch="), ARGS)
    if batchidx !== nothing
        file = split(ARGS[batchidx], "="; limit = 2)[2]
        for line in eachline(file)
            isempty(strip(line)) && continue
            P, tag = parse_args(split(line))
            try
                probe(P, tag)
            catch err
                println("RESULT {\"tag\":\"", tag, "\",\"error\":\"", err, "\"}")
            end
        end
    else
        P, tag = parse_args(ARGS)
        probe(P, tag)
    end
end
