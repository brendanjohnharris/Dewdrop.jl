#! /bin/bash
#=
exec julia +1.12 --project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/test" "${BASH_SOURCE[0]}" "$@"
=#
# Aggregate + compare every simulator's output. MODULAR: it discovers each subdirectory that contains
# an `out/performance.csv` (dewdrop/, brian/, brainpy/, … and any future nest/, neuron/), so adding a
# simulator is just dropping in its directory + data: no change here. It (1) VERIFIES every
# simulator ran the same problem (identical connectome `nedges`; statistically-matching rate / CV-ISI),
# and (2) PLOTS the SCALING of wall time and memory vs network size N across CPU and GPU backends.
#
#   ./compare_simulators.jl            # read all <sim>/out, verify + plot to out/comparison.pdf
#
# Each simulator is run separately first (e.g. `dewdrop/run.jl`, `brian/run.py`, `brainpy/run.py`).

using DelimitedFiles
using Printf
using CairoMakie
using Fathom

const HERE = @__DIR__
set_theme!(fathom())

# discover simulator directories (those with out/performance.csv)
function simulator_dirs()
    dirs = String[]
    for d in readdir(HERE; join = true)
        isdir(d) && isfile(joinpath(d, "out", "performance.csv")) && push!(dirs, d)
    end
    return sort(dirs)
end

# load performance (simulator, backend, device, N, wall_s, mem_mb) + values (N, rate, cv, nedges)
struct Series
    sim::String; backend::String; device::String; Ns::Vector{Int}; wall::Vector{Float64}; mem::Vector{Float64}
end

function load_performance()
    series = Dict{Tuple{String, String, String}, Series}()
    for d in simulator_dirs()
        raw = readdlm(joinpath(d, "out", "performance.csv"), ','; skipstart = 1)
        for i in 1:size(raw, 1)
            sim, bk, dev = string(raw[i, 1]), string(raw[i, 2]), string(raw[i, 3])
            N, w, m = Int(raw[i, 4]), Float64(raw[i, 5]), Float64(raw[i, 6])
            key = (sim, bk, dev)
            s = get!(series, key, Series(sim, bk, dev, Int[], Float64[], Float64[]))
            push!(s.Ns, N); push!(s.wall, w); push!(s.mem, m)
        end
    end
    for s in values(series)
        p = sortperm(s.Ns)
        s.Ns .= s.Ns[p]; s.wall .= s.wall[p]; s.mem .= s.mem[p]
    end
    return collect(values(series))
end

function load_values()
    vals = Dict{String, Any}()
    for d in simulator_dirs()
        f = joinpath(d, "out", "values.csv")
        isfile(f) || continue
        raw = readdlm(f, ','; skipstart = 1)
        vals[basename(d)] = (N = Int(raw[1, 1]), rate = Float64(raw[1, 2]), cv = Float64(raw[1, 3]), nedges = Int(raw[1, 4]))
    end
    return vals
end

# (1) verify: same connectome (nedges) + statistically-matching rate / CV-ISI
function verify(vals)
    println("="^66)
    println("Same-simulation check (N = $(isempty(vals) ? "?" : first(values(vals)).N))")
    println("="^66)
    @printf("%-12s %8s %10s %8s\n", "simulator", "nedges", "rate(Hz)", "CV-ISI")
    for (name, v) in sort(collect(vals); by = first)
        @printf("%-12s %8d %10.1f %8.2f\n", name, v.nedges, v.rate, v.cv)
    end
    isempty(vals) && return
    ne = [v.nedges for v in values(vals)]
    rates = [v.rate for v in values(vals)]
    cvs = [v.cv for v in values(vals)]
    same_conn = all(==(ne[1]), ne)
    rate_ok = (maximum(rates) - minimum(rates)) / max(maximum(rates), eps()) < 0.1   # within 10%
    cv_ok = (maximum(cvs) - minimum(cvs)) < 0.1
    println("-"^66)
    println("  identical connectome (nedges): ", same_conn ? "✓" : "✗ DIFFER")
    println("  rates agree (<10% spread):     ", rate_ok ? "✓" : "✗ ($(round(minimum(rates); digits = 1))–$(round(maximum(rates); digits = 1)) Hz)")
    println("  CV-ISI agree (<0.10 spread):   ", cv_ok ? "✓" : "✗")
    println(
        "  → ", (same_conn && rate_ok && cv_ok) ? "all simulators ran the SAME problem." :
            "MISMATCH — inspect the configs."
    )
    return println("="^66)
end

# (2) plot scaling: wall time + memory vs N, log-log, CPU & GPU
const SIMCOLOR = Dict(
    "dewdrop" => Fathom.baikal, "brian" => Fathom.bermejo,
    "brainpy" => Fathom.qinghai, "nest" => Fathom.seohae, "genn" => Fathom.ianthina,
    "snn" => Fathom.mesopelagic
)
_color(sim) = get(SIMCOLOR, sim, Fathom.ianthina)

# dewdrop's backends share its colour, so distinguish them by linestyle. On the CPU axis the BEST
# (fastest) backend, Turbo, is the solid blue line; Fused is dashed and Serial dotted. The GPU axis has a
# single dewdrop line (the Fused megakernel), drawn solid as the primary GPU path. Other sims stay solid.
const DEWSTYLE_CPU = Dict("turbo" => :solid, "fused" => :dash, "serial" => :dot)
function _linestyle(sim, backend, device)
    sim == "dewdrop" || return :solid
    device == "gpu" && return :solid
    return get(DEWSTYLE_CPU, backend, :solid)
end

# Apples-to-apples layout: legend across the top, CPU methods (time + memory) on row 2, GPU on row 3.
# Within a row every series is the same device, so colour = simulator; dewdrop is drawn thicker/opaque to
# stand out (others thinner/translucent), and its multiple CPU backends are further split by linestyle
# (Turbo solid, Fused dashed, Serial dotted). CPU and GPU never share an axis, so their very different
# scales don't squash.
function plot_scaling(series)
    fig = SixPanel()
    cpu = filter(s -> s.device != "gpu", series)
    gpu = filter(s -> s.device == "gpu", series)
    # row 1: CPU legend, row 2: CPU plots, row 3: GPU legend, row 4: GPU plots
    mkaxis(r, c, title, ylab) = Axis(
        fig[r, c]; xlabel = "Network size (N)", ylabel = ylab,
        xscale = log10, yscale = log10, title = title
    )
    axct = mkaxis(2, 1, "CPU: simulation time", "Simulation time (s)")
    axcm = mkaxis(2, 2, "CPU: peak memory", "Peak memory (MB)")
    axgt = mkaxis(4, 1, "GPU: simulation time", "Simulation time (s)")
    axgm = mkaxis(4, 2, "GPU: peak memory", "Peak memory (MB)")
    function draw!(axt, axm, ss)               # returns (handles, labels) for the legend
        handles, labels = Any[], String[]
        for s in sort(ss; by = x -> (x.sim, x.backend))
            length(s.Ns) ≥ 1 || continue
            lbl = s.backend == s.sim ? s.sim : "$(s.sim) $(s.backend)"
            isdew = s.sim == "dewdrop"          # dewdrop emphasised: thicker, fully opaque
            c = isdew ? _color(s.sim) : (_color(s.sim), 0.6)
            lw = isdew ? 7.0 : 3.6
            ls = _linestyle(s.sim, s.backend, s.device)   # dewdrop backends split by linestyle; others solid
            h = lines!(axt, s.Ns, s.wall; color = c, linewidth = lw, linestyle = ls)
            push!(handles, h); push!(labels, lbl)
            pos = s.mem .> 0
            any(pos) && lines!(axm, s.Ns[pos], s.mem[pos]; color = c, linewidth = lw, linestyle = ls)
        end
        return handles, labels
    end
    ch, cl = draw!(axct, axcm, cpu)
    gh, gl = draw!(axgt, axgm, gpu)
    Legend(fig[1, 1:2], ch, cl, "CPU methods"; orientation = :horizontal, nbanks = 2, framevisible = false)
    Legend(fig[3, 1:2], gh, gl, "GPU methods"; orientation = :horizontal, nbanks = 1, framevisible = false)
    addlabels!(fig)
    out = joinpath(HERE, "out", "comparison.pdf")
    mkpath(dirname(out))
    save(out, fig)                                           # vector, for docs
    save(replace(out, ".pdf" => ".png"), fig; px_per_unit = 2)   # raster, for quick preview
    println("wrote scaling plot → $out (+ .png)")
    return fig
end

function main()
    series = load_performance()
    isempty(series) && error("no <sim>/out/performance.csv found in $HERE; run each simulator first")
    verify(load_values())
    return plot_scaling(series)
end

main()
