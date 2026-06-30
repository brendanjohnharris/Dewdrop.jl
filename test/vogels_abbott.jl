using Dewdrop
using Test
using Statistics
using CairoMakie
using Fathom

set_theme!(fathom())

# The Vogels & Abbott (2005) conductance-based (COBA) balanced network, the canonical
# COBA SNN benchmark. 4000 LIF neurons (3200 E / 800 I), 2% connectivity, conductance-based
# E (Erev=0, τ=5ms) and I (Erev=−80, τ=10ms) synapses with the original 6 nS / 67 nS quanta
# (here as g/g_L ratios with R=1), held in the asynchronous-irregular state by a weak
# background drive. The COBA dynamics give the hallmark irregular firing (CV-ISI ≈ 1).

function mean_cv_isi(times, ids, N)
    cvs = Float64[]
    for i in 1:N
        ts = sort(times[ids .== i])
        length(ts) < 4 && continue
        isi = diff(ts)
        m = mean(isi)
        m > 0 && push!(cvs, std(isi) / m)
    end
    return isempty(cvs) ? NaN : mean(cvs)
end

@testset "Vogels–Abbott COBA network (asynchronous-irregular)" begin
    arch = Dewdrop.CPU()
    N, NE, NI, ε = 4000, 3200, 800, 0.02
    m = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 5.0)
    ge, gi, D = 6.0 / 10.0, 67.0 / 10.0, 1        # peak conductances g/g_L; delay 0.1 ms

    ce = fixed_prob(arch, N, N, ε; weight = ge, delay = steps(D), seed = UInt64(1), sources = 1:NE, allow_self = false)
    ci = fixed_prob(arch, N, N, ε; weight = gi, delay = steps(D), seed = UInt64(2), sources = (NE + 1):N, allow_self = false)
    projs = (Projection(ConductanceSynapse(τ = 5.0, Erev = 0.0), ce),
        Projection(ConductanceSynapse(τ = 10.0, Erev = -80.0), ci))
    drive = PoissonDrive(; rate = 6.0, weight = 0.1, seed = UInt64(7))

    sol = solve(DewdropNetwork(m, N; input = 0.0, tspan = (0.0, 500.0), projections = projs, drive = drive),
        FixedStep(0.1); record = (spikes = Spikes(),))
    times, ids = raster(sol)

    rate = 1000 * sum(firing_rate(sol)) / N
    @test 2 < rate < 100                          # sustained, few-to-tens of Hz
    cv = mean_cv_isi(times, ids, N)
    @test cv > 0.8                                # conductance-based AI: irregular firing (CV ≈ 1)

    # classical figure: spike raster over population firing rate
    win = (200.0, 400.0)
    keep = (ids .≤ 200) .& (times .≥ win[1]) .& (times .≤ win[2])
    edges = win[1]:1.0:win[2]
    poprate = [count(t -> e ≤ t < e + 1.0, times) for e in edges[1:(end - 1)]] .* (1000.0 / N)

    fig = Figure()
    ax1 = Axis(fig[1, 1]; ylabel = "Neuron", title = "Vogels–Abbott COBA (CV ≈ $(round(cv; digits = 2)))")
    scatter!(ax1, times[keep], ids[keep]; markersize = 2)
    ax2 = Axis(fig[2, 1]; xlabel = "Time (ms)", ylabel = "Pop. rate (Hz)")
    lines!(ax2, collect(edges[1:(end - 1)]), poprate)
    linkxaxes!(ax1, ax2)
    rowsize!(fig.layout, 1, Relative(0.7))
    addlabels!(fig)
    path = joinpath(@__DIR__, "plots", "m2_vogels_abbott.png")
    save(path, fig)
    @test isfile(path) && filesize(path) > 0
end
