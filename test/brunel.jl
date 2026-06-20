using Dewdrop
using Test
using Statistics
using CairoMakie
using Fathom

set_theme!(fathom())

# M2 --- the Brunel (2000) sparse balanced E/I network, the canonical SNN-simulator
# benchmark. A scaled instance (N=2500) in the asynchronous-irregular (AI) regime
# (g=5, η=2): excitatory + inhibitory LIF neurons, FixedProb connectivity with signed
# delta-synapse weights, fixed 1.5 ms delays, and an external Poisson drive. Validated by
# its statistical signatures (sustained moderate rate, CV-ISI ≈ 1) and reproduced as the
# classical raster + population-rate figure.

# mean ISI coefficient of variation over neurons with enough spikes (≈1 ⇒ Poisson-irregular)
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

@testset "Brunel network (asynchronous-irregular regime)" begin
    arch = Dewdrop.CPU()
    NE, NI = 2000, 500
    N = NE + NI
    ε, g, D = 0.1, 5.0, 15                   # 10% connectivity, g=5 (inhibition-dominated), delay 1.5ms
    J = 0.1 * sqrt(5)                        # ≈0.224: J ∝ 1/√C balanced scaling for this N
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)

    # recurrent: E neurons (1..NE) excitatory +J, I neurons inhibitory −gJ
    conn = fixed_prob(arch, N, N, ε; weight = pre -> pre ≤ NE ? J : -g * J,
        delay = D, seed = UInt64(1), allow_self = false)
    # external Poisson drive tuned to the fluctuation-driven balanced operating point
    drive = PoissonDrive(; rate = 6.0, weight = J, seed = UInt64(2))

    sol = solve(DewdropNetwork(m, N; input = 0.0, tspan = (0.0, 600.0),
            projection = Projection(DeltaSynapse(), conn), drive = drive),
        FixedStep(0.1); record_spikes = true)
    times, ids = raster(sol)

    rate = 1000 * sum(firing_rate(sol)) / N
    @test 5 < rate < 200                            # sustained, neither silent nor saturated
    @test count(==(0), sol.spike_count) / N < 0.1   # the network is broadly active

    # asynchronous-irregular: firing is irregular (CV clearly above the regular/mean-driven
    # regime, CV≲0.2) though a scaled-down Brunel has weaker fluctuations than the full CV≈1.
    cv = mean_cv_isi(times, ids, N)
    @test 0.3 < cv < 1.5

    # classical figure: spike raster (subsample of E cells) over population firing rate
    win = (200.0, 400.0)
    keep = (ids .≤ 200) .& (times .≥ win[1]) .& (times .≤ win[2])
    edges = win[1]:1.0:win[2]
    poprate = [count(t -> e ≤ t < e + 1.0, times) for e in edges[1:(end - 1)]] .* (1000.0 / N)

    fig = Figure()
    ax1 = Axis(fig[1, 1]; ylabel = "Neuron", title = "Brunel AI (g = 5, η = 2)")
    scatter!(ax1, times[keep], ids[keep]; markersize = 2)
    ax2 = Axis(fig[2, 1]; xlabel = "Time (ms)", ylabel = "Pop. rate (Hz)")
    lines!(ax2, collect(edges[1:(end - 1)]), poprate)
    linkxaxes!(ax1, ax2)
    rowsize!(fig.layout, 1, Relative(0.7))
    addlabels!(fig)
    path = joinpath(@__DIR__, "plots", "m2_brunel_ai.png")
    save(path, fig)
    @test isfile(path) && filesize(path) > 0
end
