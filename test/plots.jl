using Dewdrop
using Test
using CairoMakie
using Fathom

set_theme!(fathom())

# Scenario tests double as classical-figure reproductions: each saves its canonical plot to
# test/plots/ (gitignored) for visual inspection, and asserts the artifact was produced. The
# headline classical figures (Brunel regimes, Vogels-Abbott raster) arrive with M2.
const PLOTDIR = joinpath(@__DIR__, "plots")
isdir(PLOTDIR) || mkpath(PLOTDIR)

@testset "scenario plots (CairoMakie + Fathom)" begin
    @testset "current synapse PSC" begin
        # the post-synaptic current kernel: a delayed unit spike then exponential decay.
        syn = CurrentSynapse(τ = 5.0)
        dt = 0.1
        decay = Dewdrop.synapse_decay(syn, dt)
        buf = Dewdrop.DelayBuffer(Dewdrop.CPU(), Float64, 1, 20)
        Dewdrop.deposit!(buf, 0, 1, 1.0, 10)
        Isyn = 0.0
        trace = Float64[]
        for t in 0:200
            Isyn *= decay
            Isyn += Dewdrop.collect_due!(buf, t)[1]
            push!(trace, Isyn)
        end
        ts = collect(0:200) .* dt

        fig = OnePanel()
        ax = Axis(fig[1, 1]; xlabel = "Time (ms)", ylabel = "Synaptic current (a.u.)",
            title = "Current synapse PSC")
        lines!(ax, ts, trace)
        addlabels!(fig)
        path = joinpath(PLOTDIR, "m1b_synapse_psc.png")
        save(path, fig)
        @test isfile(path)
        @test filesize(path) > 0
    end

    @testset "network raster (input gradient)" begin
        # a population driven by a gradient of input currents fires at a gradient of rates ---
        # the canonical raster + per-neuron rate panel.
        N = 60
        m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
        Iext = collect(range(0.22, 0.5; length = N))   # all supra-rheobase (rheobase = 0.2)
        dt, tend = 0.1, 500.0
        sol = solve(DewdropNetwork(m, N; input = Iext, tspan = (0.0, tend)), FixedStep(dt); record_spikes = true)
        times, ids = raster(sol)
        counts = vec(sum(sol.spikes; dims = 2))
        @test counts[end] > counts[1]                  # more drive → higher rate

        fig = TwoPanel()
        ax1 = Axis(fig[1, 1]; xlabel = "Time (ms)", ylabel = "Neuron", title = "Raster")
        scatter!(ax1, times, ids; markersize = 3)
        ax2 = Axis(fig[1, 2]; xlabel = "Neuron", ylabel = "Firing rate (Hz)", title = "Rate")
        lines!(ax2, 1:N, 1000 .* firing_rate(sol))     # 1/ms → Hz
        addlabels!(fig)
        path = joinpath(PLOTDIR, "m1d_network_raster.png")
        save(path, fig)
        @test isfile(path)
        @test filesize(path) > 0
    end
end
