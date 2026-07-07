#!/usr/bin/env julia
# Generates the figures for docs/src/guide/tour.md. Run from the test environment (has CairoMakie +
# Fathom + ForwardDiff):  julia --project=test docs/tour_assets.jl
# Saves PNGs into docs/src/assets/tour/. The tour.md code blocks mirror this script.

using Dewdrop
using ForwardDiff
using Statistics
using CairoMakie
using Fathom
set_theme!(fathom())

const ASSETS = joinpath(@__DIR__, "src", "assets", "tour")
mkpath(ASSETS)
Dewdrop.set_advice!(false)

# ───────────────────────────── 1. Construct + simulate + record ─────────────────────────────
# A balanced conductance-based (COBA) E/I network via the fluent builder, in the asynchronous-irregular
# regime (Vogels–Abbott style).
NE, NI = 320, 80
E = LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 5.0)
nb = network(; tspan = (0.0, 600.0))
population!(nb, :E, E, NE; input = 0.0)
population!(nb, :I, E, NI; input = 0.0)
project!(nb, :E => :E, ConductanceSynapse(; τ = 5.0, Erev = 0.0);   p = 0.02, weight = 0.9, delay = 1.0, seed = 0x01)
project!(nb, :E => :I, ConductanceSynapse(; τ = 5.0, Erev = 0.0);   p = 0.02, weight = 0.9, delay = 1.0, seed = 0x02)
project!(nb, :I => :E, ConductanceSynapse(; τ = 10.0, Erev = -80.0); p = 0.02, weight = 6.7, delay = 1.0, seed = 0x03)
project!(nb, :I => :I, ConductanceSynapse(; τ = 10.0, Erev = -80.0); p = 0.02, weight = 6.7, delay = 1.0, seed = 0x04)
drive!(nb, :all, PoissonDrive(; rate = 6.0, weight = 0.1, seed = 0x07))
prob = build(nb)

sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(),))
t, ids = raster(sol)                                  # spike times (ms) and neuron ids
rate_hz = 1000 * mean(firing_rate(sol))               # population mean rate (dt in ms → Hz)
@info "network" NE NI mean_rate_Hz = round(rate_hz; digits = 1)

# population rate over time (5 ms bins)
edges = 0:5.0:600.0
counts = zeros(length(edges) - 1)
for tk in t
    b = clamp(searchsortedlast(edges, tk), 1, length(counts))
    counts[b] += 1
end
poprate = counts ./ ((NE + NI) * step(edges) / 1000)  # Hz per neuron
bincentres = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2

fig1 = TwoPanel()
ax1 = Axis(fig1[1, 1]; ylabel = "Neuron", title = "Asynchronous-irregular COBA network")
isE = ids .<= NE
scatter!(ax1, t[isE] ./ 1000, ids[isE]; color = Fathom.baikal, markersize = 2)     # E blue
scatter!(ax1, t[.!isE] ./ 1000, ids[.!isE]; color = Fathom.bermejo, markersize = 2) # I red
hidexdecorations!(ax1; grid = false)
ax2 = Axis(fig1[2, 1]; xlabel = "Time (s)", ylabel = "Population rate (Hz)")
lines!(ax2, bincentres ./ 1000, poprate; color = Fathom.ianthina)
rowsize!(fig1.layout, 1, Relative(0.66))
addlabels!(fig1)
save(joinpath(ASSETS, "raster.png"), fig1; px_per_unit = 2)

# ───────────────────────────── 2. Batching: a parameter sweep ─────────────────────────────
# Co-execute B members over ONE shared connectome, sweeping a SYNAPSE parameter per member (the excitatory
# synaptic time constant τ). This is a single fused (N,B) solve — physical parameters in, no boilerplate.
τ_sweep = collect(range(2.0, 12.0, length = 12))
Bs = length(τ_sweep)
base = build(let nb = network(; tspan = (0.0, 400.0))
    population!(nb, :E, E, 200; input = 0.0)
    project!(nb, :E => :E, ConductanceSynapse(; τ = τ_sweep[1], Erev = 0.0); p = 0.05, weight = 1.2, delay = 1.0, seed = 0x11)
    drive!(nb, :all, PoissonDrive(; rate = 12.0, weight = 0.12, seed = 0x17))
    nb
end)
bs = solve(base, FixedStep(0.1); batch = Bs, syn_overrides = Dict(1 => (; τ = τ_sweep)))
sweep_rate = [1000 * mean(bs.spike_count[:, m]) / (bs.nsteps * bs.dt) for m in 1:Bs]

fig2 = OnePanel()
ax = Axis(fig2[1, 1]; xlabel = "Excitatory synaptic τ (ms)", ylabel = "Population rate (Hz)",
    title = "Batched τ sweep ($(Bs) members)")
scatterlines!(ax, τ_sweep, sweep_rate; color = Fathom.qinghai, markersize = 10)
save(joinpath(ASSETS, "batch_sweep.png"), fig2; px_per_unit = 2)

# ───────────────────────────── 3. Training: connected surrogate-gradient descent ─────────────────────────────
# Train the recurrent synaptic weight of a CONNECTED network, by gradient descent, to raise the population
# rate to a target. The Differentiable backend flows gradients through the surrogate-weighted scatter to the
# weight. Recurrent excitation (a delta synapse near threshold) gives the weight strong leverage over the rate.
function netrate(; w, N = 40, β = 25.0, T = typeof(w))
    m = LIF(; τ = T(20.0), EL = T(0.0), Vθ = T(20.0), Vr = T(0.0), R = T(1.0), tref = T(0.0))
    conn = fixed_prob(CPU(), N, N, 0.3; weight = T(w), delay = steps(1), seed = UInt64(1), allow_self = false)
    p = DewdropNetwork(m, N; input = T(22.0), tspan = (0.0, 250.0),
        projections = (Projection(DeltaSynapse(), conn),))
    s = solve(p, FixedStep(0.1); backend = Differentiable(; β = β))
    return 1000 * sum(s.spike_count) / (N * Dewdrop.duration(s))     # population rate (Hz)
end

target = 18.0                                                       # raise the rate to this target (Hz)
w, η = 0.5, 3.0e-4
hist = Float64[]
for _ in 1:40
    push!(hist, netrate(; w = w))
    g = ForwardDiff.derivative(ww -> (netrate(; w = ww) - target)^2, w)
    global w = clamp(w - η * g, 0.0, 1.4)                           # clamp keeps w in the well-behaved region
end
push!(hist, netrate(; w = w))
@info "training" target reached = round(hist[end]; digits = 2) w_start = 0.5 w_final = round(w; digits = 4)

fig3 = OnePanel()
ax = Axis(fig3[1, 1]; xlabel = "Gradient-descent step", ylabel = "Population rate (Hz)",
    title = "Connected surrogate-gradient training")
lines!(ax, 0:(length(hist) - 1), hist; color = Fathom.seohae, label = "rate")
hlines!(ax, [target]; color = Fathom.chernoe, linestyle = :dash, label = "target")
axislegend(ax; position = :rt)
save(joinpath(ASSETS, "training.png"), fig3; px_per_unit = 2)

println("TOUR ASSETS OK: ", readdir(ASSETS))
