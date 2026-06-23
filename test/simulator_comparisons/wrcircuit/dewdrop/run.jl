#! /bin/bash
#=
exec julia +1.12 -t auto --project="$(dirname "${BASH_SOURCE[0]}")/../../.." "${BASH_SOURCE[0]}" "$@"
=#
# Dewdrop reproduction of the BrainPy WRCircuit reference (../brainpy/out). Ingests the exported
# connectomes / weights / delays / neuron params / initial V / external Poisson raster, re-expresses
# the network in native Dewdrop (FNSNeuron + FrozenDualExpSynapse + a replayed PrescribedCOBA external
# drive via `wrcircuit`), runs it, and writes the matching traces/spikes to ./out for ../compare.jl.
using Dewdrop
using DelimitedFiles

const HERE = @__DIR__
const REF = joinpath(HERE, "..", "brainpy", "out")
const OUT = joinpath(HERE, "out");
mkpath(OUT)

# Delay-convention adjustment (steps), applied to EVERY synaptic delay (recurrent + external). BrainPy's
# delay D makes the postsynaptic conductance onset at spike+D; Dewdrop's ring buffer + dual-exp onset at
# spike+D+1, so reproducing BrainPy needs delay D−1. Calibrated to −1 (verified: it makes the externally
# driven pre-recurrent trajectory bit-identical); overridable as the first CLI arg.
const DELAY_ADJ = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : -1

# --- read the flat-CSV metadata ----------------------------------------------------------------
readkv(f) = Dict(string(r[1]) => r[2] for r in eachrow(readdlm(joinpath(REF, f), ',')))
scal = readkv("scalars.csv")
nE = readkv("neuronE.csv")
nI = readkv("neuronI.csv")
projrows = readdlm(joinpath(REF, "projmeta.csv"), ','; skipstart = 1)
PROJ = Dict(string(r[1]) => (pre = string(r[2]), post = string(r[3]), τr = Float64(r[4]),
    τd = Float64(r[5]), Erev = Float64(r[6]), delay = Int(r[7])) for r in eachrow(projrows))

const DT = Float64(scal["dt"])
const T = Float64(scal["T"])
const NSTEPS = Int(scal["nsteps"])
const NE = Int(scal["NE"])
const NI = Int(scal["NI"])
const N = NE + NI

# --- neuron models (E adapts, I does not → wrcircuit merges into a Heterogeneous FNS) -----------
fns(p) = FNSNeuron(; C = p["C"], gL = p["g_L"], VL = p["V_L"], VK = p["V_K"], Vθ = p["V_th"],
    Vr = p["V_rt"], tref = p["tau_ref"], τK = p["tau_K"], ΔgK = p["Delta_g_K"])
E = fns(nE)
I = fns(nI)

# --- ingest a recurrent connectome → a global-indexed SparseCSR --------------------------------
# edge CSV rows are (pre_local, post_local, weight), 0-based, in the pre/post populations' local index.
poff(pop) = pop == "E" ? 0 : NE                 # global offset of a population's first neuron
function load_proj(name)
    p = PROJ[name]
    raw = readdlm(joinpath(REF, "edges_$(name).csv"), ',')
    preoff, postoff = poff(p.pre), poff(p.post)
    edges = [(Int(raw[k, 1]) + preoff + 1, Int(raw[k, 2]) + postoff + 1, Float64(raw[k, 3]), p.delay + DELAY_ADJ)
             for k in axes(raw, 1)]
    conn = Dewdrop.SparseCSR(Dewdrop.CPU(), edges; npre = N, npost = N)
    syn = FrozenDualExpSynapse(; τr = p.τr, τd = p.τd, Erev = p.Erev)
    return Projection(syn, conn)
end
recurrent = [load_proj(nm) for nm in ("E2E", "E2I", "I2E", "I2I")]

# --- external Poisson drive → a replayed per-target conductance matrix --------------------------
# Replays the dual-exponential COBA conductance each external spike produces on its targets, using the
# SAME deliver→read→decay recurrence as Dewdrop's recurrent synapses (so external and recurrent are
# mutually consistent): a spike at step s reaches its targets at step s+delay+DELAY_ADJ; the column
# used at step n is a·(g_decay − g_rise) BEFORE that step's deposit (the deposit cancels in the
# difference). Erev = 0, so this is the excitatory external drive into E and I.
function external_conductance(name, Npost, postoff)
    p = PROJ[name]
    a = Dewdrop._dualexp_a(p.τr, p.τd)
    decay_r, decay_d = exp(-DT / p.τr), exp(-DT / p.τd)
    raw = readdlm(joinpath(REF, "edges_$(name).csv"), ',')         # (ext_id, post_local, weight)
    outedges = Dict{Int, Vector{Tuple{Int, Float64}}}()            # ext_id → [(post_local, w)]
    for k in axes(raw, 1)
        push!(get!(outedges, Int(raw[k, 1]), Tuple{Int, Float64}[]), (Int(raw[k, 2]), Float64(raw[k, 3])))
    end
    spk = readdlm(joinpath(REF, "ext_spikes.csv"), ',')            # (step, ext_id), 0-based
    deliveries = [Tuple{Int, Float64}[] for _ in 1:(NSTEPS + 1)]
    for k in axes(spk, 1)
        s, i = Int(spk[k, 1]), Int(spk[k, 2])
        haskey(outedges, i) || continue
        dstep = s + p.delay + DELAY_ADJ
        (0 ≤ dstep ≤ NSTEPS) || continue
        for (j, w) in outedges[i]
            push!(deliveries[dstep + 1], (j, w))
        end
    end
    gr = zeros(Npost);
    gd = zeros(Npost)
    G = zeros(N, NSTEPS)                                            # full-network matrix (rows for this pop)
    for n in 0:(NSTEPS - 1)
        for j in 1:Npost
            G[postoff + j, n + 1] = a * (gd[j] - gr[j])            # used at step n (before this step's deposit)
        end
        for (j, w) in deliveries[n + 1]
            gr[j + 1] += w;
            gd[j + 1] += w
        end
        gr .*= decay_r;
        gd .*= decay_d
    end
    return G
end
gext = external_conductance("ext2E", NE, 0) .+ external_conductance("ext2I", NI, NE)

# --- positions (for the spatial subpopulation metadata) ----------------------------------------
loadpos(f) = [tuple(Float64.(r)...) for r in eachrow(readdlm(joinpath(REF, f), ','))]
positions = vcat(loadpos("posE.csv"), loadpos("posI.csv"))

# --- assemble + run ----------------------------------------------------------------------------
prob = wrcircuit(; NE = NE, NI = NI, E = E, I = I, projections = recurrent, gext = gext,
    positions = positions, tspan = (0.0, T))

v0 = vcat(vec(readdlm(joinpath(REF, "v0E.csv"), ',')), vec(readdlm(joinpath(REF, "v0I.csv"), ',')))
sol = solve(prob, FixedStep(DT); v0 = v0,
    record = (v = Trace(:V), gk = Trace(:w), spikes = Spikes()))

# --- write the matching outputs (transpose to (nsteps × N) to match the BrainPy CSVs) ----------
Vmat = permutedims(sol.record.v.data)                  # (nsteps × N)
gkmat = permutedims(sol.record.gk.data)
writedlm(joinpath(OUT, "vE.csv"), Vmat[:, 1:NE], ',')
writedlm(joinpath(OUT, "vI.csv"), Vmat[:, (NE + 1):N], ',')
writedlm(joinpath(OUT, "gKE.csv"), gkmat[:, 1:NE], ',')

t, ids = raster(sol)                                    # global ids (1-based)
sE = [(round(Int, t[k] / DT) - 1, ids[k] - 1) for k in eachindex(t) if ids[k] ≤ NE]
sI = [(round(Int, t[k] / DT) - 1, ids[k] - NE - 1) for k in eachindex(t) if ids[k] > NE]
writedlm(joinpath(OUT, "spikesE.csv"), reduce(vcat, ([s[1] s[2]] for s in sE); init = Matrix{Int}(undef, 0, 2)), ',')
writedlm(joinpath(OUT, "spikesI.csv"), reduce(vcat, ([s[1] s[2]] for s in sI); init = Matrix{Int}(undef, 0, 2)), ',')

rE = sum(sol.spike_count[1:NE]) / (NE * T / 1000)
rI = sum(sol.spike_count[(NE + 1):N]) / (NI * T / 1000)
println("Dewdrop firing rates: E=$(round(rE, digits=2)) Hz, I=$(round(rI, digits=2)) Hz  " *
        "(E spikes=$(sum(sol.spike_count[1:NE])), I spikes=$(sum(sol.spike_count[(NE+1):N])))")
println("DELAY_ADJ=$DELAY_ADJ; wrote Dewdrop outputs to $OUT")

# --- optional timing (WRC_BENCH=1): pure solve wall time, no recording, after a warmup ----------
if get(ENV, "WRC_BENCH", "") == "1"
    nedges = sum(Dewdrop.nedges(p.conn) for p in prob.projections)
    solve(prob, FixedStep(DT))                              # warmup (compile)
    wall = @elapsed solve(prob, FixedStep(DT))
    println("BENCH dewdrop N=$N nedges=$nedges wall=$(round(wall, digits=4))")
end
