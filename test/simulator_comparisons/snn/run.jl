#! /bin/bash
#=
exec julia +1.12 --project="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "${BASH_SOURCE[0]}" "$@"
=#
# SpikingNeuralNetworks.jl (JuliaSNN) implementation of the shared simulator-comparison problem
# (../spec.toml). Standalone:
#     ./run.jl                # the (cpu) sweep → out/{values,performance}.csv
#     ./run.jl <N>            # a single throughput config (used internally per subprocess)
#     ./run.jl correctness    # the statistics (rate, CV-ISI) run
# Each N runs in a FRESH subprocess so its peak memory is clean (mirrors dewdrop/run.jl). JuliaSNN has
# no GPU engine and its step loop is single-threaded, so device is always `cpu`; it has its OWN project
# (this directory), independent of Dewdrop's. Everything else comes from ../spec.toml.
#
# The spec maps onto BUILT-IN JuliaSNN types (no custom model code):
#   - AdExParameter: C,gl,El,Vt,Vr,ΔT,τw,a,b straight from the spec (τm=C/gl, R=1/gl auto-derived).
#   - CurrentSynapse: the current-based (CUBA, V-independent) single-exponential synapse, τe=τi=tau.
#     ALL signed edges (E:+GE, I:−GI) route into its `:glu` channel, so its `ge` accumulator IS the
#     spec's single signed current accumulator. JuliaSNN's membrane adds −R·syn_curr = +R·ge, matching
#     the spec's +R·I_syn; the constant drive is the population's `I` field (+R·I_ext in the same eqn).
#   - a fixed 2-step conduction delay via delay_dist = Normal(delay_steps·dt, 0).
#   - PostSpike(At=0, τabs=0): the adaptive threshold stays pinned at Vt, and there is no refractory.
# JuliaSNN fires at a hardwired 0 mV cutoff (the spec's Vpeak is −40 mV), but the AdEx exponential term
# diverges super-exponentially, so the exact cutoff shifts spike times by ≪ dt: the statistics are
# unchanged (cross-validated at N=8000: 37.8 Hz / CV-ISI 0.13, matching NEST/Dewdrop).

import SpikingNeuralNetworks as SNN
import Distributions: Normal
using SparseArrays
using Random
import TOML

const HERE = @__DIR__
const SPEC = TOML.parsefile(joinpath(HERE, "..", "spec.toml"))
const OUT = joinpath(HERE, "out")
# JuliaSNN's step loop is single-threaded; cap the sweep so the largest CPU run stays ~minutes.
const MAXN = 128_000
const REPS = 3

# language-agnostic connectome (splitmix64 fixed in-degree; identical across simulators)
@inline function _splitmix64(s::UInt64)
    s += 0x9e3779b97f4a7c15
    z = s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return (z ⊻ (z >> 31), s)
end
function connectome(N::Int, K::Int, seed::Int)
    pre = Int[]; post = Int[]
    sizehint!(pre, N * K); sizehint!(post, N * K)
    for j in 0:(N - 1)
        state = UInt64(seed) + UInt64(j) * 0x9e3779b97f4a7c15
        chosen = Set{Int}()
        while length(chosen) < K
            z, state = _splitmix64(state)
            p = Int(z % UInt64(N))
            (p == j || p in chosen) && continue
            push!(chosen, p); push!(pre, p); push!(post, j)
        end
    end
    return pre, post
end

# the signed weight matrix W[post, pre] (JuliaSNN's orientation); E sources +GE, I sources −GI
function weight_matrix(N::Int)
    nw, sy = SPEC["network"], SPEC["synapse"]
    NE = round(Int, nw["ne_frac"] * N)
    pre, post = connectome(N, Int(nw["K"]), Int(nw["seed"]))
    w = [pre[e] < NE ? Float32(sy["GE"]) : -Float32(sy["GI"]) for e in eachindex(pre)]
    return sparse(post .+ 1, pre .+ 1, w, N, N), length(w)
end

# build the AdEx E/I network from a prebuilt weight matrix; `record` monitors spikes (correctness only)
function make_net(W::SparseMatrixCSC, N::Int; record::Bool = false)
    nn, sy, nw = SPEC["neuron"], SPEC["synapse"], SPEC["network"]
    pop = SNN.AdEx(;
        N = N,
        param = SNN.AdExParameter(
            C = Float32(nn["C"]), gl = Float32(nn["gL"]), El = Float32(nn["EL"]),
            Vt = Float32(nn["VT"]), Vr = Float32(nn["Vr"]), ΔT = Float32(nn["dT"]),
            τw = Float32(nn["tauw"]), a = Float32(nn["a"]), b = Float32(nn["b"]),
        ),
        synapse = SNN.CurrentSynapse(τe = Float32(sy["tau"]), τi = Float32(sy["tau"])),
        spike = SNN.PostSpike(At = 0.0f0, τabs = 0.0f0),
    )
    rng = MersenneTwister(Int(nw["seed"]))
    pop.v .= Float32.(rand(rng, N) .* 20 .- 70)      # uniform(−70, −50), matching the other sims
    pop.I .= Float32(sy["I_ext"])                    # constant external drive
    delay = Float32(Int(sy["delay_steps"]) * SPEC["problem"]["dt"])
    syn = SNN.SpikingSynapse(pop, pop, :glu; conn = W, delay_dist = Normal(delay, 0.0f0))
    record && SNN.monitor!(pop, [:fire])
    return pop, syn
end

_rss_mb() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 1.0e6

# a single throughput config: build, warm up, time min-of-reps → one RESULT line. Memory is the RESIDENT
# delta across building the network (W + population + synapse), NOT Sys.maxrss: JuliaSNN's in-process
# package/CUDA load spikes the lifetime-peak RSS to ~1 GB, which would swamp the network footprint.
function run_single(N::Int)
    dt = Float32(SPEC["problem"]["dt"]); Tend = Float32(SPEC["problem"]["T"])
    GC.gc(); base = _rss_mb()
    W, ne = weight_matrix(N)
    p, s = make_net(W, N)
    SNN.sim!([p], [s]; dt = dt, duration = Tend)          # warmup (compile) + realise runtime buffers
    GC.gc(); mem = max(_rss_mb() - base, 0.0)             # network footprint above the post-load baseline
    walls = Float64[]
    for _ in 1:REPS
        p, s = make_net(W, N)                             # fresh state per rep (build not timed)
        t0 = time_ns(); SNN.sim!([p], [s]; dt = dt, duration = Tend); push!(walls, (time_ns() - t0) / 1.0e9)
    end
    wall = minimum(walls)
    println("RESULT snn snn cpu $N $wall $mem $ne")
    return nothing
end

# statistics at N_correctness → one CORR line
function run_correctness()
    N = Int(SPEC["problem"]["N_correctness"]); Tc = Float32(SPEC["problem"]["T_correctness"])
    dt = Float32(SPEC["problem"]["dt"])
    W, ne = weight_matrix(N)
    p, s = make_net(W, N; record = true)
    SNN.sim!([p], [s]; dt = dt, duration = Tc)
    st = SNN.spiketimes(p)
    rate = sum(length, st) / (N * Tc / 1000.0)
    println("CORR $N $rate $(cv_isi(st)) $ne")
    return nothing
end
function cv_isi(spiketimes)
    cvs = Float64[]
    for ts in spiketimes
        length(ts) ≥ 3 || continue
        isi = diff(sort(ts)); m = sum(isi) / length(isi)
        push!(cvs, sqrt(sum(x -> (x - m)^2, isi) / (length(isi) - 1)) / m)
    end
    return isempty(cvs) ? 0.0 : sum(cvs) / length(cvs)
end

writecsv(path, header, rows) = open(path, "w") do io
    println(io, header)
    for r in rows
        println(io, join(r, ","))
    end
end

# orchestrate: a subprocess per N (clean peak memory), collect RESULT/CORR lines, write the CSVs
function main()
    mkpath(OUT)
    Ns = filter(≤(MAXN), Int.(SPEC["problem"]["Ns"]))
    self = @__FILE__; proj = "--project=$HERE"
    perf = Vector{String}[]
    for N in Ns
        out = try
            read(`julia +1.12 $proj $self $N`, String)
        catch
            @warn "config failed" N = N; ""
        end
        for line in split(out, '\n')
            startswith(line, "RESULT") || continue
            f = split(line)
            push!(perf, [f[2], f[3], f[4], f[5], f[6], f[7]])
            println("  [snn/cpu] N=$(f[5])  $(round(parse(Float64, f[6]); digits = 3))s  $(round(parse(Float64, f[7]); digits = 1))MB")
        end
    end
    writecsv(joinpath(OUT, "performance.csv"), "simulator,backend,device,N,wall_s,mem_mb", perf)
    corr = try
        read(`julia +1.12 $proj $self correctness`, String)
    catch
        ""
    end
    for line in split(corr, '\n')
        startswith(line, "CORR") || continue
        f = split(line)
        writecsv(joinpath(OUT, "values.csv"), "N,rate_hz,cv_isi,nedges", [[f[2], f[3], f[4], f[5]]])
        println("  [snn/values] N=$(f[2]) rate=$(round(parse(Float64, f[3]); digits = 1))Hz CV=$(round(parse(Float64, f[4]); digits = 2))")
    end
    return println("wrote $OUT/{performance,values}.csv")
end

isempty(ARGS) ? main() :
    ARGS[1] == "correctness" ? run_correctness() :
    run_single(parse(Int, ARGS[1]))
