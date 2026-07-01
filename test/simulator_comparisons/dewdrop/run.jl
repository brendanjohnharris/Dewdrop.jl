#! /bin/bash
#=
exec julia +1.12 -t auto --project="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/test" "${BASH_SOURCE[0]}" "$@"
=#
# Dewdrop's implementation of the shared simulator-comparison problem (../spec.toml). Standalone:
#     ./run.jl                          # every backend × device → out/{values,performance}.csv
#     ./run.jl <backend> <device> <N>   # a single throughput config (used internally per subprocess)
#     ./run.jl correctness              # the statistics (rate, CV-ISI) run
# Each (backend, device, N) runs in a FRESH subprocess so its wall time and peak memory are clean.
# Backends: serial, fused, turbo (CPU) + fused (GPU). No simulator-specific parameters live here ---
# everything comes from ../spec.toml. (No Julia lives in the Python simulator folders; this is it.)

using Dewdrop
using CUDA
using LoopVectorization     # activates the Turbo backend
import TOML

const HERE = @__DIR__
const SPEC = TOML.parsefile(joinpath(HERE, "..", "spec.toml"))
const OUT = joinpath(HERE, "out")

# --- language-agnostic connectome (splitmix64 fixed in-degree; identical across simulators) ---
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

# --- build the AdEx E/I network from the spec ---
function build_prob(N::Int, arch, ::Type{T}, ::Type{IT}, tspan) where {T, IT}
    nn, sy, nw = SPEC["neuron"], SPEC["synapse"], SPEC["network"]
    NE = round(Int, nw["ne_frac"] * N)
    pre, post = connectome(N, Int(nw["K"]), Int(nw["seed"]))
    edges = [(pre[e] + 1, post[e] + 1, T(pre[e] < NE ? sy["GE"] : -sy["GI"]), Int(sy["delay_steps"])) for e in eachindex(pre)]
    conn = SparseCSR(arch, edges; npre = N, npost = N, index_type = IT)
    m = AdEx(;
        C = T(nn["C"]), gL = T(nn["gL"]), EL = T(nn["EL"]), VT = T(nn["VT"]), ΔT = T(nn["dT"]),
        Vr = T(nn["Vr"]), Vpeak = T(nn["Vpeak"]), a = T(nn["a"]), b = T(nn["b"]), τw = T(nn["tauw"]), tref = T(nn["tref"])
    )
    prob = DewdropNetwork(
        m, N; input = T(sy["I_ext"]), tspan = tspan, arch = arch,
        projection = Projection(CurrentSynapse(τ = T(sy["tau"])), conn)
    )
    return prob, Dewdrop.nedges(conn)
end

# memory: peak RSS minus the post-import baseline = the network footprint (excludes runtime/packages)
_rss_mb() = parse(Int, split(read("/proc/self/statm", String))[2]) * 4096 / 1.0e6
_peak_mb() = Sys.maxrss() / 1.0e6

# this process's GPU *device* memory (MB) via nvidia-smi --- the network's GPU footprint, not host RSS
function _gpu_proc_mem_mb()
    try
        out = read(`nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits`, String)
        for line in split(out, '\n')
            parts = strip.(split(line, ','))
            length(parts) == 2 && tryparse(Int, parts[1]) == getpid() &&
                return something(tryparse(Float64, parts[2]), 0.0)
        end
    catch
    end
    return 0.0
end

# --- a single throughput config: build, warm up, time, measure → one RESULT line ---
function run_single(backend::Symbol, device::Symbol, N::Int)
    bk = backend === :serial ? Serial() : backend === :fused ? Fused() : Turbo()
    arch = device === :gpu ? GPU() : CPU()
    T = Float32; IT = Int32
    dt = T(SPEC["problem"]["dt"]); Tend = T(SPEC["problem"]["T"])
    base = _rss_mb()
    prob, ne = build_prob(N, arch, T, IT, (T(0), Tend))
    mk() = init(prob, FixedStep(dt); v0 = (T(-70), T(-50)), backend = bk)
    sync!(g) = device === :gpu ? CUDA.@sync(solve!(g)) : solve!(g)
    sync!(mk())                                          # warmup (compile)
    gmem = device === :gpu ? _gpu_proc_mem_mb() : 0.0    # GPU device memory, one network resident
    wall = minimum((g = mk(); t0 = time_ns(); sync!(g); (time_ns() - t0) / 1.0e9) for _ in 1:4)
    mem = device === :gpu ? gmem : max(_peak_mb() - base, 0.0)
    println("RESULT dewdrop $backend $device $N $wall $mem $ne")
    return nothing
end

# --- statistics (Float64, CPU, Serial) at N_correctness → one CORR line ---
function run_correctness()
    N = Int(SPEC["problem"]["N_correctness"]); Tc = Float64(SPEC["problem"]["T_correctness"])
    prob, ne = build_prob(N, CPU(), Float64, Int, (0.0, Tc))
    sol = solve(
        prob, FixedStep(Float64(SPEC["problem"]["dt"])); v0 = (-70.0, -50.0),
        backend = Serial(), record = (spk = Spikes(),)
    )
    rate = sum(sol.spike_count) / (N * Tc / 1000.0)
    t, id = raster(sol; name = :spk)
    println("CORR $N $rate $(cv_isi(t, id, N)) $ne")
    return nothing
end
function cv_isi(times, ids, N)
    by = [Float64[] for _ in 1:N]
    for (t, j) in zip(times, ids)
        push!(by[j], t)
    end
    cvs = Float64[]
    for ts in by
        length(ts) ≥ 3 && (isi = diff(sort(ts)); m = sum(isi) / length(isi); push!(cvs, sqrt(sum(x -> (x - m)^2, isi) / (length(isi) - 1)) / m))
    end
    return isempty(cvs) ? 0.0 : sum(cvs) / length(cvs)
end

writecsv(path, header, rows) = open(path, "w") do io
    println(io, header)
    for r in rows
        println(io, join(r, ","))
    end
end

# --- orchestrate: a subprocess per config, collect RESULT/CORR lines, write the standard CSVs ---
function main()
    mkpath(OUT)
    Ns = Int.(SPEC["problem"]["Ns"])
    proj = "--project=$(joinpath(HERE, "..", "..", "..", "test"))"
    self = @__FILE__
    configs = [(:serial, :cpu), (:fused, :cpu), (:turbo, :cpu), (:fused, :gpu)]
    perf = Vector{String}[]
    for (bk, dev) in configs, N in Ns
        out = try
            read(`julia +1.12 -t auto $proj $self $bk $dev $N`, String)
        catch
            @warn "config failed" backend = bk device = dev N = N; ""
        end
        for line in split(out, '\n')
            startswith(line, "RESULT") || continue
            f = split(line)
            push!(perf, [f[2], f[3], f[4], f[5], f[6], f[7]])
            println("  [dewdrop/$(f[3])/$(f[4])] N=$(f[5])  $(round(parse(Float64, f[6]); digits = 3))s  $(round(parse(Float64, f[7]); digits = 1))MB")
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
        println("  [dewdrop/values] N=$(f[2]) rate=$(round(parse(Float64, f[3]); digits = 1))Hz CV=$(round(parse(Float64, f[4]); digits = 2))")
    end
    return println("wrote $OUT/{performance,values}.csv")
end

isempty(ARGS) ? main() :
    ARGS[1] == "correctness" ? run_correctness() :
    run_single(Symbol(ARGS[1]), Symbol(ARGS[2]), parse(Int, ARGS[3]))
