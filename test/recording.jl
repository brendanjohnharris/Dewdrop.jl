using Dewdrop
using Test
using Adapt
using JLArrays

# M1d --- opt-in, preallocated recording. The `:record` phase writes the spike mask (and,
# if requested, V) into preallocated (N x nsteps) buffers, one column per step (a strided
# broadcast write, GPU-safe). Default records only per-neuron counts.
@testset "recording: spikes + voltage traces" begin
    m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
    dt, tend = 0.1, 100.0
    prob = DewdropNetwork(m, 4; input = 0.5, tspan = (0.0, tend))   # supra-rheobase
    nsteps = round(Int, tend / dt)

    sol = solve(prob, FixedStep(dt); record_spikes = true, record_voltage = true)
    @test size(sol.spikes) == (4, nsteps)
    @test size(sol.voltages) == (4, nsteps)
    @test eltype(sol.spikes) == Bool

    # recorded raster is consistent with the per-neuron spike counts
    @test vec(sum(sol.spikes; dims = 2)) == sol.spike_count
    # recorded V is post-reset each step (never the supra-threshold crossing value), and
    # ramps from rest EL, so it lies in [EL, Vθ)
    @test all(v -> m.EL - 1e-6 ≤ v < m.Vθ, sol.voltages)
    # the last recorded column is the final state
    @test sol.voltages[:, end] == sol.state.state.V

    # default: no spike/voltage recording
    sol0 = solve(prob, FixedStep(dt))
    @test sol0.spikes === nothing
    @test sol0.voltages === nothing
    @test sol0.spike_count == sol.spike_count        # counts are unaffected by recording

    # raster extraction matches the recorded spike matrix
    times, ids = raster(sol)
    @test length(times) == sum(sol.spikes)
    @test all(i -> 1 ≤ i ≤ 4, ids)
    @test all(t -> 0 < t ≤ tend, times)

    # recording is GPU-safe: the recorded step runs under JLArray
    gpu = adapt(JLArray, init(prob, FixedStep(dt); record_spikes = true))
    @test gpu.spike_rec isa JLArray
    for _ in 1:50
        step!(gpu)
    end
    @test sum(Array(gpu.spike_rec)) ≥ 0              # ran without scalar-indexing errors
end
