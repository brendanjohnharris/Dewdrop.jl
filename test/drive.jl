using Dewdrop
using Test
using Adapt
using JLArrays

# External Poisson drive: each step every neuron receives n ~ Poisson(rate·dt)
# external spikes, each a voltage kick of `weight` (the Brunel external input). Drawn from
# the counter-based RNG keyed by (seed, step, neuron) → reproducible, GPU-safe.
@testset "Poisson external drive" begin
    # Brunel-style dimensionless units (mV from rest, ms)
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    dt, tend = 0.1, 1000.0

    drive = Dewdrop.PoissonDrive(; rate = 20.0, weight = 0.1, seed = UInt64(1))   # 20/ms × 0.1 mV
    prob = DewdropNetwork(m, 50; input = 0.0, tspan = (0.0, tend), drive = drive)
    sol = solve(prob, FixedStep(dt))
    @test all(>(0), sol.spike_count)                 # all neurons fire (purely drive-driven)

    # stronger drive → higher rate
    drive2 = Dewdrop.PoissonDrive(; rate = 30.0, weight = 0.1, seed = UInt64(1))
    sol2 = solve(DewdropNetwork(m, 50; input = 0.0, tspan = (0.0, tend), drive = drive2), FixedStep(dt))
    @test sum(sol2.spike_count) > sum(sol.spike_count)

    # reproducible (same seed → identical)
    @test solve(prob, FixedStep(dt)).spike_count == sol.spike_count

    # no drive + no input → silent
    sol0 = solve(DewdropNetwork(m, 50; input = 0.0, tspan = (0.0, tend)), FixedStep(dt))
    @test all(==(0), sol0.spike_count)

    # GPU-safe + deterministic: the Poisson drive broadcast runs under JLArray and matches
    # CPU step-for-step (the counter-based draw is identical on both). 300 steps = 30 ms
    # exceeds the first threshold crossing (~τ·ln2 ≈ 14 ms).
    gpu = adapt(JLArray, init(prob, FixedStep(dt)))
    cpu = init(prob, FixedStep(dt))
    for _ in 1:300
        step!(gpu)
        step!(cpu)
    end
    @test sum(Array(gpu.spike_count)) > 0
    @test Array(gpu.spike_count) == cpu.spike_count
end
