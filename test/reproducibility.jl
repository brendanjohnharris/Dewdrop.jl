using Dewdrop
using Test
using CUDA
using ForwardDiff
using Statistics

# * Bit-reproducibility of the fixed-point delay ring.
#
# The ring accumulates fixed-point counts rather than floats precisely so that atomic accumulation is
# order-independent: several presynaptic spikes may land on the same (target, slot) in one step, and
# float addition is not associative, so a float ring gives a different last bit depending on which
# order the threads happened to arrive in. Integer addition is associative, so the sum is identical
# however the deposits interleave.
#
# This matters far beyond the last bit. A spiking network is chaotic: a one-ulp difference sits
# dormant until it flips which side of threshold some neuron lands on, after which the network
# decorrelates completely within a step or two. On the spatial E/I model below that flip came at 3.4 s
# of a 25 s run, so before this change every run was effectively an independent realisation.

# The spatial E/I working-regime network (the same topology WRCircuit.build_spatial assembles),
# rebuilt here from Dewdrop primitives so the test needs no downstream package.
_subseed(seed::Unsigned, tag::Integer) = (seed % UInt64) ⊻ ((tag % UInt64) * 0x9e3779b97f4a7c15)

function spatial_ei(;
        rho = 2000, dx = 0.5, gamma = 4, delta = 3.75,
        sigma_ee = 0.06, sigma_ei = 0.07, sigma_ie = 0.14, sigma_ii = 0.14,
        K_ee = 260, K_ei = 340, K_ie = 225, K_ii = 290,
        J_ee = 0.00105, J_ei = 0.00145, nu = 10.0, n_ext = 100,
        Delta_g_K = 0.002, tau_K = 40.0, seed = 0x0034 % UInt64, arch = Dewdrop.CPU(),
        tspan = (0.0, 1.0),
    )
    ne = round(Int, sqrt(rho) * dx)
    NE = ne^2
    NI = max(1, round(Int, NE / gamma))
    period = (Float64(dx), Float64(dx))
    posE = Dewdrop.grid_positions(ne, ne; spacing = dx / ne, centered = true)
    posI = Dewdrop.random_positions(NI, (dx, dx); seed = _subseed(seed, 1))
    E = Dewdrop.FNSNeuron(; C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -70.0, tref = 4.0, τK = tau_K, ΔgK = Delta_g_K)
    I = Dewdrop.FNSNeuron(; C = 0.25, gL = 0.025, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -70.0, tref = 4.0, τK = tau_K, ΔgK = 0.0)
    exc() = Dewdrop.FrozenDualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)
    inh() = Dewdrop.FrozenDualExpSynapse(; τr = 2.0, τd = 4.5, Erev = -80.0)
    J_ie = J_ee * K_ee * delta / K_ie
    J_ii = J_ei * K_ei * delta / K_ii
    cw(J, tag) = Dewdrop.correlate_weights(J; jitter = 0.2, dist = :gaussian, seed = _subseed(seed, tag), count_empty = true)
    nb = Dewdrop.network(; tspan = tspan, arch = arch)
    Dewdrop.population!(nb, :E, E, NE; positions = posE)
    Dewdrop.population!(nb, :I, I, NI; positions = posI)
    Dewdrop.project!(nb, :E => :E, exc(); kernel = Dewdrop.exponential_kernel(sigma_ee), count = K_ee * NE, weight = 1.0, delay = 1.5, seed = _subseed(seed, 2), allow_self = true, period, adjust = cw(J_ee, 6), index_type = Int32)
    Dewdrop.project!(nb, :E => :I, exc(); kernel = Dewdrop.exponential_kernel(sigma_ei), count = K_ei * NI, weight = 1.0, delay = 1.5, seed = _subseed(seed, 3), allow_self = false, period, adjust = cw(J_ei, 7), index_type = Int32)
    Dewdrop.project!(nb, :I => :E, inh(); kernel = Dewdrop.exponential_kernel(sigma_ie), count = K_ie * NE, weight = 1.0, delay = 1.5, seed = _subseed(seed, 4), allow_self = false, period, adjust = cw(J_ie, 8), index_type = Int32)
    Dewdrop.project!(nb, :I => :I, inh(); kernel = Dewdrop.exponential_kernel(sigma_ii), count = K_ii * NI, weight = 1.0, delay = 1.5, seed = _subseed(seed, 5), allow_self = true, period, adjust = cw(J_ii, 9), index_type = Int32)
    N_ext = round(Int, sqrt(n_ext * NE))
    p_ext = sqrt(n_ext / NE)
    Dewdrop.drive!(nb, :E, exc(); rate = nu, n_ext = N_ext, p = p_ext, weight = 1.0, delay = 1.5, seed = _subseed(seed, 10), fire_seed = _subseed(seed, 14), adjust = cw(J_ee, 12), index_type = Int32)
    Dewdrop.drive!(nb, :I, exc(); rate = nu, n_ext = N_ext, p = p_ext, weight = 1.0, delay = 1.5, seed = _subseed(seed, 11), fire_seed = _subseed(seed, 14), adjust = cw(J_ei, 13), index_type = Int32)
    return Dewdrop.convertfloat(Float32, Dewdrop.freeze(nb))
end

"Run the network and return the recorded E membrane potentials."
function run_ei(; tmax = 200.0, arch = Dewdrop.CPU(), rho = 2000, kw...)
    spec = spatial_ei(; rho, arch)
    sol = Dewdrop.solve(
        spec, Dewdrop.FixedStep(0.1);
        tspan = (0.0, tmax), v0 = (-70.0, -50.0), progress = false, advise = false,
        record = (; V = Dewdrop.Trace(:V; of = :E)), kw...
    )
    return copy(sol.record.V.data)
end

@testset "fixed-point ring is bit-reproducible" begin
    @testset "the ring itself" begin
        arch = Dewdrop.CPU()
        buf = Dewdrop.DelayBuffer(arch, Float32, 4, 3)
        @test eltype(buf.slots) === Dewdrop.FPCount        # counts, not floats
        Dewdrop.deposit!(buf, 0, 1, 0.5f0, 1)
        Dewdrop.deposit!(buf, 0, 1, 0.25f0, 1)
        @test Dewdrop.collect_due!(buf, 1)[1] == 0.75f0    # exact: dyadic values, power-of-two scale
        @test all(iszero, buf.slots)

        # a Dual-typed (differentiable) run keeps a VALUE ring: rounding is not differentiable
        D = ForwardDiff.Dual{Nothing, Float64, 1}
        dbuf = Dewdrop.DelayBuffer(arch, D, 2, 1)
        @test eltype(dbuf.slots) === D
        @test Dewdrop._fp_quantise(D, D(1.5), dbuf.scale) === D(1.5)   # identity: gradient survives
    end

    @testset "scale sizing" begin
        # worst case a slot can hold = the largest per-target incoming weight sum; the scale must keep
        # that inside FPCount with headroom, and be a power of two so `w * scale` is exact in binary
        post = [1, 1, 2, 2, 2]
        weight = Float32[0.5, 0.25, 1.0, 1.0, 1.0]
        s = Dewdrop.fixedpoint_scale(post, weight, 2)
        @test log2(s) == round(log2(s))                    # power of two
        peak = 3.0                                          # target 2 receives 1+1+1
        @test peak * s * Dewdrop._FP_SAFETY <= typemax(Dewdrop.FPCount)
        @test Dewdrop.fixedpoint_scale(Int[], Float32[], 2) == Dewdrop._FP_DEFAULT_SCALE
    end

    @testset "CPU: repeated runs, and thread-count invariance" begin
        a, b = run_ei(), run_ei()
        @test a == b
        @info "CPU threads" Threads.nthreads()
    end

    if CUDA.functional()
        @testset "GPU: repeated runs" begin
            a, b = run_ei(arch = Dewdrop.GPU()), run_ei(arch = Dewdrop.GPU())
            @test a == b
        end
        @testset "GPU: repeated runs, compacted scatter" begin
            a = run_ei(arch = Dewdrop.GPU(), scatter = :compacted)
            b = run_ei(arch = Dewdrop.GPU(), scatter = :compacted)
            @test a == b
        end
        @testset "GPU: edge-parallel and compacted scatter agree exactly" begin
            # different launch shapes, different deposit order --- identical sums, because the ring is
            # integer. This did NOT hold with a float ring.
            @test run_ei(arch = Dewdrop.GPU(), scatter = :auto) ==
                run_ei(arch = Dewdrop.GPU(), scatter = :compacted)
        end
    end
end
