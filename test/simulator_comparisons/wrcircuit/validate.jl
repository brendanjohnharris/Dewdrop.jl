# Guarded integration test: asserts the Dewdrop WRCircuit reproduction matches the BrainPy reference.
# Requires the export (../wrcircuit/{brainpy,dewdrop}/out) to be present --- regenerate with
#   python brainpy/run.py        (needs the BrainPy env)
#   julia dewdrop/run.jl
# so it is NOT part of the default suite (which has no BrainPy); run explicitly, or via run_all.sh.
# The bit-for-bit cross-simulator agreement is the acceptance criterion: identical firing rates, a
# sub-0.05 mV mean membrane error, and the great majority of spikes matching to the exact step (all
# within one step) --- the residual is chaotic amplification of float-ordering differences (JAX vs Julia).
using Test

include(joinpath(@__DIR__, "compare.jl"))

@testset "WRCircuit: Dewdrop reproduces BrainPy" begin
    if !isfile(joinpath(DD, "vE.csv")) || !isfile(joinpath(BP, "vE.csv"))
        @info "WRCircuit reproduction: export not found --- run brainpy/run.py + dewdrop/run.jl first; skipping."
        @test_skip true
    else
        m = compare_wrcircuit(; plot = false)
        for p in (m.E, m.I)
            @test p.nspk_bp == p.nspk_dd                     # identical spike count
            @test isapprox(p.rate_bp, p.rate_dd; rtol = 1.0e-6)  # identical firing rate
            @test p.vmean < 0.05                             # mean |ΔV| well under a mV
            @test p.vcor > 0.999                             # traces track
            @test p.within1 == p.tot                         # every spike matches within ±1 step
            @test p.exact ≥ 0.85 * p.tot                     # the great majority match to the exact step
        end
    end
end
