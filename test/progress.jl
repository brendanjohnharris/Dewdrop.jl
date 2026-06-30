using Dewdrop
using Test
using Logging

# Progress bar (src/Progress.jl): `solve(prob, alg; progress = …)` emits the ProgressLogging
# convention --- a `@logmsg LogLevel(-1) name progress=frac _id=id` per update --- so VSCode /
# TerminalLoggers render a live bar and a bare script silently filters it. We do NOT depend on any
# progress package; the producer side is a logging convention, captured here with a `TestLogger`.
#
# Contract under test:
#   `:auto` (default) → on, but auto-SUPPRESSED until the run has run ~0.3 s wall-clock (so trivial
#                       runs never flash a bar); the update stride is CALIBRATED to ~2 Hz.
#   `true`            → force on from the start (visible during the calibration window too).
#   `false`           → off (no records at all).
#   "label"::String   → force on, custom bar name.
#   N::Int            → force on, update every N steps (skip calibration).
# The bar must NEVER perturb the simulation (read-only): a `progress` run is bit-identical to none.

# every captured log record carrying a `progress` kwarg → (record-id, fraction)
function _progress_records(logs)
    out = Tuple{Any, Any}[]
    for r in logs, (k, v) in r.kwargs
        k === :progress && push!(out, (r.id, v))
    end
    return out
end
_is_progress(r) = any(((k, v),) -> k === :progress, r.kwargs)

# run `f` (a `solve` call) capturing progress log records at the ProgressLogging level (-1)
function _capture(f)
    logger = Test.TestLogger(min_level = Logging.LogLevel(-1))
    res = Logging.with_logger(f, logger)
    return res, _progress_records(logger.logs), logger.logs
end

_lif() = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 100.0, tref = 2.0)
# suprathreshold input so there is real spiking dynamics; 50 neurons
_prob(; tend) = DewdropNetwork(_lif(), 50; input = 0.25, tspan = (0.0, tend))

@testset "progress bar" begin
    @testset "progress=true emits a monotone, single-id bar that completes" begin
        prob = _prob(tend = 2000.0)            # 20000 steps at dt=0.1
        _, recs, _ = _capture() do
            solve(prob, FixedStep(0.1); progress = true)
        end
        @test !isempty(recs)
        @test length(unique(first.(recs))) == 1               # exactly one bar (one _id)
        fracs = Float64[v for (_, v) in recs if v isa Real]
        @test !isempty(fracs)
        @test all(0.0 .<= fracs .<= 1.0)                      # valid fractions
        @test issorted(fracs)                                 # monotone non-decreasing
        @test last(fracs) >= 1.0                              # the bar completes
    end

    @testset "progress=false emits nothing" begin
        prob = _prob(tend = 2000.0)
        _, recs, _ = _capture() do
            solve(prob, FixedStep(0.1); progress = false)
        end
        @test isempty(recs)
    end

    @testset ":auto suppresses a trivial (sub-0.3 s) run" begin
        prob = _prob(tend = 0.2)               # 2 steps → finishes inside the calibration window
        _, recs, _ = _capture() do
            solve(prob, FixedStep(0.1); progress = :auto)
        end
        @test isempty(recs)
    end

    @testset "default is :auto (suppressed for a trivial run)" begin
        prob = _prob(tend = 0.2)
        _, recs, _ = _capture() do
            solve(prob, FixedStep(0.1))
        end
        @test isempty(recs)
    end

    @testset "custom name labels the bar" begin
        prob = _prob(tend = 100.0)
        _, _, logs = _capture() do
            solve(prob, FixedStep(0.1); progress = "FNS net")
        end
        names = unique(r.message for r in logs if _is_progress(r))
        @test "FNS net" in names
    end

    @testset "explicit cadence: progress=N updates every N steps" begin
        prob = _prob(tend = 100.0)             # 1000 steps
        _, recs, _ = _capture() do
            solve(prob, FixedStep(0.1); progress = 100)       # ~10 updates + a completion
        end
        @test !isempty(recs)
        @test last(Float64[v for (_, v) in recs if v isa Real]) >= 1.0
    end

    @testset "progress does not perturb the solution (bit-identical)" begin
        prob = _prob(tend = 100.0)
        s_on = solve(prob, FixedStep(0.1); progress = true)
        s_off = solve(prob, FixedStep(0.1); progress = false)
        @test s_on.spike_count == s_off.spike_count
        @test s_on.state.state.V == s_off.state.state.V
    end

    @testset "batched solve emits progress too" begin
        prob = _prob(tend = 100.0)
        _, recs, _ = _capture() do
            solve(prob, FixedStep(0.1); progress = 50, batch = 4)
        end
        @test !isempty(recs)
    end
end
