using Dewdrop
using Test
using Statistics
using TimeseriesBase

# The TimeseriesBase labeling extension: recorded monitors wrapped as ToolsArrays with
# meaningful `Time`/`Neuron` dimensions. Loading TimeseriesBase activates the ext and injects the
# custom `Neuron`/`Synapse` dims into Dewdrop's namespace.
@testset "TimeseriesBase labeled outputs" begin
    m = LIF(; τ = 20.0, EL = 0.0, Vθ = 20.0, Vr = 10.0, R = 1.0, tref = 2.0)
    prob = DewdropNetwork(m, 8; input = 1.5, tspan = (0.0, 50.0))
    nsteps = 500
    sol = solve(prob, FixedStep(0.1); record = (spikes = Spikes(), V = Trace(:V)))

    @testset "trace → Time × Neuron Timeseries" begin
        vts = Timeseries(sol)                                     # defaults to the :V monitor
        @test vts isa TimeseriesBase.AbstractToolsArray
        @test size(vts) == (nsteps, 8)
        @test parent(vts) == permutedims(sol.record.V.data)      # data, time-first
        @test step(dims(vts, 𝑡)) ≈ 0.1                           # Time axis matches dt
        @test collect(val(dims(vts, Dewdrop.Neuron))) == collect(1:8)
    end

    @testset "subset monitor → Neuron axis carries the actual indices" begin
        vts = Timeseries(solve(prob, FixedStep(0.1); record = (V = Trace(:V; of = [2, 5, 7]),)))
        @test size(vts) == (nsteps, 3)
        @test collect(val(dims(vts, Dewdrop.Neuron))) == [2, 5, 7]
    end

    @testset "aggregate → univariate Time series" begin
        agg = solve(prob, FixedStep(0.1); record = (meanV = Aggregate(Trace(:V), :mean),))
        ts = Timeseries(agg, :meanV)
        @test ts isa TimeseriesBase.AbstractToolsArray
        @test size(ts) == (nsteps,)
        @test collect(parent(ts)) ≈ vec(agg.record.meanV.data)
    end

    @testset "spike raster → labeled Neuron × Time" begin
        st = spiketrain(sol)
        @test st isa TimeseriesBase.AbstractToolsArray
        @test size(st) == (8, nsteps)
        @test sum(st) == sum(sol.record.spikes.data)             # same spikes, now labeled
        @test collect(val(dims(st, Dewdrop.Neuron))) == collect(1:8)
        @test step(dims(st, 𝑡)) ≈ 0.1
    end

    @testset "custom dims exposed + errors when unrecorded" begin
        @test Dewdrop.Neuron isa UnionAll || Dewdrop.Neuron isa DataType
        @test Dewdrop.Synapse isa UnionAll || Dewdrop.Synapse isa DataType
        sol0 = solve(prob, FixedStep(0.1))
        @test_throws Exception Timeseries(sol0)
        @test_throws Exception spiketrain(sol0)
    end

    @testset "Population / Var labelled outputs (named subpopulations)" begin
        ei = DewdropNetwork(m, 8; input = 1.5, tspan = (0.0, 50.0), subpops = (E = 1:4, I = 5:8))
        sol = solve(ei, FixedStep(0.1); record = (spikes = Spikes(), V = Trace(:V)))

        # a trace restricted to a subpop: Time × Neuron over that subpop's indices
        vE = Timeseries(sol, :V; of = :E)
        @test size(vE) == (nsteps, 4)
        @test collect(val(dims(vE, Dewdrop.Neuron))) == [1, 2, 3, 4]
        @test parent(vE) == permutedims(sol.record.V.data[1:4, :])

        # a raster restricted to a subpop
        sI = spiketrain(sol; of = :I)
        @test size(sI) == (4, nsteps)
        @test collect(val(dims(sI, Dewdrop.Neuron))) == [5, 6, 7, 8]
        @test sum(sI) == sum(sol.record.spikes.data[5:8, :])

        # the bpsolve-style Population × Var nested array of per-population timeseries
        X = Timeseries(sol, [:E, :I]; vars = [:V])
        @test X isa TimeseriesBase.AbstractToolsArray
        @test size(X) == (2, 1)
        @test :Population in TimeseriesBase.name(dims(X))      # the Population axis (referenced by symbol)
        @test collect(val(dims(X, :Population))) == [:E, :I]
        @test collect(val(dims(X, Var))) == [:V]
        @test parent(X[1, 1]) == parent(vE)                  # the (E, V) cell is the E trace
        @test size(X[2, 1]) == (nsteps, 4)                   # the (I, V) cell
    end
end
