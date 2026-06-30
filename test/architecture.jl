using Dewdrop
using Test
using Adapt

# The architecture seam.
# A single `AbstractArchitecture` trait with a single allocation point, so the
# whole simulation can be moved to a device via `Adapt`.
@testset "architecture seam" begin
    @test Dewdrop.CPU() isa Dewdrop.AbstractArchitecture

    # `array_type` names the backing array type for an architecture.
    @test Dewdrop.array_type(Dewdrop.CPU()) === Array

    # `on_architecture` moves data onto an architecture (no-op-ish on CPU).
    v = [1.0, 2.0, 3.0]
    w = Dewdrop.on_architecture(Dewdrop.CPU(), v)
    @test w isa Array{Float64}
    @test w == v

    # scalars / non-array data pass through unchanged.
    @test Dewdrop.on_architecture(Dewdrop.CPU(), 3.0) === 3.0

    # `allocate` is the single allocation point; it routes through the arch.
    a = Dewdrop.allocate(Dewdrop.CPU(), Float32, 5)
    @test a isa Vector{Float32}
    @test length(a) == 5
end

# The inverse map: recover the architecture that owns an array (counterpart of
# `on_architecture`). Recurses through wrapper arrays to their parent; GPU array
# types extend it in the GPU extension. Used by tests and the reinit path.
@testset "inverse architecture map" begin
    @test Dewdrop.architecture([1.0, 2.0]) === Dewdrop.CPU()
    @test Dewdrop.architecture(view([1.0, 2.0, 3.0], 1:2)) === Dewdrop.CPU()        # SubArray
    @test Dewdrop.architecture(reshape(collect(1:12), 3, 4)) === Dewdrop.CPU()
    @test Dewdrop.architecture(PermutedDimsArray(rand(2, 3), (2, 1))) === Dewdrop.CPU()
end
