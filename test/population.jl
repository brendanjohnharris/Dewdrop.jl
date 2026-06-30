using Dewdrop
using Test
using Adapt
using StructArrays
using JLArrays

# Struct-of-arrays population state of `isbits` elements,
# allocated through the architecture seam and movable to a device via `Adapt`.
@testset "SoA population state" begin
    arch = Dewdrop.CPU()
    pop = Dewdrop.Population(arch, Float32, (:V, :ge), 4)

    # columns are real Arrays allocated through the arch, zero-initialised.
    @test pop.state isa StructArray
    @test pop.state.V isa Vector{Float32}
    @test pop.state.ge isa Vector{Float32}
    @test length(pop.state.V) == 4
    @test length(pop) == 4
    @test all(iszero, pop.state.V)
    @test all(iszero, pop.state.ge)

    # the SoA element type must be isbits (the GPU-movability precondition).
    @test isbitstype(eltype(pop.state))

    # parametric float type is honoured.
    pop64 = Dewdrop.Population(arch, Float64, (:V,), 3)
    @test pop64.state.V isa Vector{Float64}

    # Adapt-movable to a device-like array type (no GPU required).
    gpop = adapt(JLArray, pop)
    @test gpop.state.V isa JLArray{Float32}
    @test gpop.state.ge isa JLArray{Float32}
    @test Array(gpop.state.V) == pop.state.V
    @test length(gpop) == 4
end
