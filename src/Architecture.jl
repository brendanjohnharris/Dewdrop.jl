# * Architecture seam (M0 contract 1)
# A single trait selecting the execution device/memory, with one allocation
# chokepoint, so an entire simulation can be moved to a device via `Adapt`.

"""
    AbstractArchitecture

Trait selecting the execution device and memory space for a simulation. [`CPU`](@ref)
ships in the core; a `GPU` architecture is provided by the CUDA/KernelAbstractions
package extensions.
"""
abstract type AbstractArchitecture end

"""
    CPU()

Host architecture; simulation state lives in `Array`s and kernels run on CPU
threads.
"""
struct CPU <: AbstractArchitecture end
export CPU

"""
    GPU()

Device architecture; simulation state lives in GPU arrays and the same KernelAbstractions
kernels run on the device. Requires a GPU package extension --- `using CUDA` enables it (the
extension maps [`array_type`](@ref)`(GPU())` to `CuArray`).
"""
struct GPU <: AbstractArchitecture end
export GPU

"""
    array_type(arch)

The backing array type used to allocate state on `arch` (`Array` for [`CPU`](@ref)).
"""
function array_type end
array_type(::CPU) = Array

"""
    on_architecture(arch, x)

Move `x` onto `arch`'s memory. Delegates to `Adapt.adapt`, so scalars and other
non-array leaves pass through unchanged and nested structures recurse field-wise.
"""
on_architecture(arch::AbstractArchitecture, x) = adapt(array_type(arch), x)

"""
    architecture(x) -> AbstractArchitecture

The architecture that owns array `x` --- the inverse of [`on_architecture`](@ref).
Recurses through Base wrapper arrays (`SubArray`, `ReshapedArray`, `PermutedDimsArray`)
to their parent; GPU array types extend this in the GPU package extension. Used by
tests and the reinit path to recover an array's device.
"""
architecture(::Array) = CPU()
architecture(a::SubArray) = architecture(parent(a))
architecture(a::Base.ReshapedArray) = architecture(parent(a))
architecture(a::Base.PermutedDimsArray) = architecture(parent(a))

"""
    allocate(arch, T, dims...)

The single allocation chokepoint (M0 contract 1): allocate an uninitialised array
of element type `T` with shape `dims` on `arch`. All core state allocation routes
through here so the architecture controls the memory space.
"""
function allocate(arch::AbstractArchitecture, ::Type{T}, dims::Integer...) where {T}
    return array_type(arch){T}(undef, dims...)
end
