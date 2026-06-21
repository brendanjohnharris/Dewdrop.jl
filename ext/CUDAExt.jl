module CUDAExt

# The GPU backend (M6). The engine's hot paths are KernelAbstractions kernels + portable
# broadcasts and its state is `Adapt`-movable, so the GPU backend is mostly the architecture
# seam: map `array_type(GPU())` to `CuArray` and teach `architecture` to recognise device arrays.
# `allocate`/`on_architecture` are generic over `array_type`, and `get_backend(::CuArray)` routes
# every kernel (scatter, monitor aggregate) to the CUDA backend --- no kernel changes needed.

using Dewdrop
using CUDA

Dewdrop.array_type(::Dewdrop.GPU) = CuArray
Dewdrop.architecture(::CuArray) = Dewdrop.GPU()
Dewdrop.architecture(a::CUDA.CuDeviceArray) = Dewdrop.GPU()

end # module CUDAExt
