```@meta
CurrentModule = Dewdrop
```

# Running on the GPU

Dewdrop runs the same simulation on the CPU or the GPU by flipping one flag. Two orthogonal choices
are involved, and only one of them changes:

- **Architecture** ([`CPU`](@ref) / [`GPU`](@ref), set on the [`DewdropNetwork`](@ref)): *where*
  the state lives. `arch = GPU()` allocates every state array on the device.
- **Execution backend** (`backend = …`, passed to [`solve`](@ref)): *how* each step runs. On the
  GPU the backend is **always the fused megakernel**; the default [`Auto`](@ref) resolves to
  [`Fused`](@ref) there (see [choosing a backend](backends.md)), and the per-phase CPU paths do not
  apply.

## Activating CUDA

The core ships only the [`CPU`](@ref) architecture. `using CUDA` loads the package extension that maps
[`array_type`](@ref)`(GPU())` to `CuArray`, so [`GPU`](@ref) becomes usable; a functional CUDA device
is required to actually run.

```julia
using Dewdrop
using CUDA          # loads the GPU extension (array_type(GPU()) -> CuArray)
```

## Swap between CPU and GPU

Build the connectome on the target architecture (the connectome is stored where it is built), set
`arch = GPU()` on the network, and solve exactly as on the CPU:

```julia
N = 10_000
conn = fixed_prob(GPU(), N, N, 0.02; weight = 0.5, delay = 1.0, seed = 0x1234)

prob = DewdropNetwork(LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 5.0), N;
                      input = 0.0, tspan = (0.0, 1000.0),
                      projections = (Projection(CurrentSynapse(; τ = 5.0), conn),),
                      drive = PoissonDrive(rate = 20.0, weight = 0.5, seed = 0x1),
                      arch = GPU())

sol = solve(prob, FixedStep(0.1))      # same call as the CPU path
```

The only line that changed from a CPU run is `arch = GPU()` (and `GPU()` in [`fixed_prob`](@ref)). The
GPU path matches the CPU path's dynamics to within the documented `exp` ULP (see
[choosing a backend](backends.md)); the threaded atomic scatter is order-dependent, so
byte-exact reproducibility needs a single CPU thread, but the statistics are identical.

## What runs on the device

- the **dense per-neuron megakernel** (deliver + accumulate + membrane + threshold + reset, one launch
  over the `N` cells);
- the **sparse synaptic scatter** (one kernel over the connectome, see below).

Recording is **windowed**: each monitor stages its samples in a device buffer and flushes in bulk, so
host transfers are `O(1)` per window rather than per step. Positions stay **host-side** (they are
metadata for the spatial measures in [analysis](analysis.md), never touched in the hot loop).

## Spike scatter strategy

On the GPU the synaptic scatter has two strategies, chosen by `scatter` on [`solve`](@ref)/[`init`](@ref):

```julia
sol = solve(prob, FixedStep(0.1); scatter = :auto)    # :auto (default) | :edge | :compacted
```

- **`:edge`**: one thread per synapse; saturates the device at small/medium sizes but reads an
  index per edge every step, so a very large connectome that spills the L2 cache degrades.
- **`:compacted`**: processes only the out-edges of neurons that actually spiked (work ∝ spikes),
  at the cost of one device→host sync per step; far faster on large, sparsely-firing networks.

`:auto` switches from `:edge` to `:compacted` once the connectome's index footprint exceeds about half
the device L2 (the measured crossover). Plastic (STDP) projections always use `:edge`: the
compacted scatter cannot drive the postsynaptic-potentiation branch. The full discussion, including
the L2-spill crossover and the CPU case, is in [choosing a backend](backends.md).

## Batching on the device

Pass `batch = B` to run `B` reproducible ensemble members together over an `(N, B)` state, amortising
launch overhead and keeping the device busy:

```julia
sol = solve(prob, FixedStep(0.1); batch = 64)         # BatchedSolution; fields are (N, B)
```

The batched path is always the fused megakernel, so `backend` does not apply; `arch` still selects
where the state lives. Each member draws an independent, bit-reproducible stream. See
[batching & ensembles](batching.md) for per-member variation and the block-diagonal alternative.

## Float32 on the GPU

The device prefers single precision. Write the model in convenient `Float64` literals and switch the
whole network (model, connectome, weights) in one call with [`convertfloat`](@ref), and use
`Int32` connectome indices to halve the scatter's index bandwidth:

```julia
m32 = convertfloat(Float32, LIF(; τ = 20.0, EL = -60.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 5.0))
conn = fixed_prob(GPU(), N, N, 0.02; weight = 0.5f0, delay = 1.0, seed = 0x1234, index_type = Int32)

prob = DewdropNetwork(m32, N; input = 0.0f0, tspan = (0.0, 1000.0),
                      projections = (Projection(CurrentSynapse(; τ = 5.0f0), conn),),
                      arch = GPU())

sol = solve(prob, FixedStep(0.1f0))   # Float32 dt keeps the propagator single-precision end to end
```

Float32 state with `Int32` indices runs roughly 1.7--2.4× faster than Float64 and matches the Float64
dynamics to within ~5%.

## The performance advisor

On a GPU run the [advisor](backends.md) emits a one-off `@info` hint when the
regime suggests a faster path: Float64 state (suggests Float32), 64-bit indices (suggests
`index_type = Int32`), sparse firing over a large connectome (suggests `scatter = :compacted`), or a
small quiet network that is launch-bound (suggests `batch = B`). Silence it with
`Dewdrop.set_advice!(false)` or `solve(...; advise = false)`.

## Progress bars during GPU compilation

`solve` reports progress as [ProgressLogging](https://github.com/JuliaLogging/ProgressLogging.jl)-style
log records (`@logmsg` at `LogLevel(-1)`), which a progress-aware logger such as
`TerminalLoggers.TerminalLogger` renders as a bar. That is independent of the architecture, but there
is a world-age pitfall specific to GPU runs.

During the first kernel compilation, GPUCompiler introspects the **global** logger to gauge its
debug-remark level, through a fixed-world `invoke(Logging.min_enabled_level, Tuple{typeof(logger)}, logger)`.
If the global logger is a *custom* logger (e.g. `TerminalLogger`) whose `min_enabled_level` method was
registered at a world *newer* than the kernel-compile world, that `invoke` throws:

```
MethodError: no method matching invoke min_enabled_level(::TerminalLogger)
  ... method too new to be called from this world context.
```

and compilation dies right after "Building network". (A stdlib `ConsoleLogger` never triggers this: its
`min_enabled_level` lives in the `Logging` stdlib, so it is visible at any compile world.) The crash comes
*only* from a custom logger sitting in the global slot when kernels compile; there are two ways to keep the
bar without it.

### 1. Register the progress logger early (e.g. in `startup.jl`)

Load `TerminalLoggers` and set it global *before* the GPU stack (CUDA / GPUCompiler) loads, so its
`min_enabled_level` method predates the kernel-compile world. Guard on `jl_generating_output` so you never
mutate global logging state while precompiling:

```julia
# ~/.julia/config/startup.jl
if ccall(:jl_generating_output, Cint, ()) == 0 && (isinteractive() || stderr isa Base.TTY)
    Base.eval(Main, quote            # one eval so global_logger sees the fresh import
        using TerminalLoggers, Logging
        global_logger(TerminalLogger())
    end)
end
```

### 2. Keep the global logger world-safe and scope the progress logger

Leave `global_logger()` as a stdlib `ConsoleLogger` (always visible to GPUCompiler) and wrap the run in
`with_logger`. GPUCompiler reads the safe global logger; `solve`'s progress records dispatch to the scoped
`TerminalLogger`, so the bar still renders:

```julia
using Logging, TerminalLoggers
with_logger(TerminalLogger()) do
    solve(prob, FixedStep(0.1); progress = :auto)
end
```

Approach 2 is robust by construction (it never puts a custom logger in the global slot); approach 1 is
convenient when you always want the bar and are willing to register the logger first. Note that
`invokelatest` on the `solve` call does *not* help: GPUCompiler captures the compile world through a
generated-function path, not the runtime world.
