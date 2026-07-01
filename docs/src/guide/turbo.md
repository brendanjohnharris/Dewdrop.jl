```@meta
CurrentModule = Dewdrop
```

# Turbo & model specialization

The [`Turbo`](@ref) backend is the fastest CPU path: it vectorises the dense per-neuron update
(notably the membrane `exp`) with [LoopVectorization](https://github.com/JuliaSIMD/LoopVectorization.jl)'s
`@turbo`, reaching compiled-C++ throughput. It is **opt-in** because it adds the LoopVectorization
dependency and is not bit-identical to the other backends (SIMD `exp` differs from scalar `libm` at
the ULP level: results are *spike*-identical, not byte-identical).

```julia
using Dewdrop
using LoopVectorization          # activates the DewdropLoopVectorizationExt extension

solve(prob, FixedStep(0.1); backend = Turbo())
```

`Turbo` only the **dense** phase is vectorised; the synaptic deliver / accumulate / decay and the
sparse scatter stay scalar (ring buffers and the projection tuple do not vectorise). So `Turbo` wins
most when the per-neuron compute dominates the synaptic work (dense-dominated networks); when the
network is synapse-heavy, the bit-identical [`Fused`](@ref) backend is usually just as fast.

## Which built-in models support Turbo

A model supports `Turbo` iff it provides a [`turbo_kernel`](@ref) specialization (registered by the
extension). Requesting `Turbo()` for an unsupported model errors at `init` with a clear message.

| model | Turbo specialization | notes |
|---|:--:|---|
| [`LIF`](@ref) | ✅ | CUBA & COBA (reads `gtot`) |
| [`AdEx`](@ref) | ✅ | the exponential-IF membrane; CUBA & COBA |
| `AdaptLIF` | ❌ | not yet; use `Fused` (contributions welcome) |
| `FNSNeuron` | ❌ | not yet; use `Fused` |
| `Heterogeneous` / `MultiModel` | ❌ | per-neuron / per-group resolution is not vectorised; use `Fused` |
| `@neuron` models | ❌ by default | add a specialization (below) |

Constraints for the `Turbo` backend (checked at `init`): the **canonical schedule**, **no
`WhiteNoise`** (the SIMD kernel is deterministic), and the CPU architecture.

## Adding a Turbo specialization to a model

A model author opts in by defining one method: [`turbo_kernel`](@ref) returns a function that runs the
SIMD dense update for that model type. The orchestration (deliver / accumulate / decay / scatter)
stays in the core; you write only the per-neuron membrane/threshold/reset kernel.

The kernel receives the integrator. Read the accumulated current/conductance from `integ.itot` /
`integ.gtot` (filled by the core before the kernel runs), and write the SoA state. Keep the body
**branch-free** (`ifelse`, `min`/`max`) so `@turbo` can vectorise it, and **replicate the model's
scalar math exactly** so it stays spike-identical.

```julia
using LoopVectorization
import Dewdrop: turbo_kernel

# register it
turbo_kernel(::Type{MyLIF}) = _turbo_mylif!

function _turbo_mylif!(integ)
    st = integ.state.state
    V, refrac, spiked, spk = st.V, st.refrac, integ.spiked, integ.spike_count
    itot, gtot, dt = integ.itot, integ.gtot, integ.dt
    m = integ.model::MyLIF
    z = zero(eltype(V))
    @turbo for i in eachindex(V)
        v = V[i]; r = refrac[i]; it = itot[i]; gt = gtot[i]
        denom = 1 + m.R * gt
        V∞  = (m.EL + m.R * it) / denom
        v_adv = V∞ + (v - V∞) * exp(-dt * denom / m.τ)     # the exact propagator
        v1 = ifelse(r > z, m.Vr, v_adv)                    # refractory clamp
        r1 = max(r - dt, z)
        s  = (r1 ≤ z) & (v1 ≥ m.Vθ)                        # threshold
        V[i]      = ifelse(s, m.Vr, v1)
        refrac[i] = ifelse(s, m.tref, r1)
        spiked[i] = s
        spk[i]   += s                                      # always-on spike count
    end
    return nothing
end
```

For a model with an auxiliary state variable (like AdEx's `w`), advance it inside the same loop in the
engine's order (the "w-first" split for the adaptation models). See the built-in `AdEx`/`LIF` kernels
in `ext/DewdropLoopVectorizationExt.jl` as templates.

Validate by comparing to the `Serial` baseline: the spike counts must match exactly and `V` to a
small tolerance (`test/brian/turbo_check.jl`).

```@docs
turbo_kernel
```
