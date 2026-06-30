```@meta
CurrentModule = Dewdrop
DocTestSetup = quote
    using Dewdrop
end
```

# Neuron and synapse models

A model in Dewdrop is a small isbits parameter struct plus pure, scalar, allocation-free functions
for its dynamics, threshold and reset --- "model as code". The same struct runs unchanged on the CPU
or GPU and across every [backend](backends.md); only the parameters live in it, so a model is
trivially copyable, `Adapt`-movable, and reproducible. This page tours the built-in zoo.

Each model declares its per-unit state variables (the struct-of-arrays column names). Subthreshold
dynamics use the **exact linear propagator** (Rotter--Diesmann) for the linear part, kept distinct
from the discontinuous spike reset; nonlinear or adaptation terms are layered on as a forcing current
or an extra conductance over that propagator.

## Neuron models

| model | extra state | nonlinearity | adaptation | typical use |
|---|---|---|---|---|
| [`LIF`](@ref) | --- | none (exact) | none | the default integrate-and-fire unit |
| [`AdaptLIF`](@ref) | `w` | none (linear) | current `w` | spike-frequency adaptation, linear |
| [`AdEx`](@ref) | `w` | exponential spike | current `w` | bursting / sharp spike initiation |
| [`FNSNeuron`](@ref) | `w` (holds `gK`) | none (exact COBA) | conductance `gK` | conductance-adaptation LIF (WRCircuit) |

All carry `V` and `refrac` (the refractory countdown); the adaptation models add one auxiliary column.
[`statevars`](@ref) reports a model's per-unit state columns:

```jldoctest
julia> Dewdrop.statevars(LIF)
(:V, :refrac)

julia> Dewdrop.statevars(AdEx)
(:V, :refrac, :w)
```

### LIF

[`LIF`](@ref) is the leaky integrate-and-fire unit, integrated by its exact propagator over the step.

```math
\tau \frac{dV}{dt} = -(V - E_L) + R\,I,\qquad V \ge V_\theta \Rightarrow V \leftarrow V_r
```

```julia
LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 2.0)
```

Spike when `V ≥ Vθ`, reset to `Vr`, then hold for the absolute refractory period `tref`. State: `V`,
`refrac`.

### AdaptLIF

[`AdaptLIF`](@ref) adds a linear spike-triggered adaptation **current** `w`: an outward current that
relaxes toward `a·(V - EL)` and jumps by `b` on each spike.

```math
\tau \frac{dV}{dt} = -(V - E_L) + R\,(I - w),\qquad
\tau_w \frac{dw}{dt} = a\,(V - E_L) - w
```

```julia
AdaptLIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 2.0,
         a = 0.0, b = 0.06, τw = 100.0)
```

`a` is the subthreshold adaptation conductance, `b` the per-spike current increment, `τw` the
adaptation time constant. State: `V`, `refrac`, `w`.

### AdEx

[`AdEx`](@ref) (Brette--Gerstner) adds an exponential spike-initiation term, so `V` diverges past a
numerical cutoff `Vpeak` rather than crossing a hard threshold at `VT`.

```math
C \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T \exp\!\left(\frac{V - V_T}{\Delta_T}\right) + I - w,
\qquad \tau_w \frac{dw}{dt} = a\,(V - E_L) - w
```

```julia
AdEx(; C = 281.0, gL = 30.0, EL = -70.6, VT = -50.4, ΔT = 2.0, Vr = -70.6,
     Vpeak = -40.0, a = 4.0, b = 0.0805, τw = 144.0, tref = 0.0)
```

The exponential term is treated as a forcing current at the pre-step `V` (exponential-Euler), the
linear part uses the exact propagator (`R = 1/gL`, `τ = C/gL`). The unit fires at the cutoff `Vpeak`
(where the exp has diverged), not at `VT`; on a spike, `V ← Vr` and `w ← w + b`. State: `V`, `refrac`,
`w`.

### FNSNeuron

[`FNSNeuron`](@ref) is a conductance-adaptation LIF: the adaptation here is a **conductance** `gK`
with reversal `VK` (not a current), folded into the exact COBA propagator as an extra leak plus
reversal drive. This is the unit used in the WRCircuit model.

```math
C \frac{dV}{dt} = -g_L(V - V_L) - g_K(V - V_K) + I,\qquad
\tau_K \frac{dg_K}{dt} = -g_K
```

```julia
FNSNeuron(; C = 0.25, gL = 0.0167, VL = -70.0, VK = -85.0, Vθ = -50.0, Vr = -60.0,
          tref = 4.0, τK = 80.0, ΔgK = 0.01)
```

Spike when `V ≥ Vθ`, reset to `Vr`, then `gK ← gK + ΔgK`. Setting `ΔgK = 0` gives a plain
conductance-LIF (used for the inhibitory population). The membrane rests at the leak reversal `VL`.
State: `V`, `refrac`, `w` (the generic auxiliary column holds `gK`).

## Heterogeneity

Two orthogonal mechanisms vary models across a population.

### Per-neuron parameters

[`Heterogeneous`](@ref) wraps one scalar model and overrides chosen parameter fields with per-neuron
arrays (each of length `N`); fields left out keep their scalar value. The engine resolves a scalar
base model per neuron in the fused kernel, so every other path is unchanged.

```julia
N = 1000
base = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 2.0)
het  = Heterogeneous(base; Vθ = per_neuron(i -> -50.0 + 2.0 * sin(i), N))
```

[`per_neuron`](@ref) materialises the array `[f(i) for i in 1:N]`; for a block E/I split use
`vcat(fill(xE, NE), fill(xI, NI))`, or draw from the counter RNG for a reproducible distribution. A
`Heterogeneous` model requires the canonical schedule and runs via the fused megakernel (so
[`Auto`](@ref) selects [`Fused`](@ref)).

### Mixed model types

[`MultiModel`](@ref) holds an ordered set of `(model, size)` groups over one flat concatenated SoA, so
a network can mix neuron model **types** (e.g. AdEx excitatory and LIF inhibitory) in a single engine.
Groups partition `1:N` contiguously in declaration order and must share one float type.

```julia
exc = AdEx(; C = 281.0, gL = 30.0, EL = -70.6, VT = -50.4, ΔT = 2.0, Vr = -70.6,
           Vpeak = -40.0, a = 4.0, b = 0.0805, τw = 144.0)
inh = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 2.0)
mm  = MultiModel([exc, inh], [800, 200])
```

The state is a union SoA (the union of every group's columns); each group's kernel touches only the
columns its own model declares, so a V-only group keeps its fast path. The [`network`](networks.md)
builder constructs a `MultiModel` automatically when populations of distinct model types are added.
Like `Heterogeneous`, it runs through the fused per-group launch.

To define a model type beyond the zoo, see [`@neuron`](@ref).

## Synapse models

A synapse model defines how a delivered presynaptic spike enters the postsynaptic state and how that
state decays between spikes (an exact exponential propagator, distinct from the spike-triggered jump).
The central distinction is **how the spike couples to the membrane**:

- **CUBA** (current-based): the spike adds to a synaptic *current* that feeds `V` directly, independent
  of `V`. Simple and linear; a strong synapse keeps driving regardless of the membrane potential.
- **COBA** (conductance-based): the spike adds to a *conductance* whose current is `g·(Erev - V)`, so
  the drive shunts as `V` approaches the reversal `Erev`. More biophysical; excitation and inhibition
  are set by `Erev` (e.g. 0 mV excitatory, -80 mV inhibitory) rather than by the sign of the weight.
- **delta**: the spike jumps `V` instantaneously by the weight, with no synaptic time constant.

| synapse | kind | kinetics | current | constructor |
|---|---|---|---|---|
| [`CurrentSynapse`](@ref) | CUBA | single-exp | `I` (direct) | `(; τ)` |
| [`DeltaSynapse`](@ref) | delta | none | instantaneous `V` jump | `()` |
| [`ConductanceSynapse`](@ref) | COBA | single-exp | `g·(Erev - V)` | `(; τ, Erev)` |
| [`DualExpSynapse`](@ref) | COBA | dual-exp | `g·(Erev - V)` | `(; τr, τd, Erev)` |
| [`FrozenDualExpSynapse`](@ref) | COBA | dual-exp | `g·(Erev - V)`, `V` frozen | `(; τr, τd, Erev)` |

```julia
CurrentSynapse(; τ = 5.0)                          # CUBA, decays with τ
DeltaSynapse()                                     # weight IS the PSP amplitude (Brunel 2000)
ConductanceSynapse(; τ = 5.0, Erev = 0.0)          # COBA single-exp, excitatory
DualExpSynapse(; τr = 1.0, τd = 5.0, Erev = 0.0)   # COBA, peak-normalised to the weight
```

[`DualExpSynapse`](@ref) kicks two accumulators (rise `τr`, decay `τd`); the conductance is their
peak-normalised difference, so the delivered weight equals the peak conductance. It requires
`τr ≠ τd`.

[`FrozenDualExpSynapse`](@ref) is a drop-in variant with identical conductance kinetics, but the
current is evaluated with `V` frozen at its pre-update value and injected as an ordinary current; it
does NOT enter the effective leak, so it does not shunt the membrane time constant. It exists to
reproduce the BrainPy/Brian frozen-current integration; prefer the exact [`DualExpSynapse`](@ref)
(the conductance shunts) unless you are matching those simulators.

Choose CUBA for the simplest linear coupling, COBA when the reversal-dependent shunt matters (the
usual choice for biophysical E/I networks), and delta for analytically tractable Brunel-style nets
where the weight is the PSP amplitude directly.

## Switching to single precision

[`convertfloat`](@ref) rebuilds any model, builder, or whole network with every floating-point leaf
converted to a target float type, recursing through structs, tuples, named tuples and arrays.
Integers, symbols and functions (distance kernels, weight adjusters) pass through unchanged.

```julia
net32 = convertfloat(Float32, build(...))
```

This lets a model be written in convenient `Float64` literals and converted afterwards (halving the
state, recorded-trace and connectome footprint), rather than wrapping every parameter by hand. It is
the recommended way to move a whole network onto a [`GPU`](@ref), where Float32 is preferred. The
model's float type follows the conversion:

```jldoctest
julia> m = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 1.0, tref = 2.0);

julia> Dewdrop.float_type(m)
Float64

julia> Dewdrop.float_type(convertfloat(Float32, m))
Float32
```
