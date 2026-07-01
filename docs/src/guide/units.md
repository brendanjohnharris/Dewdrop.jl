```@meta
CurrentModule = Dewdrop
```

# Units

Dewdrop's engine runs in a single fixed *coherent* float system. Every quantity in a model is a
plain number understood to already be in these units:

| dimension | canonical unit |
|---|---|
| time | ms |
| voltage | mV |
| conductance | nS |
| current | pA |
| capacitance | pF |
| resistance | GΩ |
| rate | kHz |

The system is *coherent*: the units combine with no stray numerical factors, so the dynamics
equations are written exactly as the maths. With `R` in GΩ and `I` in pA,

```
R·I = GΩ·pA = mV       R·g = GΩ·nS = 1       R·C = GΩ·pF = ms       rate·dt = kHz·ms = 1
```

Choosing the units this way is what lets the hot loop stay free of conversion constants; an LIF
step is `τ dV/dt = -(V - EL) + R·I` with no `1e-3` or `1e9` anywhere.

## Writing a model in plain floats

The canonical system is the default. Pass numbers and they pass through untouched:

```julia
using Dewdrop

lif = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 2.0)
```

That is `τ = 20 ms`, `EL = -70 mV`, `R = 0.1 GΩ` (= 100 MΩ), `tref = 2 ms`. The same convention
holds everywhere a parameter is accepted: synapse time constants and reversal potentials, drive
rates and weights, the integration step in [`FixedStep`](@ref), and the network's `tspan`.

## The Unitful boundary

Loading `Unitful` activates an extension that lets you write parameters as physical quantities.
They are converted to the canonical unit for their dimension and stripped to a plain float at the
construction boundary, so the stored model is byte-for-byte identical to the plain-float one:

```julia
using Dewdrop
using Unitful

lif = LIF(; τ = 20u"ms", EL = -70u"mV", Vθ = -50u"mV", Vr = -60u"mV",
          R = 100u"MΩ", tref = 2u"ms")

adex = AdEx(; C = 281u"pF", gL = 30u"nS", EL = -70.6u"mV", VT = -50.4u"mV",
            ΔT = 2u"mV", Vr = -70.6u"mV", Vpeak = 20u"mV",
            a = 4u"nS", b = 0.0805u"nA", τw = 144u"ms", tref = 2u"ms")
```

`100u"MΩ"` becomes `0.1` (GΩ) and `0.0805u"nA"` becomes `80.5` (pA); you never spell out the
conversion. Conversion happens through a small set of seams (`to_time`, `to_voltage`,
`to_current`, `to_conductance`, `to_resistance`, `to_capacitance`, `to_rate`, `to_weight`) that are
the identity on plain numbers and dimension-converting on `Quantity` inputs. Two consequences:

- A wrong dimension is rejected at construction. `τ = 20u"mV"` throws a `DimensionError` rather
  than silently producing garbage.
- Units never enter a kernel. They live only at the API boundary; the SoA state, connectome, and
  recorded traces are always plain isbits floats (the GPU requirement).

A synaptic weight has no fixed dimension of its own (it is a voltage jump for a delta synapse,
a current for CUBA, a conductance for COBA), so `to_weight` reads the weight's *own*
dimension to decide. A `Quantity` in mV, pA, or nS is accepted; anything else is an error.

```julia
# COBA conductance weight, given in nS; CUBA would use pA, a delta synapse mV
conn = fixed_prob(CPU(), 800, 800, 0.02; weight = 6u"nS", delay = 0.0, seed = 0x1)
```

Mixing styles is fine: a unitful `τ` and a plain-float `R` in the same constructor both end up as
canonical floats.

## Switching precision with `convertfloat`

Models are usually written in `Float64` literals for convenience, but the state, connectome, and
trace buffers are often better stored as `Float32` (half the memory, faster on GPU).
[`convertfloat`](@ref) rebuilds a whole object at a new float type without touching the
parameters by hand:

```julia
net32 = convertfloat(Float32, build(...))
```

It recurses through structs, `Tuple`s, `NamedTuple`s, and arrays, converting every
`AbstractFloat` leaf to `T`. Integers, booleans, symbols, strings, ranges, and functions (distance
kernels, weight adjusters) pass through unchanged. So a single call switches a neuron or synapse
model, a builder from [building networks](networks.md), or a deferred network spec to `Float32`:

```julia
lif64 = LIF(; τ = 20.0, EL = -70.0, Vθ = -50.0, Vr = -60.0, R = 0.1, tref = 2.0)
lif32 = convertfloat(Float32, lif64)   # all parameters now Float32
```

`convertfloat` and the Unitful boundary compose: write the model in `Float64` Unitful quantities
for readability, then drop the whole thing to `Float32` for the run.
