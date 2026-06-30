```@meta
CurrentModule = Dewdrop
```

# Custom models with @neuron

A neuron model in Dewdrop is "model as code": a small isbits parameter struct plus a handful of
pure, scalar hooks ([`statevars`](@ref), [`float_type`](@ref), a subthreshold propagator,
[`threshold`](@ref), [`reset_value`](@ref), [`refractory`](@ref)). Writing those out by hand is
mechanical for the common case of a *linear* membrane, so [`@neuron`](@ref) generates the whole lot
from a short declaration. You write the hooks as expressions over the model's parameters; the macro
emits the struct, a keyword constructor, and every method.

## The linear-membrane constraint

[`@neuron`](@ref) covers the LIF family: the subthreshold dynamics must be **linear** in `V`, so the
exact propagator (the Rotter--Diesmann exponential step) applies over a time step at constant input.
That is the same step every built-in linear model uses, and it is what lets conductance synapses fold
in as an effective leak. You do not write `dV/dt`; you declare its two consequences directly:

- `@asymptote` --- the steady-state potential `V∞` for total input current `I` (must be linear in `I`).
- `@resistance` --- `dV∞/dI`, the slope coupling [`ConductanceSynapse`](@ref) / COBA input to the membrane.

Nonlinear membranes (Izhikevich's quadratic term, AdEx's exponential spike initiation) cannot be
expressed this way yet; full `dV/dt` parsing is a later front-end. For those, drop to the
[hand-written path](#The-hand-written-path) below.

## Sub-macros

Inside the `@neuron` block, each line is a sub-macro. Parameter symbols in any expression are
rewritten to field accesses; the reserved names `I` (total input current) and `V` (membrane
potential) are left alone, as are functions and literals.

| sub-macro | meaning |
|---|---|
| `@parameters` | parameter names; `name=value` gives a constructor default |
| `@state` | per-unit state column names (must include `V`, plus `refrac` for the refractory clock) |
| `@asymptote` | steady-state `V∞` as a function of total input `I` (linear in `I`) |
| `@resistance` | `dV∞/dI`; couples conductance (COBA) synapses |
| `@timeconstant` | the membrane time constant for the exponential decay |
| `@threshold` | spike predicate over `V`, e.g. `V ≥ Vθ` |
| `@reset` | the post-spike membrane value |
| `@refractory` | absolute refractory duration |

All of `@asymptote`, `@resistance`, `@timeconstant`, `@threshold`, `@reset` and `@refractory` are
required; omitting one errors at macro expansion.

## A worked example

A leaky integrator with a constant bias current folded into the resting drive. The bias shifts the
fixed point but not its slope, so [`@resistance`](@ref) stays `R` (the bias is a constant, not a
function of `I`):

```julia
using Dewdrop

@neuron BiasLIF begin
    @parameters  τ=20.0 EL=-60.0 Vθ=-50.0 Vr=-60.0 R=1.0 tref=5.0 Ibias=0.0
    @state       V refrac
    @asymptote   EL + R * (I + Ibias)     # linear in I; Ibias is a constant offset
    @resistance  R                        # dV∞/dI --- independent of Ibias
    @timeconstant τ
    @threshold   V ≥ Vθ
    @reset       Vr
    @refractory  tref
end
```

The macro returns the type. Construct it with keywords (defaults from `@parameters`); every
parameter promotes to a common float type:

```julia
model = BiasLIF(; τ = 15.0, Ibias = 0.2)     # other parameters take their declared defaults
```

It now behaves like any built-in model --- pass it to [`DewdropNetwork`](@ref) (or the fluent
[builder](networks.md)) and [`solve`](@ref):

```julia
net = DewdropNetwork(model, 64; input = 0.5, tspan = (0.0, 500.0))
sol = solve(net, FixedStep(0.1))
firing_rate(sol)
```

`@neuron` models run on every backend except [`Turbo`](@ref), which needs a per-model SIMD kernel;
see [adding a Turbo specialization](#Turbo-for-a-custom-model).

## The hand-written path

When the membrane is nonlinear, define the hooks directly on a struct `<: AbstractNeuronModel`. The
core never special-cases your type; it dispatches on these methods. For a `V`-only model you provide:

- `statevars(::Type{YourModel})` returning the state column names (a `Tuple` of `Symbol`s);
- [`float_type`](@ref)`(::YourModel)` returning the parameter/state float type;
- `membrane_step(m, V, gtot, itot, dt)` --- one subthreshold step from `V` given accumulated
  conductance `gtot` and total current `itot` (this is where your nonlinear update lives; the linear
  models route through the shared exact-COBA step internally);
- [`threshold`](@ref), [`reset_value`](@ref), [`refractory`](@ref) as for the macro.

```julia
import Dewdrop: AbstractNeuronModel, statevars, float_type, membrane_step,
                threshold, reset_value, refractory

struct QuadIF{T} <: AbstractNeuronModel    # quadratic IF: nonlinear, not @neuron-expressible
    τ::T; Vc::T; Vθ::T; Vr::T; R::T; tref::T
end

statevars(::Type{<:QuadIF}) = (:V, :refrac)
float_type(::QuadIF{T}) where {T} = T

# explicit Euler on τ dV/dt = (V - Vc)^2 + R·I  (your own integrator; nonlinear in V)
@inline membrane_step(m::QuadIF, V, gtot, itot, dt) =
    V + dt * ((V - m.Vc)^2 + m.R * itot) / m.τ

@inline threshold(m::QuadIF, V)  = V ≥ m.Vθ
@inline reset_value(m::QuadIF)   = m.Vr
@inline refractory(m::QuadIF)    = m.tref
```

The step must be pure and allocation-free; it runs per neuron on CPU and per thread on GPU
unchanged. `gtot` is the summed synaptic conductance (use it if you support COBA input; ignore it
for a pure current-based model).

Models that carry an auxiliary state variable (a spike-triggered adaptation current or conductance,
like the built-in [`AdaptLIF`](@ref), [`AdEx`](@ref) and [`FNSNeuron`](@ref)) add a third state column
and hook into an internal aux-state seam (the `w`-first split) rather than `membrane_step` alone. That
seam is not part of the public API; read `src/Adaptation.jl` and copy the closest built-in as a
template.

## Turbo for a custom model

A custom model runs on [`Serial`](@ref), [`Fused`](@ref) and [`Auto`](@ref) with no extra work. To get
the vectorised [`Turbo`](@ref) path you register one method, [`turbo_kernel`](@ref), replicating your
scalar math branch-free. See [Turbo & model specialization](turbo.md) for the kernel idiom and the
constraints (canonical schedule, no [`WhiteNoise`](@ref), CPU only).
