```@meta
CurrentModule = Dewdrop
```

# Reference

The complete public API, grouped by role. The execution backends
([`Auto`](@ref)/[`Serial`](@ref)/[`Fused`](@ref)/[`Turbo`](@ref)) and [`turbo_kernel`](@ref) are
documented in [Choosing a backend](guide/backends.md) and [Turbo & model
specialization](guide/turbo.md).

## Architecture

`arch` selects *where* the state lives; the [backend](guide/backends.md) selects *how* each step runs.

```@docs
CPU
GPU
```

## Building & solving

```@docs
DewdropNetwork
FixedStep
Projection
PoissonDrive
solve
init
step!
solve!
```

## The network builder

The fluent way to assemble multi-population networks with named subpopulations; see [building
networks](guide/networks.md).

```@docs
network
population!
project!
drive!
build
Dewdrop.NetworkBuilder
```

## Deferred network specs

A network described once and materialised at solve time; see [building networks](guide/networks.md).

```@docs
AbstractNetworkSpec
freeze
defer
materialize
```

## Neuron models

The model zoo and per-neuron / multi-type populations; see [neuron & synapse models](guide/models.md)
and [custom models](guide/neuron-macro.md).

```@docs
LIF
AdaptLIF
AdEx
FNSNeuron
Heterogeneous
per_neuron
MultiModel
convertfloat
@neuron
```

### Model interface (for model authors)

The hooks a custom neuron model provides (directly or via [`@neuron`](@ref)).

```@docs
AbstractNeuronModel
statevars
float_type
asymptote
propagator_decay
subthreshold_step
```

## Synapses

```@docs
AbstractSynapseModel
synapse_decay
CurrentSynapse
DeltaSynapse
ConductanceSynapse
DualExpSynapse
FrozenDualExpSynapse
```

## Connectivity

```@docs
SparseCSR
fixed_prob
steps
correlate_weights
correlate_weights!
```

## Positions & distance kernels

```@docs
line_positions
grid_positions
ring_positions
random_positions
gaussian_kernel
exponential_kernel
box_kernel
distance_prob
distance_fixed_count
```

## External drive & noise

```@docs
PoissonSource
WhiteNoise
```

## Scheduling

```@docs
Schedule
```

## Plasticity

```@docs
STDP
```

## Differentiable backend

The surrogate-gradient backend; see [differentiable simulation & training](guide/training.md).

```@docs
Differentiable
```

## Recording

The `record = (...)` monitors; see [recording & outputs](guide/recording.md).

```@docs
Trace
Spikes
Aggregate
Probe
```

### On-device temporal reducers

Streaming temporal statistics computed during the run and fused into recording.

```@docs
MADev
Welch
SpikeRate
Fano
```

## Solutions & outputs

```@docs
DewdropSolution
SubSolution
firing_rate
raster
duration
```

## Batching

Ensemble (tensor) and block-diagonal batching; see [batching & ensembles](guide/batching.md).

```@docs
batch
NetworkBatch
nmembers
BatchedSolution
BatchSolution
```

## Statistical observables

Host-side spatial-network analysis; see [analysis & observables](guide/analysis.md).

```@docs
coarsegrain
susceptibility
mua
temporal_average
grand_distribution
cv_isi
power_spectrum
efficiency
radial_autocorrelation
```

## Performance advisor

```@docs
set_advice!
```
