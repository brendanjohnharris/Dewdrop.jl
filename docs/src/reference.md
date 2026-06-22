```@meta
CurrentModule = Dewdrop
```

# Reference

## Building & solving

```@docs
DewdropNetwork
FixedStep
Projection
solve
init
network
population!
project!
drive!
build
```

## Neuron models

```@docs
LIF
AdEx
AdaptLIF
FNSNeuron
Heterogeneous
MultiModel
per_neuron
```

## Synapses

```@docs
CurrentSynapse
ConductanceSynapse
DeltaSynapse
DualExpSynapse
```

## Connectivity & positions

```@docs
SparseCSR
fixed_prob
distance_prob
distance_fixed_count
gaussian_kernel
exponential_kernel
box_kernel
line_positions
grid_positions
ring_positions
random_positions
```

## Recording & outputs

```@docs
Trace
Spikes
Aggregate
Probe
DewdropSolution
SubSolution
firing_rate
raster
duration
```

## Statistical observables

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

## Advisor

```@docs
set_advice!
```
