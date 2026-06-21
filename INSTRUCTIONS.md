Ok I want to create a julia spiking neural network called 'Dewdrop.jl'. I've included a template project showcasing my code structure and style in this current directory.

The idea is that the spiking neural circuit will consolidate ideas form existing gold-standard simulators in pyhton, c++, etc into a fast, performant, but above all intuitive simulation framework in julia.

So to design this system, you should become familiar witht he strengths and weaknesses of the following precedents:
- Brian2 (Python)
- NEST (C++)
- NEURON (C++)
- BrainPy (Python)

Brainpy in particular is particularly impotnat, because it has addressed some of the issues needed to run spiking circuits on GPUs.


You should also familiarize yourself with the tools available in julia to make simulation fas, efficient, and intuitive:
- DifferentialEquations.jl
- ModellingToolkit.jl
- CUDA.jl

And others you can find that are related; prefer to follow SciML best preactices, package usage, and code style guidelines.

So the most importnat poitns to design around are:
- Intuitive API for defining spiking neural circuits, including neuron models, synapse models, and network architectures.
- Efficient simulation engine that can leverage CPU and GPU resources effectively. Extract max. performance (agaisnt simplicity tradeoff) form GPU acceleration
- Support for a wide range of neuron and synapse models, but we start with LIF neurons and simple synapse models. Eventually we will want a simple interface for defining new neuron models (as close to code as possible) and also for defining new connectivity structures/modules/hierarchical structures (including spatial connectivity /2d, 3d, ring embedidngs etc.)

You should look at the code in `../DDC/WorkingRegime.jl/WRCircuit.jl/' to understand the sorts of networks I want to target (at minimum). The idea is to be able to easily define and simulate circuits like this, but also to be able to easily define more complex circuits with more complex neuron and synapse models, and more complex connectivity structures.

Start by reading widely, understanding the entire state of the field, and distill best practices and opportunities fo rimporvement into a clear, cohesive, actionable, and testable sequential design plan.

