# * Within-step schedule
# The phase order is carried in the Schedule's TYPE PARAMETER, so the engine dispatches
# phases at COMPILE TIME: the generated `run_phases!` (in Engine.jl) unrolls to a
# straight-line sequence of `run_phase!(Val(:phase), integ)` calls with no runtime Symbol
# comparison and no dynamic dispatch. The pinned default order remains the inspectable,
# equality-comparable default. Borrowed from Brian2's named-slot scheduler.

"""
    Schedule(phases...)

An ordered, inspectable sequence of within-step phase names (`Symbol`s), carried in the
type so the engine can unroll them at compile time. The engine executes the phases in
order each fixed-`dt` step. Two schedules are equal iff their phase sequences are equal
(order matters).
"""
struct Schedule{P} end
Schedule(phases::Tuple{Vararg{Symbol}}) = Schedule{phases}()
Schedule(phases::Symbol...) = Schedule(phases)
export Schedule

"""
    phases(schedule)

The tuple of phase names, in execution order (a compile-time constant).
"""
phases(::Schedule{P}) where {P} = P

Base.length(::Schedule{P}) where {P} = length(P)
Base.:(==)(::Schedule{P}, ::Schedule{Q}) where {P, Q} = P == Q

# Canonical within-step order:
#   deliver   --- move due conductance increments from the delay ring buffer into inputs
#   integrate --- advance neuron + synapse state by dt (exact linear propagator)
#   threshold --- detect threshold crossings into a spike mask (respecting refractory)
#   reset     --- reset V, arm refractory, apply spike-triggered adaptation increment
#   propagate --- scatter each spike into the delay ring buffer at (now + delay)
#   record    --- snapshot monitored variables
const DEFAULT_PHASES = (:deliver, :integrate, :threshold, :reset, :propagate, :record)

"""
    default_schedule()

The canonical within-step [`Schedule`](@ref): `deliver → integrate → threshold →
reset → propagate → record`.
"""
default_schedule() = Schedule(DEFAULT_PHASES)
