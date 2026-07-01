module UnitfulExt

# Optional Unitful boundary: physical-unit inputs (`τ = 20u"ms"`, `g = 6u"nS"`) are converted
# and stripped to Dewdrop's coherent canonical float system at construction, so the engine state
# stays plain isbits floats (the GPU contract). Activated automatically by `using Unitful`.
#
# Canonical system (coherent: the dynamics carry no stray factors):
#   time ms · voltage mV · conductance nS · current pA · capacitance pF · resistance GΩ · rate kHz
#   R·I = GΩ·pA = mV   ·   R·g = GΩ·nS = 1   ·   R·C = GΩ·pF = ms   ·   rate·dt = kHz·ms = 1
#
# `ustrip(unit, x)` converts `x` to `unit` and drops it, THROWING a `DimensionError` if `x` has
# the wrong dimension, so a mis-dimensioned input is rejected at the boundary. `float` keeps
# the canonical system in floating point even for integer-valued quantities.

using Dewdrop
using Unitful: Unitful, Quantity, ustrip, dimension, @u_str

@inline Dewdrop.to_time(x::Quantity) = float(ustrip(u"ms", x))
@inline Dewdrop.to_voltage(x::Quantity) = float(ustrip(u"mV", x))
@inline Dewdrop.to_current(x::Quantity) = float(ustrip(u"pA", x))
@inline Dewdrop.to_conductance(x::Quantity) = float(ustrip(u"nS", x))
@inline Dewdrop.to_resistance(x::Quantity) = float(ustrip(u"GΩ", x))
@inline Dewdrop.to_capacitance(x::Quantity) = float(ustrip(u"pF", x))
@inline Dewdrop.to_rate(x::Quantity) = float(ustrip(u"kHz", x))

# Per-neuron input may be a unitful array.
@inline Dewdrop.to_current(x::AbstractArray{<:Quantity}) = float.(ustrip.(u"pA", x))

# A synaptic weight's role (voltage jump / current / conductance) is fixed by its OWN dimension.
function Dewdrop.to_weight(x::Quantity)
    d = dimension(x)
    d === dimension(u"mV") && return float(ustrip(u"mV", x))   # delta-synapse voltage jump
    d === dimension(u"pA") && return float(ustrip(u"pA", x))   # CUBA current
    d === dimension(u"nS") && return float(ustrip(u"nS", x))   # COBA conductance
    throw(
        ArgumentError(
            "synaptic weight has dimension $d; expected a voltage (delta), " *
                "current (CUBA), or conductance (COBA)"
        )
    )
end

end # module UnitfulExt
