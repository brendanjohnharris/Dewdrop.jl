# * Unit-boundary seams.
# Dewdrop's engine works in a fixed, *coherent* canonical float system: time in ms, voltage
# in mV, conductance in nS, current in pA, capacitance in pF, resistance in GΩ, rate in kHz ---
# chosen so the dynamics equations carry no stray numerical factors (e.g. R·I = GΩ·pA = mV,
# R·g = GΩ·nS = 1, R·C = GΩ·pF = ms). Plain numbers passed to the API are assumed to already be
# in this system and pass through untouched.
#
# These functions are the seams the optional `ext/UnitfulExt.jl` overloads: loading `Unitful`
# activates methods that convert + strip a `Quantity` to the canonical unit for its dimension,
# so the API accepts physical units (`τ = 20u"ms"`, `g = 6u"nS"`) while the SoA state stays
# plain isbits floats. Units thus live ONLY at the construction boundary, never in a kernel.
@inline to_time(x) = x
@inline to_voltage(x) = x
@inline to_current(x) = x
@inline to_conductance(x) = x
@inline to_resistance(x) = x
@inline to_capacitance(x) = float(x)   # canonical pF; floats plain numbers (capacitance is continuous)
@inline to_rate(x) = x
@inline to_weight(x) = x        # dimension inferred by the ext (voltage / current / conductance)
