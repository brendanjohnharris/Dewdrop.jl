# * @neuron macro --- declarative sugar for defining a linear (LIF-family) neuron model
# without the struct + hook boilerplate. The user writes hook expressions over the model's
# parameters plus the reserved names `I` (total input current) and `V` (membrane potential);
# the macro rewrites every parameter symbol to a field access `m.<param>` (no equation parsing,
# so it is robust to how the expressions are written) and emits the parameter struct plus all
# the model hooks (`statevars`, `float_type`, `asymptote`, `membrane_step`, `threshold`,
# `reset_value`, `refractory`). The membrane must be LINEAR (the exact propagator applies);
# nonlinear models (Izhikevich/AdEx) and full `dV/dt` equation parsing are a later front-end.
#
#   @neuron MyLIF begin
#       @parameters  τ=20.0 EL=-60.0 Vθ=-50.0 Vr=-60.0 R=1.0 tref=5.0
#       @state       V refrac
#       @asymptote   EL + R * I        # steady-state V for total input current I (linear in I)
#       @resistance  R                  # ∂V∞/∂I --- couples conductance (COBA) synapses
#       @timeconstant τ
#       @threshold   V ≥ Vθ
#       @reset       Vr
#       @refractory  tref
#   end

# Generic linear, COBA-capable subthreshold step shared by every @neuron-generated model: a
# conductance `gtot` sets an effective leak `denom`, scaling both the fixed point and the decay
# rate. `V∞I` is the current-driven asymptote `asymptote(m, itot)`; with `gtot = 0` this is the
# plain exact propagator. (LIF predates the macro and keeps its own equivalent `_coba_step`.)
@inline function _linear_membrane_step(V∞I, R, τ, V, gtot, dt)
    denom = 1 + R * gtot
    V∞ = V∞I / denom
    return V∞ + (V - V∞) * exp(-dt * denom / τ)
end

# Rewrite parameter symbols to `m.<param>` field accesses; leave reserved args (V, I) and
# everything else (functions, literals) untouched.
_subparams(ex, ::Vector{Symbol}) = ex
_subparams(ex::Symbol, params::Vector{Symbol}) = ex in params ? :(m.$ex) : ex
_subparams(ex::Expr, params::Vector{Symbol}) = Expr(ex.head, map(a -> _subparams(a, params), ex.args)...)

"""
    @neuron Name begin ... end

Define a linear (LIF-family) neuron model from hook expressions, without the struct + interface
boilerplate. The block lists `@parameters`, `@state`, and the membrane hooks `@asymptote`,
`@resistance`, `@timeconstant`, `@threshold`, `@reset` and `@refractory`; parameter symbols in any
expression are rewritten to field accesses `m.<param>`. The membrane must be linear in `V` (the exact
propagator applies). Generated models run on every backend except [`Turbo`](@ref).

    @neuron MyLIF begin
        @parameters   τ=20.0 EL=-60.0 Vθ=-50.0 Vr=-60.0 R=1.0 tref=5.0
        @state        V refrac
        @asymptote    EL + R * I     # steady-state V for total input current I
        @resistance   R              # ∂V∞/∂I --- couples COBA synapses
        @timeconstant τ
        @threshold    V ≥ Vθ
        @reset        Vr
        @refractory   tref
    end
"""
macro neuron(name, block)
    params = Symbol[]
    kwargs = Any[]            # constructor keyword args (with defaults where given)
    states = Symbol[]
    asym = Rparam = τparam = thr = rst = refr = nothing
    for line in block.args
        (line isa Expr && line.head === :macrocall) || continue
        mname = line.args[1]::Symbol
        args = line.args[3:end]                      # past the macro symbol + LineNumberNode
        if mname === Symbol("@parameters")
            for a in args
                if a isa Symbol
                    push!(params, a); push!(kwargs, a)
                elseif a isa Expr && a.head === :(=)
                    push!(params, a.args[1]); push!(kwargs, Expr(:kw, a.args[1], a.args[2]))
                end
            end
        elseif mname === Symbol("@state")
            append!(states, args)
        elseif mname === Symbol("@asymptote")
            asym = args[1]
        elseif mname === Symbol("@resistance")
            Rparam = args[1]
        elseif mname === Symbol("@timeconstant")
            τparam = args[1]
        elseif mname === Symbol("@threshold")
            thr = args[1]
        elseif mname === Symbol("@reset")
            rst = args[1]
        elseif mname === Symbol("@refractory")
            refr = args[1]
        end
    end
    any(isnothing, (asym, Rparam, τparam, thr, rst, refr)) &&
        error("@neuron $name: needs @asymptote, @resistance, @timeconstant, @threshold, @reset and @refractory")

    fields = [:($p::T) for p in params]
    statetuple = Expr(:tuple, QuoteNode.(states)...)
    sa, st, sr, sf = _subparams(asym, params), _subparams(thr, params),
        _subparams(rst, params), _subparams(refr, params)

    return esc(quote
        struct $name{T} <: Dewdrop.AbstractNeuronModel
            $(fields...)
        end
        $name(; $(kwargs...)) = $name(Base.promote($(params...))...)
        Dewdrop.statevars(::Type{<:$name}) = $statetuple
        Dewdrop.float_type(::$name{T}) where {T} = T
        @inline Dewdrop.asymptote(m::$name, I) = $sa
        @inline Dewdrop.membrane_step(m::$name, V, gtot, itot, dt) =
            Dewdrop._linear_membrane_step(Dewdrop.asymptote(m, itot), m.$Rparam, m.$τparam, V, gtot, dt)
        @inline Dewdrop.threshold(m::$name, V) = $st
        @inline Dewdrop.reset_value(m::$name) = $sr
        @inline Dewdrop.refractory(m::$name) = $sf
        $name
    end)
end
export @neuron
