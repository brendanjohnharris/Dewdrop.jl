# * Progress reporting (host-side) for the `solve` time loop.
#
# A progress bar here is just a LOGGING CONVENTION --- the ProgressLogging protocol: emit a
# `@logmsg LogLevel(-1) name progress=frac _id=id` per update, keyed to one stable `id`. VSCode,
# TerminalLoggers, Pluto and Juno all render such records as a live bar with an ETA; a bare
# `julia script.jl` (default ConsoleLogger, min level Info) filters them cheaply, so the loop stays
# silent and near-free off the interactive path. We therefore add NO dependency: stdlib `Logging`
# (the producer side is a convention, not the ProgressLogging package) + `UUIDs` suffice.
#
# The user-facing `progress` kwarg on `solve` / `init`:
#   :auto (default) → on, but SUPPRESSED until ~`_PROGRESS_CALIBRATE` s of wall-clock have elapsed
#                     (so trivial runs never flash a bar); the update stride is calibrated to ~10 Hz.
#   true            → force on from the start (also renders during the calibration window).
#   false           → off (no reporter is built; every loop hook compiles to a no-op).
#   name::String    → force on with a custom bar name.
#   N::Integer (>0) → force on, update every N steps (skip calibration).
#
# The reporter is built at the TOP of `solve!` (not at `init`), so the integrator stores only the
# cheap immutable spec and never carries a UUID across an `adapt` to the device.

import UUIDs
using Logging: LogLevel, @logmsg

const _PROGRESS_LEVEL = LogLevel(-1)      # the ProgressLogging level (a progress record sits below Info)
const _PROGRESS_HZ = 10                   # target update rate (Hz) the calibrated stride aims for
const _PROGRESS_CALIBRATE = 0.3           # calibration window == :auto suppression window (s)
const _PROGRESS_NAME = "Simulating"       # default bar name

# Mutable loop state for one bar. `every`/`next_emit`/`t0`/`calibrating`/`shown` advance during the run.
mutable struct ProgressReporter
    const id::UUIDs.UUID
    const name::String
    const total::Int                      # total steps (the fraction denominator)
    const show_during_calib::Bool         # render during the calibration window? (true unless :auto)
    t0::Float64                           # wall-clock start (set in `_progress_start!`)
    every::Int                            # steady-state stride (steps between updates)
    next_emit::Int                        # next step index `n` at which to act
    calibrating::Bool                     # in the exponential-probe calibration phase?
    shown::Bool                           # has any update been emitted? (gates the completion record)
end

# --- build a reporter from the user spec, or `nothing` (no bar). `Bool` is matched BEFORE `Integer`
# so `true`/`false` never fall through to the cadence path (Bool <: Integer in Julia). ---
_progress_reporter(spec, total::Integer) = _progress_reporter(spec, Int(total))
_progress_reporter(::Nothing, ::Int) = nothing
_progress_reporter(spec::Bool, total::Int) =
    spec ? _build_reporter(_PROGRESS_NAME, total; calibrate = true, show_during_calib = true) : nothing
_progress_reporter(spec::AbstractString, total::Int) =
    _build_reporter(String(spec), total; calibrate = true, show_during_calib = true)
function _progress_reporter(spec::Symbol, total::Int)
    spec === :auto && return _build_reporter(_PROGRESS_NAME, total; calibrate = true, show_during_calib = false)
    throw(ArgumentError("progress::Symbol must be :auto (got :$spec); use true/false, a name String, or a step-count Int"))
end
function _progress_reporter(spec::Integer, total::Int)
    spec < 0 && throw(ArgumentError("progress step count must be ≥ 0 (got $spec)"))
    spec == 0 && return nothing
    return _build_reporter(_PROGRESS_NAME, total; calibrate = false, every = Int(spec), show_during_calib = true)
end

function _build_reporter(name::String, total::Int; calibrate::Bool, show_during_calib::Bool, every::Int = 1)
    total > 0 || return nothing                          # no steps → nothing to report
    # calibrating: probe at n = 1,2,4,…; explicit cadence: fixed `every`, first emit at `every`.
    first_emit = calibrate ? 1 : every
    return ProgressReporter(UUIDs.uuid4(), name, total, show_during_calib, 0.0, every, first_emit, calibrate, false)
end

# --- loop hooks. A `nothing` reporter makes every hook a no-op (so `progress = false` is zero-cost). ---

@inline _progress_start!(::Nothing) = nothing
@inline _progress_start!(rep::ProgressReporter) = (rep.t0 = time(); nothing)

# The per-step hook: its ONLY cost on the common path is the `n >= rep.next_emit` compare (a single,
# well-predicted branch). `time()` and `@logmsg` are touched only at emit points --- O(log n) during
# calibration (probes at powers of two) and ~10 Hz thereafter.
@inline _progress_step!(::Nothing, ::Integer) = nothing
function _progress_step!(rep::ProgressReporter, n::Integer)
    n >= rep.next_emit || return nothing
    if rep.calibrating
        elapsed = time() - rep.t0
        if elapsed >= _PROGRESS_CALIBRATE
            rate = n / elapsed                           # measured steps per second
            rep.every = max(1, round(Int, rate / _PROGRESS_HZ))
            rep.calibrating = false
            rep.next_emit = n + rep.every
            _progress_emit!(rep, n)                      # first visible update (also un-suppresses :auto)
        else
            rep.next_emit = max(n + 1, 2n)               # exponential probe: next sample at 2n
            rep.show_during_calib && _progress_emit!(rep, n)
        end
    else
        rep.next_emit = n + rep.every
        _progress_emit!(rep, n)
    end
    return nothing
end

@inline _progress_finish!(::Nothing) = nothing
function _progress_finish!(rep::ProgressReporter)
    rep.shown || return nothing                          # an auto-suppressed trivial run stayed silent
    @logmsg _PROGRESS_LEVEL rep.name progress=1.0 _id=rep.id   # fraction ≥ 1 ⇒ completed (closes the bar)
    return nothing
end

@inline function _progress_emit!(rep::ProgressReporter, n::Integer)
    rep.shown = true
    frac = clamp(n / rep.total, 0.0, 1.0)
    @logmsg _PROGRESS_LEVEL rep.name progress=frac _id=rep.id
    return nothing
end

# Total steps for a (possibly resumed) integrator: round((tend - tstart)/dt), with tstart = t - n·dt.
# Both `DewdropIntegrator` and `BatchedIntegrator` expose `t`, `n`, `dt`, `tend`.
@inline _progress_total(integ) = round(Int, (integ.tend - (integ.t - integ.n * integ.dt)) / integ.dt)
