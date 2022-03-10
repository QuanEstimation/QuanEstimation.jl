abstract type AbstractOpt end

struct Opt <: AbstractOpt
    opt_target::Symbol
end

struct ControlOpt <: Opt
    opt_target::Symbol
    control_coefficients::AbstractVector
end

struct StateOpt <: Opt
    opt_target::Symbol
    ψ₀::AbstractVector
end

struct MeasurementOpt <: Opt
    opt_target::Symbol
    measurement::AbstractVector
end

struct CompOpt <: Opt
    opt_target::Symbol
end

struct StateControlOpt <: CompOpt
    opt_target::Symbol
    ψ₀::AbstractVector
    control_coefficients::AbstractVector
end

struct StateMeasurementOpt <: CompOpt
    opt_target::Symbol
    ψ₀::AbstractVector
    measurement::AbstractVector
end

struct StateControlMeasurementOpt <: CompOpt
    opt_target::Symbol
    control_coefficients::AbstractVector
    ψ₀::AbstractVector
    measurement::AbstractVector
end

ControlOpt(ctrl::AbstractVector) = ControlOpt(:Copt, ctrl)
StateOpt(ψ₀::AbstractVector) = StateOpt(:Sopt, ψ₀)
MeasurementOpt(M::AbstractVector) = MeasurementOpt(:Mopt, M)
StateControlOpt(ψ₀::AbstractVector, ctrl::AbstractVector) =
    StateControlOpt(:SCopt, ψ₀, ctrl)
StateMeasurementOpt(ψ₀::AbstractVector, M::AbstractVector) =
    StateMeasurementOpt(:SMopt, ψ₀, M)
StateControlMeasurementOpt(ψ₀::AbstractVector, ctrl::AbstractVector, M::AbstractVector) =
    StateControlMeasurementOpt(:SCMOpt, ψ₀, ctrl, M)
