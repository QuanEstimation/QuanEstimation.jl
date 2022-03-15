abstract type AbstractOpt end

abstract type AbstractMeasurementType end
abstract type Projection <: AbstractMeasurementType end
abstract type LinearComb <: AbstractMeasurementType end
abstract type Rotation <: AbstractMeasurementType end

abstract type Opt <: AbstractOpt end

mutable struct ControlOpt <: Opt
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
end

mutable struct StateOpt <: Opt
    ψ₀::AbstractVector
end

abstract type AbstractMopt <: Opt end

mutable struct MeasurementOpt{S} <: AbstractMopt
    measurement::AbstractVector
end

mutable struct Mopt_Projection <: AbstractMopt
    measurement::AbstractVector
    C::AbstractVector
end

mutable struct Mopt_LinearComb <: AbstractMopt
    measurement::AbstractVector
    POVM_basis::AbstractVector
    M_num::Number
end

mutable struct Mopt_Rotation <: AbstractMopt
    measurement::AbstractVector
    POVM_basis::AbstractVector
end

MeasurementOpt{Projection}(measurement, C) = Mopt_Projection(measurement, C)
MeasurementOpt{LinearComb}(measurement, povm, M_num) =
    Mopt_LinearComb(measurement, povm, M_num)
MeasurementOpt{Rotation}(measurement, povm) = Mopt_Rotation(measurement, povm)

abstract type CompOpt <: Opt end

mutable struct StateControlOpt <: CompOpt
    ψ₀::AbstractVector
    control_coefficients::AbstractVector
end

mutable struct ControlMeasurementOpt <: CompOpt
    control_coefficients::AbstractVector
    measurement::AbstractVector
end

mutable struct StateMeasurementOpt <: CompOpt
    ψ₀::AbstractVector
    measurement::AbstractVector
end

mutable struct StateControlMeasurementOpt <: CompOpt
    control_coefficients::AbstractVector
    ψ₀::AbstractVector
    measurement::AbstractVector
end

MeasurementOpt(M, mtype::Symbol = :Projection) = MeasurementOpt{eval(mtype)}(M)
opt_target(::ControlOpt) = :Copt
opt_target(::StateOpt) = :Sopt
opt_target(::MeasurementOpt) = :Mopt
opt_target(::MeasurementOpt{Projection}) = :Mopt_proj
opt_target(::MeasurementOpt{LinearComb}) = :Mopt_lc
opt_target(::MeasurementOpt{Rotation}) = :Mopt_rot
opt_target(::CompOpt) = :CompOpt
opt_target(::StateControlOpt) = :SCopt
opt_target(::ControlMeasurementOpt) = :CMopt
opt_target(::StateMeasurementOpt) = :SMopt
opt_target(::StateControlMeasurementOpt) = :SCMopt

result(opt::ControlOpt) = [opt.control_coefficients]
result(opt::StateOpt) = [opt.ψ₀]
result(opt::MeasurementOpt) = [opt.measurement]
result(opt::StateControlOpt) = [opt.ψ₀, opt.control_coefficients]
result(opt::ControlMeasurementOpt) = [opt.control_coefficients, opt.measurement]
result(opt::StateMeasurementOpt) = [opt.ψ₀, opt.measurement]
result(opt::StateControlMeasurementOpt) =
    [opt.ψ₀, opt.control_coefficients, opt.measurement]

#with reward
result(opt, ::Type{Val{:save_reward}}) = [result(opt)..., [0.0]]

const res_file_name = Dict(
    :Copt => ["control.csv"],
    :Sopt => ["states.csv"],
    :Mopt => ["measurement.csv"],
    :SCopt => ["states.csv", "control.csv"],
    :CMopt => ["control.csv", "measurement.csv"],
    :SMopt => ["states.csv", "measurement.csv"],
    :SCMopt => ["states.csv", "control.csv", "measurement.csv"],
)

res_file(opt::AbstractOpt) = res_file_name[opt_target(opt)]
