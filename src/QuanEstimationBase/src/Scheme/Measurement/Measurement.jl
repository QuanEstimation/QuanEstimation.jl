abstract type AbstractMeasurement end
abstract type AbstractPOVM end
abstract type SIC_POVM <: AbstractPOVM end
abstract type POVM <: AbstractPOVM end
struct GeneralMeasurement{M<:AbstractPOVM} <: AbstractMeasurement
    data::Union{Nothing,Vector{Matrix{T}},Int64} where {T}
end

function GeneralMeasurement(dim::Int64)
    return GeneralMeasurement{SIC_POVM}(dim)
end

function GeneralMeasurement(::Nothing)
    return GeneralMeasurement{SIC_POVM}(nothing)
end

function GeneralMeasurement(M::Vector{Matrix{T}}) where {T<:Number}
    return GeneralMeasurement{POVM}(M)
end

function GeneralMeasurement(meas_hook::Function, args...)
    return GeneralMeasurement(meas_hook(args...))
end
