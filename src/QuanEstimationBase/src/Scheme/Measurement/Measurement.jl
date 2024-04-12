abstract type AbstractMeasurement end
abstract type AbstractPOVM end
abstract type SIC_POVM <: AbstractPOVM end
abstract type POVM <: AbstractPOVM end
struct GeneralMeasurement{M<:AbstractPOVM} <: AbstractMeasurement
	data::M
end

function GeneralMeasurement(dim::Int)
	return GeneralMeasurement{SIC_POVM}(dim)
end

function GeneralMeasurement(M::Vector{Matrix{T}}) where {T<:Number}
	return GeneralMeasurement{POVM}(M)
end

function GeneralMeasurement(meas_hook::Function, args...)
	return GeneralMeasurement(meas_hook(args...) )
end