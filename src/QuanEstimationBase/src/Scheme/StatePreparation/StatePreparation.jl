
abstract type AbstractStatePreparation end
abstract type AbstractState end
abstract type DensityMatrix <: AbstractState end
abstract type Ket <: AbstractState end

struct GeneralState{S<:AbstractState} <: AbstractStatePreparation
	data
end

function GeneralState(probe::Matrix{T}) where {T<:Number}
	return GeneralState{DensityMatrix}(probe)
end

function GeneralState(probe::Vector{T}) where {T<:Number}
	return GeneralState{Ket}(probe)
end

function GeneralState(probe_hook::Function, args...)
	return GeneralState(probe_hook(args...) )
end

state_data(state::GeneralState) = state.data