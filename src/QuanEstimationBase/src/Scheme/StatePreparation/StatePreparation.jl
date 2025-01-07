
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

PlusState() = complex([1.0, 1.0] / sqrt(2))
MinusState() = complex([1.0, -1.0] / sqrt(2))
BellState() = complex([1.0, 0.0, 0.0, 1.0] / sqrt(2))

function BellState(n::Int)
    if n == 1
        return complex([1.0, 0.0, 0.0, 1.0] / sqrt(2))
    elseif n == 2
        return complex([1.0, 0.0, 0.0, -1.0] / sqrt(2))
    elseif n == 3
        return complex([0.0, 1.0, 1.0, 0.0] / sqrt(2))
    elseif n == 4
        return complex([0.0, 1.0, -1.0, 0.0] / sqrt(2))
    else
        throw("Supported values for n are 1 to 4.")
    end
end

