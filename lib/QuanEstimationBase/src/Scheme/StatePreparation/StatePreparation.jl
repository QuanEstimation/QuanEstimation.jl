"""
    AbstractStatePreparation <: AbstractScheme

Abstract supertype for state preparation components.
"""
abstract type AbstractStatePreparation <: AbstractScheme end

"""
    AbstractState

Abstract supertype for quantum states.
"""
abstract type AbstractState end

"""
    DensityMatrix <: AbstractState

Tag type for density matrix representation ``\rho``.
raw"""
abstract type DensityMatrix <: AbstractState end

raw"""
    Ket <: AbstractState

Tag type for pure state (ket) representation ``|\psi\rangle``.
raw"""
abstract type Ket <: AbstractState end

@doc raw"""
    GeneralState{S<:AbstractState} <: AbstractStatePreparation

State preparation component holding the initial quantum state.

# Type Parameters

- `S`: State type tag (`DensityMatrix` or `Ket`).

# Fields

- `data`: The state data — a matrix for ``\rho_0`` or a vector for ``|\psi_0\rangle``.
"""
mutable struct GeneralState{S<:AbstractState} <: AbstractStatePreparation
    data
end

"""
    GeneralState(probe::Matrix{T}) where {T<:Number}

Construct a density matrix state ``\rho_0``.
"""
function GeneralState(probe::Matrix{T}) where {T<:Number}
    return GeneralState{DensityMatrix}(complex(probe))
end

raw"""
    GeneralState(probe::Vector{T}) where {T<:Number}

Construct a pure state ket ``|\psi_0\rangle``.
"""
function GeneralState(probe::Vector{T}) where {T<:Number}
    return GeneralState{Ket}(complex(probe))
end

"""
    GeneralState(probe_hook::Function, args...)

Construct a state from a function that returns the state data.
"""
function GeneralState(probe_hook::Function, args...)
    return GeneralState(probe_hook(args...))
end

"""
    state_data(state::GeneralState)

Return the raw state data.
"""
state_data(state::GeneralState) = state.data
