# dynamics in Lindblad form
"""
    AbstractDynamics

Abstract supertype for dynamics parameterizations in Lindblad form.
"""
abstract type AbstractDynamics <: AbstractParameterization end

"""
    AbstractDynamicsData

Abstract supertype for dynamics data containers.
raw"""
abstract type AbstractDynamicsData end

@doc raw"""
    LindbladDynamics{H,D,C,S,P} <: AbstractDynamics

Parameterization of Lindblad dynamics.

# Type Parameters
- `H`: Hamiltonian type.
- `D`: Decay type (``\mathrm{NonDecay}`` or ``\mathrm{Decay}``).
- `C`: Control type (``\mathrm{Control}`` or ``\mathrm{NonControl}``).
- `S`: Solver type (``\mathrm{Expm}`` or ``\mathrm{Ode}``).
- `P`: Parameter presence type.

# Fields
- `data::AbstractDynamicsData`: Dynamics data (Hamiltonian, tspan, decay, controls, etc.).
- `params::Union{Nothing,AbstractVector}`: Parameter values, or `nothing` if unset.
"""
mutable struct LindbladDynamics{H,D,C,S,P} <: AbstractDynamics
    data::AbstractDynamicsData
    params::Union{Nothing,AbstractVector}
end

# Lindblad(data::D) where D = LindbladDynamics{D, Nothing}(data, nothing)

include("LindbladData.jl")
include("LindbladDynamics.jl")

"""
    get_param(scheme)

Get the parameters from a scheme's Hamiltonian.
"""
function get_param(scheme::Scheme{S,P,M,E}) where {S,P<:AbstractDynamics,M,E}
    return param_data(scheme).hamiltonian.params
end

"""
    set_param!(scheme, x)

Set the parameters of a scheme's Hamiltonian in-place.
"""
function set_param!(scheme::Scheme{S,P,M,E}, x) where {S,P<:AbstractDynamics,M,E}
    param_data(scheme).hamiltonian.params = [x...]
end

"""
    eachparam(scheme)

Return a vector of zipped parameter tuples for iteration.
"""
function eachparam(scheme::Scheme{S,P,M,E}) where {S,P<:AbstractDynamics,M,E}
    return [p for p in zip(scheme.Parameterization.params...)]
end

"""
    set_ctrl!(dynamics::LindbladDynamics, ctrl)

Set control coefficients on the dynamics in-place.
"""
function set_ctrl!(dynamics::LindbladDynamics, ctrl)
    set_ctrl!(dynamics.data, ctrl)
    dynamics
end

"""
    set_ctrl!(data::LindbladData, ctrl)

Set control coefficients on the Lindblad data in-place.
"""
function set_ctrl!(data::LindbladData, ctrl)
    setfield!(data, :ctrl, ctrl)
    data
end

"""
    set_ctrl(dynamics::Scheme, ctrl)

Return a copy of the scheme with new control coefficients.
"""
function set_ctrl(dynamics::Scheme, ctrl)
    temp = deepcopy(dynamics)
    setfield!(temp.Parameterization.data, :ctrl, ctrl)
    return temp
end

"""
    set_state(dynamics::LindbladDynamics, state::AbstractVector)

Return a copy of the dynamics with the initial pure state set to `state`.
"""
function set_state(dynamics::LindbladDynamics, state::AbstractVector)
    temp = deepcopy(dynamics)
    setfield!(temp.data, :ψ0, state)
    return temp
end

"""
    set_state(dynamics::LindbladDynamics, state::AbstractMatrix)

Return a copy of the dynamics with the initial density matrix set to `state`.
"""
function set_state(dynamics::LindbladDynamics, state::AbstractMatrix)
    temp = deepcopy(dynamics)
    setfield!(temp.data, :ρ0, state)
    return temp
end
