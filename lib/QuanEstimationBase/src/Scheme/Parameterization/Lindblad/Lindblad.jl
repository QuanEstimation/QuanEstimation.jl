
# dynamics in Lindblad form
abstract type AbstractDynamics <: AbstractParameterization end

abstract type AbstractDynamicsData end

mutable struct LindbladDynamics{H,D,C,S,P} <: AbstractDynamics
    data::AbstractDynamicsData
    params::Union{Nothing,AbstractVector}
end

# Lindblad(data::D) where D = LindbladDynamics{D, Nothing}(data, nothing)

include("LindbladData.jl")
include("LindbladDynamics.jl")

function get_param(scheme::Scheme{S,P,M,E}) where {S,P<:AbstractDynamics,M,E}
    return param_data(scheme).hamiltonian.params
end

function set_param!(scheme::Scheme{S,P,M,E}, x) where {S,P<:AbstractDynamics,M,E}
    param_data(scheme).hamiltonian.params = [x...]
end

function eachparam(scheme::Scheme{S,P,M,E}) where {S,P<:AbstractDynamics,M,E}
    [p for p in zip(scheme.Parameterization.params...)]
end

function set_ctrl!(dynamics::LindbladDynamics, ctrl)
    set_ctrl!(dynamics.data, ctrl)
    dynamics
end

function set_ctrl!(data::LindbladData, ctrl)
    setfield!(data, :ctrl, ctrl)
    data
end

function set_ctrl(dynamics::Scheme, ctrl)
    temp = deepcopy(dynamics)
    setfield!(temp.Parameterization.data, :ctrl, ctrl)
    temp
end

function set_state(dynamics::LindbladDynamics, state::AbstractVector)
    temp = deepcopy(dynamics)
    setfield!(temp.data, :ψ0, state)
    temp
end

function set_state(dynamics::LindbladDynamics, state::AbstractMatrix)
    temp = deepcopy(dynamics)
    setfield!(temp.data, :ρ0, state)
    temp
end
