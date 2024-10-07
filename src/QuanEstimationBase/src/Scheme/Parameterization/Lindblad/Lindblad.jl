
# dynamics in Lindblad form
abstract type AbstractDynamics <: AbstractParameterization end

abstract type AbstractDynamicsData end

"""
$(TYPEDEF)

### Fields
* 'data': LindbladData.
* 'params': Other parameters.
"""
mutable struct Lindblad{H, D, C, S, P} <: AbstractDynamics
    data::AbstractDynamicsData
    params::Union{Nothing, AbstractVector}
end

# Lindblad(data::D) where D = Lindblad{D}(data, nothing)

include("LindbladData.jl")
include("LindbladDynamics.jl")
# include("LindbladWrapper.jl")

function set_ctrl(dynamics::Lindblad, ctrl)
    temp = deepcopy(dynamics)
    setfield!(temp.data, :ctrl, ctrl)
    temp
end

function set_state(dynamics::Lindblad, state::AbstractVector)
    temp = deepcopy(dynamics)
    setfield!(temp.data, :ψ0, state)
    temp
end

function set_state(dynamics::Lindblad, state::AbstractMatrix)
    temp = deepcopy(dynamics)
    setfield!(temp.data, :ρ0, state)
    temp
end