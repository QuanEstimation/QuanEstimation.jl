
abstract type AbstractDynamics <: AbstractParameterization end
# dynamics in Lindblad form
struct Lindblad{D, P} <: AbstractDynamics
    data::D
    params::P
end

Lindblad(data::D) where D = Lindblad{D}(data, nothing)

include("LindbladData.jl")
include("LindbladDynamics.jl")
include("LindbladWrapper.jl")

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