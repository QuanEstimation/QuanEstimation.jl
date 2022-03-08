abstract type AbstractSystem end

mutable struct QuanEstSystem{T<:AbstractOpt, F<:AbstractObjective, A<:AbstractAlogrithm, D<:AbstractDynamics, O<:AbstractOutput} <:AbstractSystem
    opt::T
    objective::F
    algorithm::A
    dynamics::D
    output::O
end



