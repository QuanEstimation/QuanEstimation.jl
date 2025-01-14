abstract type AbstractKrausData end

mutable struct Kraus{KT,DKT,NK,NP} <: AbstractParameterization
    data::AbstractKrausData
    params::Union{Nothing,AbstractVector}
end

include("KrausData.jl")
include("KrausParamerization.jl")
# include("KrausWrapper.jl")

# function set_state(dynamics::Kraus, state::AbstractVector)
#     temp = deepcopy(dynamics)
#     temp.data.ψ0 = state
#     temp
# end

# function set_state(dynamics::Kraus, state::AbstractMatrix)
#     temp = deepcopy(dynamics)
#     temp.data.ρ0 = state
#     temp
# end
