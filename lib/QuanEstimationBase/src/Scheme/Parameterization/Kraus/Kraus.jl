abstract type AbstractKraus <: AbstractParameterization end
abstract type AbstractKrausData end


mutable struct Kraus{KT,DKT,NK,NP} <: AbstractKraus
    data::AbstractKrausData
    params::Union{Nothing,AbstractVector}
end

include("KrausData.jl")
include("KrausParamerization.jl")
