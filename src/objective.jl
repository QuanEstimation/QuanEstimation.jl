abstract type AbstractObj end

abstract type AbstracParaType end
abstract type single_para <: AbstracParaType end
abstract type multi_para <: AbstracParaType end

include("AsymptoticBound/CramerRao.jl")
include("AsymptoticBound/Holevo")
