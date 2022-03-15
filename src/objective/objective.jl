abstract type AbstractObj end

abstract type quantum end
abstract type classical end

abstract type AbstracParaType end
abstract type single_para <: AbstracParaType end
abstract type multi_para <: AbstracParaType end

include("AsymptoticBound/AsympototicBound.jl")
