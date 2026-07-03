"""
    AbstractObj

Abstract supertype for optimization objective function wrappers.
"""
abstract type AbstractObj end

"""
    quantum

Tag type for quantum objective functions (QFI, QFIM, HCRB, NHB).
"""
abstract type quantum end

"""
    classical

Tag type for classical objective functions (CFI, CFIM).
"""
abstract type classical end

"""
    AbstractParaType

Abstract supertype for parameter type tags.
"""
abstract type AbstractParaType end

"""
    single_para <: AbstractParaType

Tag for single-parameter estimation.
"""
abstract type single_para <: AbstractParaType end

"""
    multi_para <: AbstractParaType

Tag for multi-parameter estimation.
"""
abstract type multi_para <: AbstractParaType end

include("AsymptoticBound/AsymptoticBound.jl")
include("BayesianBound/BayesianBound.jl")
