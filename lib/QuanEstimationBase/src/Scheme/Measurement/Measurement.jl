"""
    AbstractMeasurement <: AbstractScheme

Abstract supertype for measurement components of a scheme.
"""
abstract type AbstractMeasurement <: AbstractScheme end

"""
    AbstractPOVM

Abstract supertype for POVM types.
"""
abstract type AbstractPOVM end

"""
    SIC_POVM <: AbstractPOVM

Tag for symmetric informationally complete POVM.
"""
abstract type SIC_POVM <: AbstractPOVM end

"""
    POVM <: AbstractPOVM

Tag for a general user-specified POVM.
raw"""
abstract type POVM <: AbstractPOVM end

@doc raw"""
    GeneralMeasurement{M<:AbstractPOVM} <: AbstractMeasurement

Measurement component holding POVM elements ``\{\Pi_y\}`` or a dimension
for default SIC-POVM construction.

# Type Parameters

- `M`: POVM type tag (`SIC_POVM` or `POVM`).

# Fields

- `data`: Either POVM elements as a vector of matrices, a dimension `Int64`
  (for SIC-POVM), or `nothing`.
"""
struct GeneralMeasurement{M<:AbstractPOVM} <: AbstractMeasurement
    data::Union{Nothing,Vector{Matrix{T}},Int64} where {T}
end

"""
    GeneralMeasurement(dim::Int64)

Construct a measurement with default SIC-POVM of given dimension.
"""
function GeneralMeasurement(dim::Int64)
    return GeneralMeasurement{SIC_POVM}(dim)
end

"""
    GeneralMeasurement(::Nothing)

Construct a measurement placeholder (SIC-POVM to be determined later).
"""
function GeneralMeasurement(::Nothing)
    return GeneralMeasurement{SIC_POVM}(nothing)
end

"""
    GeneralMeasurement(M::Vector{Matrix{T}}) where {T<:Number}

Construct a measurement from explicit POVM element matrices.
"""
function GeneralMeasurement(M::Vector{Matrix{T}}) where {T<:Number}
    return GeneralMeasurement{POVM}(M)
end

"""
    GeneralMeasurement(meas_hook::Function, args...)

Construct a measurement from a function that returns POVM elements.
"""
function GeneralMeasurement(meas_hook::Function, args...)
    return GeneralMeasurement(meas_hook(args...))
end
