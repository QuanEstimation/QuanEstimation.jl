"""
    EstimationStrategy <: AbstractScheme

Abstract supertype for estimation strategy components.
raw"""
abstract type EstimationStrategy <: AbstractScheme end

@doc raw"""
    Strategy <: EstimationStrategy

Estimation strategy for non-adaptive (fixed) parameter estimation.

Holds the parameter grid, prior distribution, and its derivatives,
used in Bayesian bound calculations.

# Fields

- `x::Union{Nothing,AbstractVector}`: Parameter grid points.
- `p::Union{Nothing,AbstractVector}`: Prior distribution ``p(\mathbf{x})``.
- `dp::Union{Nothing,AbstractVector}`: Prior derivatives ``\partial_a p(\mathbf{x})``.
"""
struct Strategy <: EstimationStrategy
    x::Union{Nothing, AbstractVector}
    p::Union{Nothing, AbstractVector}
    dp::Union{Nothing, AbstractVector}
end

"""
    Strategy(; x, p, dp)

Keyword constructor for [`Strategy`](@ref).
"""
Strategy(; x = nothing, p = nothing, dp = nothing) = Strategy(Vector(x), p, dp)

"""
    GeneralEstimation(x, p, dp)

Construct a [`Strategy`](@ref) from an estimation specification.

Handles `nothing` inputs by passing them through unchanged.
"""
function GeneralEstimation(x, p, dp)
    return Strategy(isnothing(x) ? x : Vector(x), p, dp)
end

include("AdaptiveEstimation/AdaptiveEstimation.jl")
