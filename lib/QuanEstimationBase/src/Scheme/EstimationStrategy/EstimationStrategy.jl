abstract type EstimationStrategy <: AbstractScheme end
# abstract type ParameterRegion end
struct Strategy <: EstimationStrategy
    x::Union{Nothing, AbstractVector} #ParameterRegion
    p::Union{Nothing, AbstractVector} #PriorDistribution
    dp::Union{Nothing, AbstractVector} #DistributionDerivative
end

Strategy(; x = nothing, p = nothing, dp = nothing) = Strategy(Vector(x), p, dp)

function GeneralEstimation(x, p, dp)
    return Strategy(isnothing(x) ? x : Vector(x), p, dp)
end

include("AdaptiveEstimation/AdaptiveEstimation.jl")
