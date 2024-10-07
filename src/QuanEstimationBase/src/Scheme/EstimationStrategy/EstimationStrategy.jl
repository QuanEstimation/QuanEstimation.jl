abstract type EstimationStrategy end

"""
$(TYPEDEF)

### Fields
* 'x': ParameterRegion.
* 'p': PriorDistribution.
* 'dp': DistributionDerivative.
"""
struct Strategy <: EstimationStrategy
    x #ParameterRegion
    p #PriorDistribution
    dp #DistributionDerivative
end

Strategy(;x=nothing, p=nothing, dp=nothing) = Strategy(x, p, dp)

function GeneralEstimation(x, p, dp)
    return Strategy(x, p, dp)
end

include("AdaptiveEstimation/AdaptiveEstimation.jl")