struct Strategy
    ParameterRegion
    PriorDistribution
    DistributionDerivative
end

Strategy(;x=nothing, p=nothing, dp=nothing) = Strategy(x, p, dp)

function GeneralStrategy(x, p, dp)
    return Strategy(x, p, dp)
end