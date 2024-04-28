abstract type AbstractScheme end
include("StatePreparation/StatePreparation.jl")
include("Parameterization/Parameterization.jl")
include("Measurement/Measurement.jl")
# include("ClassicalEstimation/ClassicalEstimation.jl")

struct GeneralScheme <: AbstractScheme
	StatePreparation::AbstractStatePreparation
	Parameterization::AbstractParameterization
	Measurement
	EstimationStrategy
end

function GeneralScheme(;
	probe=nothing,
	param=nothing,
	measurement=nothing,
	x=nothing,
	p=nothing,
	dp=nothing,
)
	return GeneralScheme(
		GeneralState(probe),
		param,
		GeneralMeasurement(measurement),
		GeneralStrategy(x,p,dp),
	)
end
