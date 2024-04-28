abstract type AbstractScheme end
struct Scheme{S,P,M,E} <: AbstractScheme
	StatePreparation
	Parameterization
	Measurement
	EstimationStrategy
end

include("StatePreparation/StatePreparation.jl")
include("Parameterization/Parameterization.jl")
include("Measurement/Measurement.jl")
include("EstimationStrategy/EstimationStrategy.jl")

function Scheme(state::GeneralState{S}, param::P, meas::M, strat::E) where {S,P,M,E}
	return Scheme{S,P,M,E}(state, param, meas, strat)
end

function GeneralScheme(;
	probe=nothing,
	param=nothing,
	measurement=nothing,
	x=nothing,
	p=nothing,
	dp=nothing,
)
	return Scheme(
		GeneralState(probe),
		param,
		GeneralMeasurement(measurement),
		GeneralStrategy(x,p,dp),
	)
end

state_data(scheme::Scheme) = scheme.StatePreparation.data
param_data(scheme::Scheme) = scheme.Parameterization.data
meas_data(scheme::Scheme) = scheme.Measurement.data
strat_data(scheme::Scheme) = scheme.EstimationStrategy.data