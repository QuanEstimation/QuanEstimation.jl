abstract type AbstractScheme end
struct Scheme{S,P,M,E} <: AbstractScheme
	StatePreparation
	Parameterization
	Measurement
	EstimationStrategy
end

include("StatePreparation/StatePreparation.jl")
include("Measurement/Measurement.jl")
include("EstimationStrategy/EstimationStrategy.jl")
include("Parameterization/Parameterization.jl")
include("error_evaluation.jl")
include("error_control.jl")

function Scheme(state::GeneralState{S}, param::P, meas::M, strat::E) where {S,P,M,E}
	return Scheme{S,P,M,E}(state, param, meas, strat)
end

function GeneralScheme(;
	probe=nothing,
	param=nothing,
	measurement=nothing,
	strat=nothing,
	x=nothing,
	p=nothing,
	dp=nothing,
)
	return Scheme(
		GeneralState(probe),
		param,
		isnothing(measurement) ? GeneralMeasurement(SIC(get_dim(param))) : GeneralMeasurement(measurement),
		isnothing(strat) ? GeneralEstimation(x,p,dp) : strat,
	)
end


state_data(scheme::Scheme) = scheme.StatePreparation.data
param_data(scheme::Scheme) = scheme.Parameterization.data
meas_data(scheme::Scheme) = scheme.Measurement.data
strat_data(scheme::Scheme) = scheme.EstimationStrategy

set_ctrl!(scheme, ctrl) = set_ctrl!(scheme.Parameterization, ctrl)
set_ctrl(scheme, ctrl) = set_ctrl(scheme.Parameterization, ctrl)
set_state!(scheme, state) = @set scheme.StatePreparation.data=state

get_dim(scheme) = get_dim(scheme.Parameterization)