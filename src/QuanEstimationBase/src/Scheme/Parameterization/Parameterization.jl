abstract type AbstractParameterization end
abstract type AbstractDynamics <: AbstractParameterization end
abstract type AbstractDynamicsData end

abstract type AbstractStateType end
abstract type ket <: AbstractStateType end
abstract type dm <: AbstractStateType end

abstract type AbstractNoiseType end
abstract type noiseless <: AbstractNoiseType end
abstract type noisy <: AbstractNoiseType end

abstract type AbstractCtrlType end
abstract type free <: AbstractCtrlType end
abstract type controlled <: AbstractCtrlType end
abstract type timedepend <: AbstractCtrlType end

abstract type AbstractDynamicsProblemType end
abstract type Expm <: AbstractDynamicsProblemType end
abstract type Ode <: AbstractDynamicsProblemType end

evolve(scheme::AbstractScheme) = evolve(scheme.Parameterization)
evolve(param::AbstractParameterization) = _evolve(param.data)

# check if the dynamics are with noise
isNoisy(::noiseless) = false
isNoisy(::noisy) = true
isNoisy(dynamics::AbstractDynamics) = dynamics.noise_type |> isNoisy

# check if the dynamics are in control
isCtrl(::free) = false
isCtrl(::controlled) = true
isCtrl(dynamics::AbstractDynamics) = dynamics.ctrl_type |> isCtrl

include("Lindblad/Lindblad.jl")
include("Kraus/Kraus.jl")
