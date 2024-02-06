abstract type AbstractDynamics end
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
