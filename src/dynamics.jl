abstract type AbstractDynamics end
abstract type AbstractDynamicsData end

abstract type AbstractState end
abstract type ket <: AbstractState end
abstract type dm <: AbstractState end

abstract type AbstractNoise end
abstract type noiseless <: AbstractNoise end
abstract type noisy <: AbstractNoise end

abstract type AbstractCtrl end
abstract type free <: AbstractCtrl end
abstract type controlled <: AbstractCtrl end
abstract type timedepend <: AbstractCtrl end

# check if the dynamics are with noise
isNoisy(::noiseless) = false
isNoisy(::noisy) = true
isNoisy(dynamics::AbstractDynamics) = dynamics.noise_type |> eval |> isNoisy

# check if the dynamics are in control
isCtrl(::free) = false
isCtrl(::controlled) = true
isCtrl(dynamics::AbstractDynamics) = dynamics.ctrl_type |> eval |> isCtrl

include("dynamics/Lindblad.jl")
include("dynamics/Kraus.jl")
