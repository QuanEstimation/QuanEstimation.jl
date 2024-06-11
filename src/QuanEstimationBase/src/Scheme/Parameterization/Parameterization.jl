abstract type AbstractParameterization end

# evolve(scheme::AbstractScheme) = evolve(scheme.Parameterization)
# evolve(param::AbstractParameterization) = _evolve(param.data)

# # check if the dynamics are with noise
# isNoisy(::noiseless) = false
# isNoisy(::noisy) = true
# isNoisy(dynamics::AbstractDynamics) = dynamics.noise_type |> isNoisy

# # check if the dynamics are in control
# isCtrl(::free) = false
# isCtrl(::controlled) = true
# isCtrl(dynamics::AbstractDynamics) = dynamics.ctrl_type |> isCtrl

include("Lindblad/Lindblad.jl")
include("Kraus/Kraus.jl")
