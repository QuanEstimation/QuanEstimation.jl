"""
    AbstractParameterization <: AbstractScheme

Abstract supertype for dynamics parameterization components. Includes
Lindblad master equation, Kraus operators, and qubit dephasing.
"""
abstract type AbstractParameterization <: AbstractScheme end

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
include("QubitDynamics.jl")

"""
    get_parameter_region_length(scheme::AbstractScheme)

Return the number of parameter points in the estimation parameter region, read from `scheme.EstimationStrategy.x`.

For a scalar parameter, `x` is a vector of parameter values. For multiple parameters, `x` is a vector of vectors, and the length is taken from the first parameter's grid.
raw"""
function get_parameter_region_length(scheme::AbstractScheme)
    x = scheme.EstimationStrategy.x
    return length(x isa Vector{<:Number} ? x : x[1])
end

raw"""
    evolve_parameter_region(scheme::AbstractScheme)

Evolve the quantum state ``\rho`` and its derivatives ``\partial\rho`` for each parameter point in the estimation region.

Iterates over all parameter points via [`eachparam`](@ref), calls [`evolve`](@ref) at each point, and collects the results.

# Returns
- `rho::Vector{Matrix{ComplexF64}}`: Density matrices for each parameter point.
- `drho::Vector{Vector{Matrix{ComplexF64}}}`: Derivative vectors for each parameter point.
"""
function evolve_parameter_region(scheme::AbstractScheme)
    (; x,) = getfield(scheme, :EstimationStrategy)
    scheme.Parameterization.params = x isa Vector{<:Number} ? [x] : x
    all_params = eachparam(scheme)
    x_num = get_parameter_region_length(scheme)
    rho = Vector{Matrix{ComplexF64}}(undef, x_num)
    drho = Vector{Vector{Matrix{ComplexF64}}}(undef, x_num)
    for i = 1:x_num
        set_param!(scheme, [all_params[i]...])
        rho_tmp, drho_tmp = evolve(scheme)
        rho[i] = rho_tmp
        drho[i] = drho_tmp
    end
    return rho, drho
end
