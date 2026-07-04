@doc raw"""
    QubitDephasing(r::Vector{Float64}, para_est::String, gamma::Float64, tspan)

Construct a Lindblad dynamics for a single qubit undergoing dephasing.

The Hamiltonian is ``H = r_x\sigma_x + r_y\sigma_y + r_z\sigma_z`` with
dephasing channel ``\Gamma = \sigma_z`` at rate ``\gamma``.

# Arguments

- `r::Vector{Float64}`: Bloch vector components ``[r_x, r_y, r_z]``.
- `para_est::String`: Parameter to estimate (`"x"`, `"y"`, or `"z"`), determining
  which Pauli matrix derivative to use.
- `gamma::Float64`: Dephasing rate.
- `tspan`: Time span for evolution.

# Returns

- `LindbladDynamics`: A Lindblad dynamics object with ODE solver, single-parameter,
  with decay and no control.

# Example

```julia
dyn = QubitDephasing([1.0, 0.0, 0.0], "z", 0.1, 0:0.1:10)
```
"""
function QubitDephasing(r::Vector{Float64}, 
                        para_est::String,
                        gamma::Float64,  
                        tspan::Union{Vector{Float64}, StepRangeLen})
    # Hamiltonian
    H0 = r[1]*SigmaX()+r[2]*SigmaY()+r[3]*SigmaZ()

    # Choose the derivative of the Hamiltonian according to the parameter to be estimated
    if para_est == "x" || para_est == :x
        dH = [SigmaX()]
    elseif para_est == "y" || para_est == :y
        dH = [SigmaY()]
    elseif para_est == "z" || para_est == :z
        dH = [SigmaZ()]
    end

    # Define the decay
    decay = [[SigmaZ()], [gamma]]
    return Lindblad(H0, dH, tspan, decay, dyn_method=:Ode)
end # function QubitDephasing