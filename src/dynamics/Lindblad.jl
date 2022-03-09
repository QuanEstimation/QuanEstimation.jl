
# dynamics in Lindblad form
struct Lindblad{N, C} <: AbstractDynamics 
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
end

struct Lindblad_noiseless_free <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
end

struct Lindblad_noisy_free <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end

struct Lindblad_noiseless_controlled <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
end

struct Lindblad_noisy_controlled <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
end

# Constructor of Lindblad dynamics
Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector) = Lindblad{noiseless, free}(Lindblad_noiseless_free(H0, dH, ρ0, tspan), noiseless, false)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector) = Lindblad(Lindblad_noisy_free(H0, dH, ρ0, tspan, decay_opt, γ), noisy, free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad(Lindblad_noiseless_controlled(H0, dH, ρ0, tspan, Hc, cc, cb), noiseless, controlled)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector,Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad(Lindblad_noisy_controlled(H0, dH, ρ0, tspan, decay_opt, γ, Hc, cc, cb), noisy, controlled)