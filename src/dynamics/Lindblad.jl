
# dynamics in Lindblad form
struct Lindblad{N, C, R} <: AbstractDynamics 
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
    state_rep::Symbol
end

Lindblad{n,c}(data::AbstractDynamicsData,N::Symbol,C::Symbol) where {n,c} =Lindblad{n,c,dm}(data,N,C,:dm)

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

struct Lindblad_noiseless_free_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
end

struct Lindblad_noisy_free_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end

struct Lindblad_noiseless_controlled_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
end

struct Lindblad_noisy_controlled_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
end


# Constructor of Lindblad dynamics
Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector) = Lindblad{noiseless, free}(Lindblad_noiseless_free(H0, dH, ρ0, tspan), :noiseless, :free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector) = Lindblad{noisy,free}(Lindblad_noisy_free(H0, dH, ρ0, tspan, decay_opt, γ), :noisy, :free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad{noiseless,controlled}(Lindblad_noiseless_controlled(H0, dH, ρ0, tspan, Hc, cc, cb), :noiseless, :controlled)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector,Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad{noisy, controlled}(Lindblad_noisy_controlled(H0, dH, ρ0, tspan, decay_opt, γ, Hc, cc, cb), :noisy, :controlled)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ψ0::AbstractVector, tspan::AbstractVector) = Lindblad{noiseless, free, ket}(Lindblad_noiseless_free(H0, dH, ψ0, tspan), noiseless, free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ψ0::AbstractVector, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector) = Lindblad{noisy,free,ket}(Lindblad_noisy_free(H0, dH, ψ0, tspan, decay_opt, γ), noisy, free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ψ0::AbstractVector, tspan::AbstractVector, Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad{noiseless,controlled, ket}(Lindblad_noiseless_controlled(H0, dH, ψ0, tspan, Hc, cc, cb), noiseless, controlled)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ψ0::AbstractVector, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector,Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad{noisy, controlled, ket}(Lindblad_noisy_controlled(H0, dH, ψ0, tspan, decay_opt, γ, Hc, cc, cb), noisy, controlled)