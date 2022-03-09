# dynamics in Kraus rep.
struct Kraus{R} <: AbstractDynamics 
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
    state_rep::Symbol
end

struct Kraus_dm <: AbstractDynamicsData
    K::AbstractVector
    dK::AbstractVector
    ρ0::AbstractMatrix
end

struct Kraus_pure <: AbstractDynamicsD
    K::AbstractVector
    dK::AbstractVector
    ψ0::AbstractVector
end

# Constructor for Kraus dynamics
Kraus(K::AbstractVector, dK::AbstractVector, ρ::AbstractMatrix) = Kraus{dm}(Kraus_data(K,dK,ρ), :noiseless, :free, :dm)
Kraus(K::AbstractVector, dK::AbstractVector, ψ::AbstractVector) = Kraus{ket}(Kraus_data(K,dK,ψ), :noiseless, :free, :ket)
