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
Kraus(K::AbstractVector, dK::AbstractVector, ρ::AbstractMatrix) =
    Kraus{dm}(Kraus_data(K, dK, ρ), :noiseless, :free, :dm)
Kraus(K::AbstractVector, dK::AbstractVector, ψ::AbstractVector) =
    Kraus{ket}(Kraus_data(K, dK, ψ), :noiseless, :free, :ket)

#### evolution of pure states under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Kraus{noiseless,free,ket})
    (K, dK, ψ₀) = dynamics
    ρ₀ = ψ₀ * ψ₀'
    ρ = [K * ρ₀ * K' for K in K] |> sum
    dρ = [[dK * ρ₀ * K' + K * ρ₀ * dK'] |> sum for dK in dK]

    ρ, dρ
end

#### evolution of density matrix under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Kraus{noiseless,free,dm})
    (K, dK, ρ₀) = dynamics
    ρ = [K * ρ₀ * K' for K in K] |> sum
    dρ = [[dK * ρ₀ * K' + K * ρ₀ * dK'] |> sum for dK in dK]

    ρ, dρ
end
