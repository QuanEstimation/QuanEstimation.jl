# dynamics in Kraus rep.
struct Kraus{N, C} <: AbstractDynamics 
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
end

struct Kraus_data <: AbstractDynamics 
    K::AbstractVector
    dK::AbstractVector
end

# Constructor for Kraus dynamics
Kraus(K::AbstractVector, dK::AbstractVector) = Kraus(Kraus_data(K,dK))
