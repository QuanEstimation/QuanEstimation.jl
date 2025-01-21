mutable struct KrausData <: AbstractKrausData
    K::AbstractVector
    dK::AbstractVecOrMat
end


# # Constructor for Kraus dynamics
# @doc raw"""

#     Kraus(ρ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector)

# The parameterization of a state is ``\rho=\sum_i K_i\rho_0K_i^{\dagger}`` with ``\rho`` the evolved density matrix and ``K_i`` the Kraus operator.
# - `ρ0`: Initial state (density matrix).
# - `K`: Kraus operators.
# - `dK`: Derivatives of the Kraus operators with respect to the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.
# """
Kraus(K::KT, dK::DKT) where {KT<:AbstractVector,DKT<:AbstractVector} =
    Kraus{KT,DKT,length(K),length(dK[1])}(KrausData(K, dK), nothing)
Kraus(K::KT, dK::DKT, params) where {KT<:Function,DKT<:Function} =
    Kraus{KT,DKT,length(K(params)),length([params...])}(KrausData(K, dK), params)
Kraus(K::KT, dK::DKT) where {KT<:Function,DKT<:Function} =
    Kraus{KT,DKT,Nothing,Nothing}(KrausData(K, dK), nothing)

# para_type(data::KrausData) = length(data.dK[1]) == 1 ? :single_para : :multi_para

get_dim(k::KrausData) = size(k.K[1], 1)
get_dim(k::Kraus) = get_dim(k.data)

get_param_num(::Type{Kraus{KT,DKT,NK,NP}}) where {KT,DKT,NK,NP} = NP
get_param_num(::Scheme{S,K,M,E}) where {S,K<:AbstractKraus,M,E} = get_param_num(K)

para_type(::Kraus{KT,DKT,NK,NP}) where {KT,DKT,NK,NP} = NP == 1 ? :single_para : :multi_para
para_type(::Scheme{S,K,M,E}) where {S,K<:AbstractKraus,M,E} = para_type(K)
