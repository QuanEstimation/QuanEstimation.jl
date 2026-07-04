"""
    mutable struct KrausData <: AbstractKrausData

Stores Kraus operators and their derivatives for parameterized dynamics.

# Fields
- `K`: Kraus operators as a vector of matrices.
- `dK`: Derivatives of the Kraus operators with respect to the estimated parameters.
raw"""
mutable struct KrausData <: AbstractKrausData
    K
    dK
end


@doc raw"""
    Kraus(K::AbstractVector, dK::AbstractVector)

Construct Kraus dynamics from pre-evaluated Kraus operators.

The Kraus channel is ``\rho = \sum_i K_i \rho_0 K_i^\dagger``.

# Arguments
- `K::AbstractVector`: Vector of Kraus operator matrices ``K_i``.
- `dK::AbstractVector`: Vector of vectors, where ``dK[a]`` contains the derivatives ``\partial_a K_i`` with respect to parameter ``a``.
"""
Kraus(K::KT, dK::DKT) where {KT<:AbstractVector,DKT<:AbstractVector} =
    Kraus{KT,DKT,length(K),length(dK[1]),}(KrausData(K, dK), nothing)

@doc raw"""
    Kraus(K::Function, dK::Function, params::Number)

Construct Kraus dynamics from a function returning operators, for a scalar parameter value.

``K(\boldsymbol{x})`` returns a vector of Kraus operator matrices;
``dK(\boldsymbol{x})`` returns a vector of their parameter derivatives.

# Arguments
- `K::Function`: Function ``K(\boldsymbol{x})`` returning a vector of matrices.
- `dK::Function`: Function ``dK(\boldsymbol{x})`` returning the derivative vector.
- `params::Number`: Scalar parameter value at which to evaluate the operators.
"""
Kraus(K::KT, dK::DKT, params::Number) where {KT<:Function,DKT<:Function} =
    Kraus{KT,DKT,length(K(params)),length([params...])}(KrausData(K, dK), [params])

@doc raw"""
    Kraus(K::Function, dK::Function, params::AbstractVector)

Construct Kraus dynamics from a function returning operators, evaluated at a vector of parameter values (multi-parameter case).

# Arguments
- `K::Function`: Function ``K(\boldsymbol{x})`` returning a vector of matrices.
- `dK::Function`: Function ``dK(\boldsymbol{x})`` returning the derivative vector.
- `params::AbstractVector`: Parameter values at which to evaluate.
"""
Kraus(K::KT, dK::DKT, params) where {KT<:Function,DKT<:Function} =
    Kraus{KT,DKT,length(K(params)),length([params...])}(KrausData(K, dK), params)

@doc raw"""
    Kraus(K::Function, dK::Function)

Construct Kraus dynamics from a function returning operators, without pre-specifying parameter values.

# Arguments
- `K::Function`: Function ``K(\boldsymbol{x})`` returning a vector of matrices.
- `dK::Function`: Function ``dK(\boldsymbol{x})`` returning the derivative vector.
raw"""
Kraus(K::KT, dK::DKT) where {KT<:Function,DKT<:Function} =
    Kraus{KT,DKT,Nothing,Nothing}(KrausData(K, dK),nothing)
    
@doc raw"""
    evaluate_kraus(k::Kraus)

Return the Kraus operators and their parameter derivatives.

When the operators are stored as matrices (pre-evaluated), they are returned directly.
When stored as functions, they are evaluated at the stored parameter values.

The evolved density matrix is ``\rho = \sum_i K_i \rho_0 K_i^\dagger``.

# Returns
- `K`: Vector of Kraus operator matrices.
- `dK`: Vector of derivative vectors ``\partial_a K_i``.
"""
evaluate_kraus(k::Kraus{KT, DKT}) where {KT<:AbstractVector,DKT<:AbstractVector}= k.data.K, k.data.dK
evaluate_kraus(k::Kraus) = k.data.K(k.params...), k.data.dK(k.params...)

"""
    get_dim(k::Kraus)

Return the dimension of the Hilbert space on which the Kraus operators act.
"""
get_dim(k::Kraus) = size(evaluate_kraus(k)[1], 1)

"""
    get_param_num(::Type{Kraus{KT,DKT,NK,NP}})

Extract the number of parameters `NP` from the `Kraus` type.

See also: [`get_param_num(::Scheme{...})`](@ref).
"""
get_param_num(::Type{Kraus{KT,DKT,NK,NP}}) where {KT,DKT,NK,NP} = NP
"""
    get_param_num(::Scheme{S,K,M,E}) where {K<:AbstractKraus}

Extract the number of parameters from a `Scheme` containing Kraus dynamics.
"""
get_param_num(::Scheme{S,K,M,E}) where {S,K<:AbstractKraus,M,E} = get_param_num(K)

"""
    para_type(::Kraus{KT,DKT,NK,NP})

Return `:single_para` if `NP == 1`, otherwise `:multi_para`.
"""
para_type(::Kraus{KT,DKT,NK,NP}) where {KT,DKT,NK,NP} = NP == 1 ? :single_para : :multi_para
"""
    para_type(::Scheme{S,K,M,E}) where {K<:AbstractKraus}

Extract the parameter type label from a `Scheme` containing Kraus dynamics.
"""
para_type(::Scheme{S,K,M,E}) where {S,K<:AbstractKraus,M,E} = para_type(K)