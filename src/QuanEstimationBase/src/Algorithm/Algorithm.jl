abstract type AbstractAlgorithm end

abstract type AbstractGRAPE <: AbstractAlgorithm end

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
"""
struct GRAPE{T<:Number} <: AbstractGRAPE
    max_episode::Int 
    epsilon::T
end

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
"""
struct GRAPE_Adam{T<:Number, N<:Number} <: AbstractGRAPE
    max_episode::Int
    epsilon::T
    beta1::N
    beta2::N
end

GRAPE(max_episode, epsilon, beta1, beta2) = GRAPE_Adam(max_episode, epsilon, beta1, beta2)

"""
$(TYPEDSIGNATURES)

Control optimization algorithm: GRAPE.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.   
"""
GRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true) = Adam ? GRAPE_Adam(max_episode, epsilon, beta1, beta2) : GRAPE(max_episode, epsilon)

abstract type AbstractautoGRAPE <: AbstractAlgorithm end

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
"""
struct autoGRAPE{T<:Number} <: AbstractautoGRAPE
    max_episode::Int
    epsilon::T
end

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
"""
struct autoGRAPE_Adam{T<:Number, N<:Number} <: AbstractautoGRAPE
    max_episode::Int
    epsilon::T
    beta1::N
    beta2::N
end

autoGRAPE(max_episode, epsilon, beta1, beta2) = autoGRAPE_Adam(max_episode, epsilon, beta1, beta2)

"""
$(TYPEDSIGNATURES)

Control optimization algorithm: auto-GRAPE.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.   
"""
autoGRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true) = Adam ? autoGRAPE_Adam(max_episode, epsilon, beta1, beta2) : autoGRAPE(max_episode, epsilon)

abstract type AbstractAD <:  AbstractAlgorithm end

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
"""
struct AD{T<:Number} <: AbstractAD
    max_episode::Number
    epsilon::T
end

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
"""
struct AD_Adam{T<:Number, N<:Number} <: AbstractAD
    max_episode::Number
    epsilon::T
    beta1::N
    beta2::N
end

AD(max_episode, epsilon, beta1, beta2) = AD_Adam(max_episode, epsilon, beta1, beta2)
"""
$(TYPEDSIGNATURES)

Optimization algorithm: AD.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.
"""
AD(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true) = Adam ? AD_Adam(max_episode, epsilon, beta1, beta2) : AD(max_episode, epsilon)

## TODO: try using immutable struct here. Consider [Accesors.jl](https://github.com/JuliaObjects/Accessors.jl) or Setfield.jl for update. 

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of episodes, it accepts both integer and array with two elements.
- `p_num`: The number of particles. 
- `ini_particle`: Initial guesses of the optimization variables.
- `c0`: The damping factor that assists convergence, also known as inertia weight.
- `c1`: The exploitation weight that attracts the particle to its best previous position, also known as cognitive learning factor.
- `c2`: The exploitation weight that attracts the particle to the best position in the neighborhood, also known as social learning factor. 
"""
mutable struct PSO{T<:Number} <: AbstractAlgorithm
    max_episode::Union{Int,Vector{Int}} 
    p_num::Int
    ini_particle::Union{Tuple, Missing}
    c0::T
    c1::T
    c2::T
end

"""
$(TYPEDSIGNATURES)

Optimization algorithm: PSO.
- `max_episode`: The number of episodes, it accepts both integer and array with two elements.
- `p_num`: The number of particles. 
- `ini_particle`: Initial guesses of the optimization variables.
- `c0`: The damping factor that assists convergence, also known as inertia weight.
- `c1`: The exploitation weight that attracts the particle to its best previous position, also known as cognitive learning factor.
- `c2`: The exploitation weight that attracts the particle to the best position in the neighborhood, also known as social learning factor. 
"""
PSO(;max_episode::Union{T,Vector{T}} where {T<:Int}=[1000, 100], p_num::Number=10, ini_particle=missing, c0::Number=1.0, c1::Number=2.0, c2::Number=2.0) =
    PSO(max_episode, p_num, ini_particle, c0, c1, c2)

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of populations.
- `p_num`: The number of particles. 
- `ini_population`: Initial guesses of the optimization variables.
- `c`: Mutation constant.
- `cr`: Crossover constant.
"""
mutable struct DE{T<:Number} <: AbstractAlgorithm
    max_episode::Int
    p_num::Int
    ini_population::Union{Tuple, Missing}
    c::T
    cr::T
end

"""
$(TYPEDSIGNATURES)

Optimization algorithm: DE.
- `max_episode`: The number of populations.
- `p_num`: The number of particles. 
- `ini_population`: Initial guesses of the optimization variables.
- `c`: Mutation constant.
- `cr`: Crossover constant.
"""
DE(;max_episode::Number=1000, p_num::Number=10, ini_population=missing, c::Number=1.0,cr::Number=0.5) = DE(max_episode, p_num, ini_population, c, cr)

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of populations.
- `layer_num`: The number of layers (include the input and output layer).
- `layer_dim`: The number of neurons in the hidden layer.
- `rng`: random number generator.
"""
struct DDPG{R<:AbstractRNG} <: AbstractAlgorithm
    max_episode::Int
    layer_num::Int
    layer_dim::Int
    rng::R
end

DDPG(max_episode, layer_num, layer_dim) =
    DDPG(max_episode, layer_num, layer_dim, StableRNG(1234))
DDPG(max_episode, layer_num, layer_dim, seed::Number) =
    DDPG(max_episode, layer_num, layer_dim, StableRNG(seed))
"""
$(TYPEDSIGNATURES)

Optimization algorithm: DDPG.
- `max_episode`: The number of populations.
- `layer_num`: The number of layers (include the input and output layer).
- `layer_dim`: The number of neurons in the hidden layer.
- `seed`: Random seed.
"""
DDPG(;max_episode::Int=500, layer_num::Int=3, layer_dim::Int=200, seed::Number=1234) =
    DDPG(max_episode, layer_num, layer_dim, StableRNG(seed))

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of populations.
- `p_num`: The number of the input states.
- `nelder_mead`: Initial guesses of the optimization variables.
- `ar`: Reflection constant.
- `ae`: Expansion constant.
- `ac`: Constraction constant.
- `as0`: Shrink constant.
"""
struct NM{N<:Number} <: AbstractAlgorithm
    max_episode::Int
    p_num::Int
    ini_state::Union{AbstractVector, Missing} 
    ar::N
    ae::N
    ac::N
    as0::N
end

"""
$(TYPEDSIGNATURES)

State optimization algorithm: NM.
- `max_episode`: The number of populations.
- `p_num`: The number of the input states.
- `nelder_mead`: Initial guesses of the optimization variables.
- `ar`: Reflection constant.
- `ae`: Expansion constant.
- `ac`: Constraction constant.
- `as0`: Shrink constant.
"""
NM(;max_episode::Int=1000, p_num::Int=10, nelder_mead=missing, ar::Number=1.0, ae::Number=2.0, ac::Number=0.5, as0::Number=0.5) = NM(max_episode, p_num, nelder_mead, ar, ae, ac, as0)

"""
$(TYPEDEF)

### Fields
- `max_episode`: The number of episodes.
"""
struct RI <: AbstractAlgorithm
    max_episode::Int
end

"""
$(TYPEDSIGNATURES)

State optimization algorithm: RI.
- `max_episode`: The number of episodes.
"""
RI(;max_episode::Int=300) = RI(max_episode)

alg_type(::AD) = :AD
alg_type(::AD_Adam) = :AD
alg_type(::GRAPE) = :GRAPE
alg_type(::GRAPE_Adam) = :GRAPE
alg_type(::autoGRAPE) = :autoGRAPE
alg_type(::autoGRAPE_Adam) = :autoGRAPE
alg_type(::PSO) = :PSO
alg_type(::DDPG) = :DDPG
alg_type(::DE) = :DE
alg_type(::NM) = :NM
alg_type(::RI) = :RI

include("AD.jl")
# include("DDPG.jl")
include("DE.jl")
include("GRAPE.jl")
include("NM.jl")
include("RI.jl")
include("PSO.jl")
