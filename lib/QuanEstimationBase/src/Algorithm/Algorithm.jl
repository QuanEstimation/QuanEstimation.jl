"""
    AbstractAlgorithm

Abstract supertype for all optimization algorithms.
"""
abstract type AbstractAlgorithm end

"""
    AbstractGRAPE <: AbstractAlgorithm

Abstract supertype for GRAPE (GRadient Ascent Pulse Engineering) algorithms.
"""
abstract type AbstractGRAPE <: AbstractAlgorithm end

@doc raw"""
    GRAPE{T<:Number} <: AbstractGRAPE

GRadient Ascent Pulse Engineering for control optimization.

Uses the gradient of the objective function w.r.t. control amplitudes
to iteratively improve the controls with a fixed learning rate ``\epsilon``.

# Fields

- `max_episode::Int`: Number of optimization iterations.
- `epsilon::T`: Learning rate (step size).
"""
struct GRAPE{T<:Number} <: AbstractGRAPE
    max_episode::Int
    epsilon::T
end

@doc raw"""
    GRAPE_Adam{T,N} <: AbstractGRAPE

GRAPE with Adam optimizer.

Uses adaptive moment estimation (Adam) for gradient-based control updates
with momentum (``\beta_1``) and RMS scaling (``\beta_2``).

# Fields

- `max_episode::Int`: Number of optimization iterations.
- `epsilon::T`: Learning rate.
- `beta1::N`: Exponential decay rate for first moment estimates.
- `beta2::N`: Exponential decay rate for second moment estimates.
"""
struct GRAPE_Adam{T<:Number,N<:Number} <: AbstractGRAPE
    max_episode::Int
    epsilon::T
    beta1::N
    beta2::N
end

"""
    GRAPE(max_episode, epsilon, beta1, beta2)

Positional constructor — always returns `GRAPE_Adam`.
"""
GRAPE(max_episode, epsilon, beta1, beta2) = GRAPE_Adam(max_episode, epsilon, beta1, beta2)

"""

    GRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)

Control optimization algorithm: GRAPE.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.   
"""
GRAPE(; max_episode = 300, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99, Adam::Bool = true) =
    Adam ? GRAPE_Adam(max_episode, epsilon, beta1, beta2) : GRAPE(max_episode, epsilon)

"""
    AbstractautoGRAPE <: AbstractAlgorithm

Abstract supertype for auto-GRAPE (automatic differentiation + GRAPE) algorithms.
"""
abstract type AbstractautoGRAPE <: AbstractAlgorithm end

"""
    autoGRAPE{T} <: AbstractautoGRAPE

Auto-GRAPE: automatic differentiation + GRAPE with fixed learning rate.
"""
struct autoGRAPE{T<:Number} <: AbstractautoGRAPE
    max_episode::Int
    epsilon::T
end

"""
    autoGRAPE_Adam{T,N} <: AbstractautoGRAPE

Auto-GRAPE with Adam optimizer.
"""
struct autoGRAPE_Adam{T<:Number,N<:Number} <: AbstractautoGRAPE
    max_episode::Int
    epsilon::T
    beta1::N
    beta2::N
end

"""
    autoGRAPE(max_episode, epsilon, beta1, beta2)

Positional constructor — always returns `autoGRAPE_Adam`.
"""
autoGRAPE(max_episode, epsilon, beta1, beta2) =
    autoGRAPE_Adam(max_episode, epsilon, beta1, beta2)

"""

    autoGRAPE(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)

Control optimization algorithm: auto-GRAPE.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.   
"""
autoGRAPE(;
    max_episode = 300,
    epsilon = 0.01,
    beta1 = 0.90,
    beta2 = 0.99,
    Adam::Bool = true,
) =
    Adam ? autoGRAPE_Adam(max_episode, epsilon, beta1, beta2) :
    autoGRAPE(max_episode, epsilon)

"""
    AbstractAD <: AbstractAlgorithm

Abstract supertype for automatic differentiation (AD) based optimizers.
"""
abstract type AbstractAD <: AbstractAlgorithm end

"""
    AD{T} <: AbstractAD

Gradient-based optimization using automatic differentiation with fixed learning rate.
"""
struct AD{T<:Number} <: AbstractAD
    max_episode::Number
    epsilon::T
end

"""
    AD_Adam{T,N} <: AbstractAD

Gradient-based optimization using AD with the Adam optimizer.
"""
struct AD_Adam{T<:Number,N<:Number} <: AbstractAD
    max_episode::Number
    epsilon::T
    beta1::N
    beta2::N
end

"""
    AD(max_episode, epsilon, beta1, beta2)

Positional constructor — always returns `AD_Adam`.
"""
AD(max_episode, epsilon, beta1, beta2) = AD_Adam(max_episode, epsilon, beta1, beta2)
"""

    AD(;max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99, Adam::Bool=true)

Optimization algorithm: AD.
- `max_episode`: The number of episodes.
- `epsilon`: Learning rate.
- `beta1`: The exponential decay rate for the first moment estimates.
- `beta2`: The exponential decay rate for the second moment estimates.
- `Adam`: Whether or not to use Adam for updating control coefficients.
"""
AD(; max_episode = 300, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99, Adam::Bool = true) =
    Adam ? AD_Adam(max_episode, epsilon, beta1, beta2) : AD(max_episode, epsilon)

@doc raw"""
    PSO{T} <: AbstractAlgorithm

Particle Swarm Optimization (PSO).

A population-based stochastic optimizer where each particle's position
in the search space is updated based on its own best and the swarm's
global best position.

# Fields

- `max_episode`: Number of episodes (scalar or ``[\text{global},\text{local}]``).
- `p_num::Int`: Number of particles.
- `ini_particle`: Initial particle positions (or `nothing` for random init).
- `c0::T`: Inertia weight (damping).
- `c1::T`: Cognitive learning factor (personal best).
- `c2::T`: Social learning factor (global best).
"""
mutable struct PSO{T<:Number} <: AbstractAlgorithm
    max_episode::Union{Int,Vector{Int}}
    p_num::Int
    ini_particle::Union{Tuple,Nothing}
    c0::T
    c1::T
    c2::T
end

"""

    PSO(;max_episode::Union{T,Vector{T}} where {T<:Int}=[1000, 100], p_num::Number=10, ini_particle=nothing, c0::Number=1.0, c1::Number=2.0, c2::Number=2.0, seed::Number=1234)

Optimization algorithm: PSO.
- `max_episode`: The number of episodes, it accepts both integer and array with two elements.
- `p_num`: The number of particles. 
- `ini_particle`: Initial guesses of the optimization variables.
- `c0`: The damping factor that assists convergence, also known as inertia weight.
- `c1`: The exploitation weight that attracts the particle to its best previous position, also known as cognitive learning factor.
- `c2`: The exploitation weight that attracts the particle to the best position in the neighborhood, also known as social learning factor. 
"""
PSO(;
    max_episode::Union{T,Vector{T}} where {T<:Int} = [1000, 100],
    p_num::Number = 10,
    ini_particle = nothing,
    c0::Number = 1.0,
    c1::Number = 2.0,
    c2::Number = 2.0,
) = PSO(max_episode, p_num, ini_particle, c0, c1, c2)

@doc raw"""
    DE{T} <: AbstractAlgorithm

Differential Evolution (DE).

A population-based stochastic optimizer that uses mutation (difference vector)
and crossover to evolve a population toward the optimum.

# Fields

- `max_episode::Int`: Number of generations.
- `p_num::Int`: Population size.
- `ini_population`: Initial population (or `nothing` for random init).
- `c::T`: Mutation constant.
- `cr::T`: Crossover probability.
"""
mutable struct DE{T<:Number} <: AbstractAlgorithm
    max_episode::Int
    p_num::Int
    ini_population::Union{Tuple,Nothing}
    c::T
    cr::T
end

"""

    DE(;max_episode::Number=1000, p_num::Number=10, ini_population=nothing, c::Number=1.0, cr::Number=0.5, seed::Number=1234)

Optimization algorithm: DE.
- `max_episode`: The number of populations.
- `p_num`: The number of particles. 
- `ini_population`: Initial guesses of the optimization variables.
- `c`: Mutation constant.
- `cr`: Crossover constant.
"""
DE(;
    max_episode::Number = 1000,
    p_num::Number = 10,
    ini_population = nothing,
    c::Number = 1.0,
    cr::Number = 0.5,
) = DE(max_episode, p_num, ini_population, c, cr)

"""
    DDPG{R} <: AbstractAlgorithm

Deep Deterministic Policy Gradient (DDPG). A reinforcement learning algorithm
using neural networks for policy and value function approximation.

# Fields

- `max_episode::Int`: Number of training episodes.
- `layer_num::Int`: Number of neural network layers.
- `layer_dim::Int`: Number of neurons per hidden layer.
- `rng::R`: Random number generator (StableRNG).
"""
struct DDPG{R<:AbstractRNG} <: AbstractAlgorithm
    max_episode::Int
    layer_num::Int
    layer_dim::Int
    rng::R
end

"""
    DDPG(max_episode, layer_num, layer_dim)

Construct DDPG with default seed 1234.
"""
DDPG(max_episode, layer_num, layer_dim) =
    DDPG(max_episode, layer_num, layer_dim, StableRNG(1234))

"""
    DDPG(max_episode, layer_num, layer_dim, seed::Number)

Construct DDPG with a custom random seed.
"""
DDPG(max_episode, layer_num, layer_dim, seed::Number) =
    DDPG(max_episode, layer_num, layer_dim, StableRNG(seed))
"""

    DDPG(;max_episode::Int=500, layer_num::Int=3, layer_dim::Int=200, seed::Number=1234)

Optimization algorithm: DE.
- `max_episode`: The number of populations.
- `layer_num`: The number of layers (include the input and output layer).
- `layer_dim`: The number of neurons in the hidden layer.
- `seed`: Random seed.
"""
DDPG(;
    max_episode::Int = 500,
    layer_num::Int = 3,
    layer_dim::Int = 200,
    seed::Number = 1234,
) = DDPG(max_episode, layer_num, layer_dim, StableRNG(seed))

@doc raw"""
    NM{N} <: AbstractAlgorithm

Nelder-Mead simplex algorithm for derivative-free state optimization.

# Fields

- `max_episode::Int`: Maximum iterations.
- `p_num::Int`: Number of simplex vertices.
- `ini_state`: Initial simplex (or `nothing`).
- `ar::N`: Reflection coefficient (default 1.0).
- `ae::N`: Expansion coefficient (default 2.0).
- `ac::N`: Contraction coefficient (default 0.5).
- `as0::N`: Shrink coefficient (default 0.5).
"""
struct NM{N<:Number} <: AbstractAlgorithm
    max_episode::Int
    p_num::Int
    ini_state::Union{AbstractVector,Nothing}
    ar::N
    ae::N
    ac::N
    as0::N
end

"""

    NM(;max_episode::Int=1000, p_num::Int=10, nelder_mead=nothing, ar::Number=1.0, ae::Number=2.0, ac::Number=0.5, as0::Number=0.5, seed::Number=1234)

State optimization algorithm: NM.
- `max_episode`: The number of populations.
- `p_num`: The number of the input states.
- `nelder_mead`: Initial guesses of the optimization variables.
- `ar`: Reflection constant.
- `ae`: Expansion constant.
- `ac`: Constraction constant.
- `as0`: Shrink constant.
"""
NM(;
    max_episode::Int = 1000,
    p_num::Int = 10,
    nelder_mead = nothing,
    ar::Number = 1.0,
    ae::Number = 2.0,
    ac::Number = 0.5,
    as0::Number = 0.5,
) = NM(max_episode, p_num, nelder_mead, ar, ae, ac, as0)

"""
    RI <: AbstractAlgorithm

Random Initialization (RI). Generates random state guesses and selects
the one with the best objective value.

# Fields

- `max_episode::Int`: Number of random tries.
"""
struct RI <: AbstractAlgorithm
    max_episode::Int
end

"""

    RI(;max_episode::Int=300, seed::Number=1234)

State optimization algorithm: RI.
- `max_episode`: The number of episodes.
"""
RI(; max_episode::Int = 300) = RI(max_episode)

"""
    alg_type(alg)

Return the algorithm type as a symbol for dispatch purposes.
"""
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
