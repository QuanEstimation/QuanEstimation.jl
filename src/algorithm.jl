abstract type AbstractAlgorithm end

abstract type AbstractUpdateType end
abstract type GradDescent<: AbstractUpdateType end
abstract type Adam <: AbstractUpdateType end

struct GRAPE{U} <: AbstractAlgorithm
    max_episode::Number
    update_type::Symbol
end

GRAPE(max_episode::Number,update_type::Symbol) = GRAPE{eval(update_type)}(max_episode,update_type)
struct AD{U} <: AbstractAlgorithm
    max_episode::Number
    update_type::Symbol
end

AD(max_episode::Number,update_type::Symbol) = AD{eval(update_type)}(max_episode,update_type)

struct PSO <: AbstractAlgorithm
    max_episode::Number
    particle_num::Number
    ini_particle::AbstractVector
    c0::Number
    c1::Number
    c2::Number
    rng::AbstractRNG
end

PSO(max_episode,para_num,ini_particle,c0,c1,c2) = PSO(max_episode,para_num,ini_particle,c0,c1,c2,GLOBAL_RNG)
PSO(max_episode,para_num, ini_particle,c0,c1,c2,seed::Number) = PSO(max_episode,para_num,ini_particle,c0,c1,c2,MersenneTwister(seed))

struct DE <: AbstractAlgorithm
    max_episode::Number
    popsize::Number
    ini_population::AbstractVector
    c::Number
    cr::Number
    rng::AbstractRNG = GLOBAL_RNG
end

DE(max_episode, popsize, c, cr) = DE(max_episode,popsize,c,cr,GLOBAL_RNG)
DE(max_episode, popsize, ini_population,c, cr,seed::Number) = DE(max_episode,popsize,ini_population,c,cr,MersenneTwister(seed))

struct DDPG <: AbstractAlgorithm
    max_episode::Number
    layer_num::Number
    layer_dim::Number
    rng::AbstractRNG = GLOBAL_RNG
end

DDPG(max_episode,layer_num,layer_dim) = DDPG(max_episode,layer_num,layer_dim,GLOBAL_RNG)
DDPG(max_episode,layer_num,layer_dim,seed::Number) = DDPG(max_episode,layer_num,layer_dim,MersenneTwister(seed))

struct NM <: AbstractAlgorithm
    max_episode::Number
    state_num::Number
    ini_state::AbstractVector
    ar::Number
    ae::Number
    ac::Number
    as0::Number
    rng::AbstractRNG = GLOBAL_RNG
end

NM(max_episode,state_num, ini_state,ar,ae,ac,as0) = NM(max_episode,state_num,ini_state,ar,ae,ac,as0,GLOBAL_RNG)
NM(max_episode,state_num,ini_state,ar,ae,ac,as0,seed:Number) = NM(max_episode,state_num,ini_state,ar,ae,ac,as0,MersenneTwister(seed))