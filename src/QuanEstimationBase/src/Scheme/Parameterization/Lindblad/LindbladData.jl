## TODO: reconstruct dynamicsdata structs

abstract type AbstractDynamicsData end
abstract type AbstractDecay end
abstract type NonDecay <: AbstractDecay end
# struct Decay <: AbstractDecay 
#     decay_opt::AbstractVector
#     γ::AbstractVector
# end
abstract type Decay <: AbstractDecay end

abstract type AbstractQuantumDynamics end
abstract type NonControl <: AbstractQuantumDynamics end
# struct Control <: AbstractQuantumDynamics
#     Hc::AbstractVector
#     ctrl::AbstractVector
# end
abstract type Control <: AbstractQuantumDynamics end 

abstract type AbstractDynamicsSolver end
abstract type Expm <: AbstractDynamicsSolver end
abstract type Ode <: AbstractDynamicsSolver end

abstract type AbstractHamiltonian end 
struct Hamiltonian{T1,T2,N} <: AbstractHamiltonian
    H0
    dH
    params
end

function Hamiltonian(H0::T, dH::Vector{T}) where T
    N = length(dH)
    return Hamiltonian{T, Vector{T}, N}(H0, dH, nothing)
end

function Hamiltonian(H0::Function, dH::Function, params::NTuple{N, R}) where {N, R}
    return Hamiltonian{typeof(H0), typeof(dH), N}(H0, dH, params)
end

function Hamiltonian(H0::Function, dH::Function, params::Vector{R}) where {R}
    N = length(params)
    return Hamiltonian{typeof(H0), typeof(dH), N}(H0, dH, params)
end



struct LindbladData <: AbstractDynamicsData
    hamiltonian::AbstractHamiltonian
    tspan::AbstractVector
    decay::Union{AbstractVector, Nothing}
    Hc::Union{AbstractVector, Nothing}
    ctrl::Union{AbstractVector, Nothing}
    abstol::Real
    reltol::Real
end

LindbladData(hamiltonian, tspan; decay=nothing, Hc=nothing, ctrl=nothing, abstol=1e-6, reltol=1e-3) = LindbladData(hamiltonian, tspan, decay, Hc, ctrl, abstol, reltol)

# Constructor of Lindblad dynamics
# NonDecay, NonControl
function Lindblad(ham::Hamiltonian,  tspan::AbstractVector; dyn_method::Union{Symbol, String}=:Ode)
    return Lindblad{typeof(ham), NonDecay, NonControl, eval(Symbol(dyn_method))}(LindbladData(ham, tspan), ham.params)
end

function Lindblad(H0::T, dH::D, tspan::AbstractVector; dyn_method::Union{Symbol, String}=:Ode) where {T, D} 
    ham = Hamiltonian(H0, dH)
    return Lindblad{typeof(ham), NonDecay, NonControl, eval(Symbol(dyn_method))}(LindbladData(ham, tspan), nothing)
end

# Decay, NonControl,

function Lindblad( ham::Hamiltonian, tspan::AbstractVector, decay::AbstractVector; dyn_method::Union{Symbol, String}=:Ode,)
    return Lindblad{typeof(ham), Decay, NonControl, eval(Symbol(dyn_method))}(LindbladData(ham, tspan; decay=decay), nothing)
end

function Lindblad( H0::H, dH::D, tspan::AbstractVector, decay::AbstractVector; dyn_method::Union{Symbol, String}=:Ode,) where {H, D}
    ham = Hamiltonian(H0, dH)
    return Lindblad{typeof(ham), Decay, NonControl, eval(Symbol(dyn_method))}(LindbladData(ham, tspan; decay=decay), nothing)
end

# NonDecay, Control
function Lindblad( H0::H, dH::D, tspan::AbstractVector, Hc::AbstractVector, ctrl::AbstractVector; dyn_method::Union{Symbol, String}=:Ode,) where {H, D}
    ham = Hamiltonian(H0, dH)
    return Lindblad{typeof(ham), NonDecay, Control, eval(Symbol(dyn_method))}(LindbladData(
        ;hamiltonian=ham, tspan=tspan,  Hc = Hc, ctrl=ctrl, 
    ), nothing)
end

function Lindblad( ham::Hamiltonian, tspan::AbstractVector, Hc::AbstractVector, ctrl::AbstractVector; dyn_method::Union{Symbol, String}=:Ode,)
    return Lindblad{typeof(ham), NonDecay, Control, eval(Symbol(dyn_method))}(LindbladData(
        ham, tspan;  Hc = Hc, ctrl=ctrl, 
    ), nothing)
end

# Decay, Control
function Lindblad(H0::H, dH::D, tspan::AbstractVector, Hc::AbstractVector, ctrl::AbstractVector,  decay::AbstractVector; dyn_method::Union{Symbol, String}=:Ode, ) where {H, D}
    ham = Hamiltonian(H0, dH)
    return Lindblad{typeof(ham), Decay, Control, eval(Symbol(dyn_method))}(LindbladData(
        ham, tspan; decay = decay, Hc = Hc, ctrl = ctrl
    ), nothing)
end

function Lindblad(ham::Hamiltonian, tspan::AbstractVector, Hc::AbstractVector, ctrl::AbstractVector,  decay::AbstractVector; dyn_method::Union{Symbol, String}=:Ode, ) 
    return Lindblad{typeof(ham), Decay, Control, eval(Symbol(dyn_method))}(LindbladData(
        ham, tspan; decay = decay, Hc = Hc, ctrl = ctrl
    ), nothing)
end

para_type(::Lindblad{Hamiltonian{T,D, 1}, TS}) where {T,D,TS} = :single_para
para_type(::Lindblad{Hamiltonian{T,D, N}, TS}) where {T,D,N,TS} = :multi_para

get_param_num(::Type{Lindblad{H,TS}}) where {H,TS} = get_param_num(H)
get_param_num(::Type{Hamiltonian{H,D,N}}) where {H,D,N} = N
# get_param_num(::Scheme{S,L,M,E}) where {S,L,M,E} = get_param_num(L)
# get_param_num(::Type{T}) where {T} = get_param_num(T)