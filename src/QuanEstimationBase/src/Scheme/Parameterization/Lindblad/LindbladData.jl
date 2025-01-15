## TODO: reconstruct dynamicsdata structs

abstract type AbstractDecay end
abstract type NonDecay <: AbstractDecay end
abstract type Decay <: AbstractDecay end

abstract type AbstractQuantumDynamics end
abstract type Control <: AbstractQuantumDynamics end
abstract type NonControl <: AbstractQuantumDynamics end

abstract type AbstractDynamicsSolver end
abstract type Expm <: AbstractDynamicsSolver end
abstract type Ode <: AbstractDynamicsSolver end

abstract type AbstractHamiltonian end

struct ZeroCTRL <: Control end

Base.@kwdef struct LinearCTRL <: Control
    k::Real = 1.0
    c0::Real = 0.0
end

Base.@kwdef struct SineCTRL <: Control
    A::Real = 1.0
    ω::Real = 1.0
    ϕ::Real = 0.0
end

Base.@kwdef struct SawCTRL
    k::Real = 1.0
    n::Real = 1.0
end

Base.@kwdef struct TriangleCTRL
    k::Real = 1.0
    n::Real = 1.0
end

Base.@kwdef struct GaussianCTRL
    A::Real = 1.0
    μ::Real = 0.0
    σ::Real = 1.0
end

Base.@kwdef struct GaussianEdgeCTRL <: Control
    A::Real = 1.0
    σ::Real = 1.0
end


mutable struct Hamiltonian{T1,T2,N} <: AbstractHamiltonian
    H0
    dH
    params
end

function Hamiltonian(H0::T, dH::Vector{T}) where {T}
    N = length(dH)
    return Hamiltonian{T,Vector{T},N}(H0, dH, nothing)
end

function Hamiltonian(H0::H, dH::D, params::NTuple{N,R}) where {H<:Function,D<:Function,N,R}
    return Hamiltonian{typeof(H0),typeof(dH),N}(H0, dH, params)
end

function Hamiltonian(H0::H, dH::D, params::Vector{R}) where {H<:Function,D<:Function,R}
    N = length(params)
    return Hamiltonian{typeof(H0),typeof(dH),N}(H0, dH, params)
end

function Hamiltonian(H0::H, dH::D, params::Number) where {H<:Function,D<:Function}
    N = length(params)
    return Hamiltonian{typeof(H0),typeof(dH),N}(H0, dH, [params])
end

function Hamiltonian(H0::H, dH::D) where {H<:Function,D<:Function}
    return Hamiltonian{H,D,Nothing}(H0, dH, nothing)
end

mutable struct LindbladData <: AbstractDynamicsData
    hamiltonian::AbstractHamiltonian
    tspan::AbstractVector
    decay::Union{AbstractVector,Nothing}
    Hc::Union{AbstractVector,Nothing}
    ctrl::Union{AbstractVector,Nothing}
    abstol::Real
    reltol::Real
end

LindbladData(
    hamiltonian,
    tspan;
    decay = nothing,
    Hc = nothing,
    ctrl = nothing,
    abstol = 1e-6,
    reltol = 1e-3,
) = LindbladData(hamiltonian, tspan, decay, Hc, ctrl, abstol, reltol)

# Constructor of Lindblad dynamics
# NonDecay, NonControl
function Lindblad(
    ham::Hamiltonian,
    tspan::AbstractVector;
    dyn_method::Union{Symbol,String} = :Ode,
)
    p = ham.params
    return Lindblad{typeof(ham),NonDecay,NonControl,eval(Symbol(dyn_method)),Some}(
        LindbladData(ham, tspan),
        p,
    )
end

function Lindblad(
    H0::T,
    dH::D,
    tspan::AbstractVector;
    dyn_method::Union{Symbol,String} = :Ode,
) where {T,D}
    ham = Hamiltonian(H0, dH)
    return Lindblad{typeof(ham),NonDecay,NonControl,eval(Symbol(dyn_method)),Nothing}(
        LindbladData(ham, tspan),
        nothing,
    )
end

# Decay, NonControl,

function Lindblad(
    ham::Hamiltonian,
    tspan::AbstractVector,
    decay::AbstractVector;
    dyn_method::Union{Symbol,String} = :Ode,
)
    p = ham.params
    return Lindblad{typeof(ham),Decay,NonControl,eval(Symbol(dyn_method)),Some}(
        LindbladData(ham, tspan; decay = decay),
        p,
    )
end

function Lindblad(
    H0::H,
    dH::D,
    tspan::AbstractVector,
    decay::AbstractVector;
    dyn_method::Union{Symbol,String} = :Ode,
) where {H,D}
    ham = Hamiltonian(H0, dH)
    return Lindblad{typeof(ham),Decay,NonControl,eval(Symbol(dyn_method)),Nothing}(
        LindbladData(ham, tspan; decay = decay),
        nothing,
    )
end

# NonDecay, Control
function Lindblad(
    H0::H,
    dH::D,
    tspan::AbstractVector,
    Hc::Vector{M};
    ctrl = ZeroCTRL(),
    dyn_method::Union{Symbol,String} = :Ode,
) where {H,D,M<:AbstractMatrix}
    ham = Hamiltonian(H0, dH)
    return Lindblad{typeof(ham),NonDecay,Control,eval(Symbol(dyn_method)),Nothing}(
        LindbladData(ham, tspan; Hc = complex.(Hc), ctrl = init_ctrl(Hc, tspan, ctrl)),
        nothing,
    )
end

function Lindblad(
    ham::Hamiltonian,
    tspan::AbstractVector,
    Hc::Vector{M};
    ctrl = ZeroCTRL(),
    dyn_method::Union{Symbol,String} = :Ode,
) where {M<:AbstractMatrix}
    p = ham.params
    return Lindblad{typeof(ham),NonDecay,Control,eval(Symbol(dyn_method)),Some}(
        LindbladData(ham, tspan; Hc = complex.(Hc), ctrl = init_ctrl(Hc, tspan, ctrl)),
        p,
    )
end

# Decay, Control
function Lindblad(
    H0::H,
    dH::D,
    tspan::AbstractVector,
    Hc::Vector{M},
    decay::AbstractVector;
    ctrl = ZeroCTRL(),
    dyn_method::Union{Symbol,String} = :Ode,
) where {H,D,M<:AbstractMatrix}
    ham = Hamiltonian(H0, dH)
    return Lindblad{typeof(ham),Decay,Control,eval(Symbol(dyn_method)),Nothing}(
        LindbladData(
            ham,
            tspan;
            decay = decay,
            Hc = complex.(Hc),
            ctrl = init_ctrl(Hc, tspan, ctrl),
        ),
        nothing,
    )
end


function Lindblad(
    ham::Hamiltonian,
    tspan::AbstractVector,
    Hc::Vector{M},
    decay::AbstractVector;
    ctrl = ZeroCTRL(),
    dyn_method::Union{Symbol,String} = :Ode,
) where {M<:AbstractMatrix}
    p = ham.params
    return Lindblad{typeof(ham),Decay,Control,eval(Symbol(dyn_method)),Some}(
        LindbladData(
            ham,
            tspan;
            decay = decay,
            Hc = complex.(Hc),
            ctrl = init_ctrl(Hc, tspan, ctrl),
        ),
        p,
    )
end

init_ctrl(Hc, tspan, ctrl::AbstractVector) = ctrl
init_ctrl(Hc, tspan, ::ZeroCTRL) = [zero(tspan[1:end-1]) for _ in eachindex(Hc)]
init_ctrl(Hc, tspan, ctrl::LinearCTRL) =
    [[ctrl.k * t .+ ctrl.c0 for t in tspan[1:end-1]] for _ in eachindex(Hc)]
init_ctrl(Hc, tspan, ctrl::SineCTRL) =
    [[ctrl.A * sin(ctrl.ω * t .+ ctrl.ϕ) for t in tspan[1:end-1]] for _ in eachindex(Hc)]
function init_ctrl(Hc, tspan, ctrl::SawCTRL)
    ramp = (tspan[end] - tspan[1]) / ctrl.n
    return [
        [2 * ctrl.k * (t / ramp - floor(1 / 2 + t / ramp)) for t in tspan[1:end-1]] for
        _ in eachindex(Hc)
    ]
end
function init_ctrl(Hc, tspan, ctrl::TriangleCTRL)
    ramp = (tspan[end] - tspan[1]) / ctrl.n
    return [
        [
            2 * abs(2 * ctrl.k * (t / ramp - floor(1 / 2 + t / ramp))) - 1 for
            t in tspan[1:end-1]
        ] for _ in eachindex(Hc)
    ]
end

init_ctrl(Hc, tspan, ctrl::GaussianCTRL) = [
    [ctrl.A * exp(-((t - ctrl.μ)^2) / (2 * ctrl.σ)) for t in tspan[1:end-1]] for
    _ in eachindex(Hc)
]
init_ctrl(Hc, tspan, ctrl::GaussianEdgeCTRL) = [
    [
        ctrl.A * (1 - exp(-t^2 / ctrl.σ) - exp(-(t - (tspan[end] - tspan[1]))^2 / ctrl.σ)) for
        t in tspan[1:end-1]
    ] for _ in eachindex(Hc)
]


para_type(::Lindblad{Hamiltonian{T,D,1},TS}) where {T,D,TS} = :single_para
para_type(::Lindblad{Hamiltonian{T,D,N},TS}) where {T,D,N,TS} = :multi_para

get_param_num(::Type{Lindblad{H,TS}}) where {H,TS} = get_param_num(H)
get_param_num(::Type{Hamiltonian{H,D,N}}) where {H,D,N} = N
# get_param_num(::Scheme{S,L,M,E}) where {S,L,M,E} = get_param_num(L)
# get_param_num(::Type{T}) where {T} = get_param_num(T)

get_dim(ham::Hamiltonian) = size(ham.H0, 1)
get_dim(data::LindbladData) = get_dim(data.hamiltonian)
get_dim(dynamics::Lindblad) = get_dim(dynamics.data.hamiltonian)

get_ctrl_num(data::LindbladData) = length(data.Hc)
get_ctrl_num(dynamics::Lindblad) = get_ctrl_num(dynamics.data)
get_ctrl_num(scheme::Scheme) = get_ctrl_num(scheme.Parameterization)

get_ctrl_length(data::LindbladData) = length(data.ctrl[1])
get_ctrl_length(dynamics::Lindblad) = get_ctrl_length(dynamics.data)
get_ctrl_length(scheme::Scheme) = get_ctrl_length(scheme.Parameterization)
