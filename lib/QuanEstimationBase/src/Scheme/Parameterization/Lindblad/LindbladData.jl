
"""
    AbstractDecay

Abstract supertype for decay types (NonDecay or Decay).
"""
abstract type AbstractDecay end
"""
    NonDecay <: AbstractDecay

No decay in the Lindblad dynamics.
"""
abstract type NonDecay <: AbstractDecay end
"""
    Decay <: AbstractDecay

Decay present in the Lindblad dynamics.
"""
abstract type Decay <: AbstractDecay end

"""
    AbstractQuantumDynamics

Abstract supertype for quantum dynamics classifications (Control or NonControl).
"""
abstract type AbstractQuantumDynamics end
"""
    Control <: AbstractQuantumDynamics

Controls present in the dynamics.
"""
abstract type Control <: AbstractQuantumDynamics end
"""
    NonControl <: AbstractQuantumDynamics

No controls in the dynamics.
"""
abstract type NonControl <: AbstractQuantumDynamics end

"""
    AbstractDynamicsSolver

Abstract supertype for dynamics solvers (Expm or Ode).
"""
abstract type AbstractDynamicsSolver end
"""
    Expm <: AbstractDynamicsSolver

Matrix exponential solver for dynamics evolution.
"""
abstract type Expm <: AbstractDynamicsSolver end
"""
    Ode <: AbstractDynamicsSolver

Ordinary differential equation solver for dynamics evolution.
"""
abstract type Ode <: AbstractDynamicsSolver end

"""
    DYN_METHOD_MAP

Mapping from solver symbols `:Expm` and `:Ode` to their corresponding
abstract solver types [`Expm`](@ref) and [`Ode`](@ref).
"""
const DYN_METHOD_MAP = Dict{Symbol,Type{<:AbstractDynamicsSolver}}(
    :Expm => Expm,
    :Ode  => Ode,
)

"""
    AbstractHamiltonian

Abstract supertype for Hamiltonian types.
"""
abstract type AbstractHamiltonian end

@doc raw"""
    ZeroCTRL <: Control

Zero control: ``c(t) = 0``.
"""
struct ZeroCTRL <: Control end

@doc raw"""
    LinearCTRL <: Control

Linear-in-time control: ``c(t) = k t + c_0``.

# Fields
- `k::Real=1.0`: Slope.
- `c0::Real=0.0`: Offset.
raw"""
Base.@kwdef struct LinearCTRL <: Control
    k::Real = 1.0
    c0::Real = 0.0
end

@doc raw"""
    SineCTRL <: Control

Sinusoidal control: ``c(t) = A \sin(\omega t + \phi)``.

# Fields
- `A::Real=1.0`: Amplitude.
- `ω::Real=1.0`: Angular frequency.
- `ϕ::Real=0.0`: Phase offset.
"""
Base.@kwdef struct SineCTRL <: Control
    A::Real = 1.0
    ω::Real = 1.0
    ϕ::Real = 0.0
end

"""
    SawCTRL

Sawtooth-wave control.

# Fields
- `k::Real=1.0`: Amplitude.
- `n::Real=1.0`: Number of periods.
"""
Base.@kwdef struct SawCTRL
    k::Real = 1.0
    n::Real = 1.0
end

raw"""
    TriangleCTRL

Triangle-wave control.

# Fields
- `k::Real=1.0`: Amplitude.
- `n::Real=1.0`: Number of periods.
raw"""
Base.@kwdef struct TriangleCTRL
    k::Real = 1.0
    n::Real = 1.0
end

@doc raw"""
    GaussianCTRL

Gaussian-pulse control: ``c(t) = A \exp(-(t-\mu)^2/(2\sigma))``, where ``\sigma`` is the variance.

# Fields
- `A::Real=1.0`: Amplitude.
- `μ::Real=0.0`: Center.
- `σ::Real=1.0`: Width (interpreted as variance).
raw"""
Base.@kwdef struct GaussianCTRL
    A::Real = 1.0
    μ::Real = 0.0
    σ::Real = 1.0
end

@doc raw"""
    GaussianEdgeCTRL <: Control

Gaussian-edge control: ``c(t) = A(1 - e^{-t^2/\sigma} - e^{-(t-T)^2/\sigma})``.

# Fields
- `A::Real=1.0`: Amplitude.
- `σ::Real=1.0`: Width.
raw"""
Base.@kwdef struct GaussianEdgeCTRL <: Control
    A::Real = 1.0
    σ::Real = 1.0
end


@doc raw"""
    Hamiltonian{T1,T2,N} <: AbstractHamiltonian

Parameterized Hamiltonian with free part ``H_0`` and derivatives ``\partial H``.

# Type Parameters
- `T1`: Type of ``H_0``.
- `T2`: Type of ``\partial H``.
- `N`: Number of parameters.

# Fields
- `H0`: Free Hamiltonian (matrix or function).
- `dH`: Derivatives with respect to parameters.
- `params`: Parameter values.
"""
mutable struct Hamiltonian{T1,T2,N} <: AbstractHamiltonian
    H0
    dH
    params
end

"""
    Hamiltonian(H0::T, dH::Vector{T}) where {T}

Construct a Hamiltonian from a matrix ``H_0`` and a vector of derivative matrices.
raw"""
function Hamiltonian(H0::T, dH::Vector{T}) where {T}
    N = length(dH)
    return Hamiltonian{T,Vector{T},N}(H0, dH, nothing)
end

raw"""
    Hamiltonian(H0::H, dH::D, params::NTuple{N,R}) where {H<:Function,D<:Function,N,R}

Construct a Hamiltonian from function-typed ``H_0``, ``\partial H`` and a tuple of parameters.
raw"""
function Hamiltonian(H0::H, dH::D, params::NTuple{N,R}) where {H<:Function,D<:Function,N,R}
    return Hamiltonian{typeof(H0),typeof(dH),N}(H0, dH, params)
end

raw"""
    Hamiltonian(H0::H, dH::D, params::Vector{R}) where {H<:Function,D<:Function,R}

Construct a Hamiltonian from function-typed ``H_0``, ``\partial H`` and a vector of parameters.
raw"""
function Hamiltonian(H0::H, dH::D, params::Vector{R}) where {H<:Function,D<:Function,R}
    N = length(params)
    return Hamiltonian{typeof(H0),typeof(dH),N}(H0, dH, params)
end

raw"""
    Hamiltonian(H0::H, dH::D, params::Number) where {H<:Function,D<:Function}

Construct a Hamiltonian from function-typed ``H_0``, ``\partial H`` and a scalar parameter (single-parameter case).
raw"""
function Hamiltonian(H0::H, dH::D, params::Number) where {H<:Function,D<:Function}
    N = length(params)
    return Hamiltonian{typeof(H0),typeof(dH),N}(H0, dH, [params])
end

raw"""
    Hamiltonian(H0::H, dH::D) where {H<:Function,D<:Function}

Construct a Hamiltonian from function-typed ``H_0`` and ``\partial H`` without parameter values.
"""
function Hamiltonian(H0::H, dH::D) where {H<:Function,D<:Function}
    return Hamiltonian{H,D,Nothing}(H0, dH, nothing)
end

"""
    LindbladData <: AbstractDynamicsData

Data container for Lindblad dynamics.

# Fields
- `hamiltonian::AbstractHamiltonian`: System Hamiltonian.
- `tspan::AbstractVector`: Time span for evolution.
- `decay::Union{AbstractVector,Nothing}`: Decay operators and rates.
- `Hc::Union{AbstractVector,Nothing}`: Control Hamiltonians.
- `ctrl::Union{AbstractVector,Nothing}`: Control coefficients.
- `abstol::Real`: Absolute tolerance for ODE solver.
- `reltol::Real`: Relative tolerance for ODE solver.
"""
mutable struct LindbladData <: AbstractDynamicsData
    hamiltonian::AbstractHamiltonian
    tspan::AbstractVector
    decay::Union{AbstractVector,Nothing}
    Hc::Union{AbstractVector,Nothing}
    ctrl::Union{AbstractVector,Nothing}
    abstol::Real
    reltol::Real
end

"""
    LindbladData(hamiltonian, tspan; decay=nothing, Hc=nothing, ctrl=nothing, abstol=1e-6, reltol=1e-3)

Keyword-argument constructor for [`LindbladData`](@ref).

# Arguments
- `hamiltonian`: System Hamiltonian ([`Hamiltonian`](@ref) struct or raw matrix/function).
- `tspan::AbstractVector`: Time span for evolution.
- `decay=nothing`: Decay operators and rates.
- `Hc=nothing`: Control Hamiltonians.
- `ctrl=nothing`: Control coefficients.
- `abstol=1e-6`: Absolute tolerance for ODE solver.
- `reltol=1e-3`: Relative tolerance for ODE solver.
"""
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
"""
    Lindblad(ham::Hamiltonian, tspan; dyn_method=:Ode)

Construct Lindblad dynamics (NoDecay, NoControl) from a `Hamiltonian` struct.
raw"""
function Lindblad(
    ham::Hamiltonian,
    tspan::AbstractVector;
    dyn_method::Union{Symbol,String} = :Ode,
)
    p = ham.params
    return LindbladDynamics{typeof(ham),NonDecay,NonControl,DYN_METHOD_MAP[Symbol(dyn_method)],Some}(
        LindbladData(ham, tspan),
        p,
    )
end

raw"""
    Lindblad(H0::T, dH::D, tspan; dyn_method=:Ode) where {T,D}

Construct Lindblad dynamics (NoDecay, NoControl) from raw ``H_0`` and ``\partial H``.
"""
function Lindblad(
    H0::T,
    dH::D,
    tspan::AbstractVector;
    dyn_method::Union{Symbol,String} = :Ode,
) where {T,D}
    ham = Hamiltonian(H0, dH)
    return LindbladDynamics{typeof(ham),NonDecay,NonControl,DYN_METHOD_MAP[Symbol(dyn_method)],Nothing}(
        LindbladData(ham, tspan),
        nothing,
    )
end

Decay, NonControl,
"""
    Lindblad(ham::Hamiltonian, tspan, decay; dyn_method=:Ode)

Construct Lindblad dynamics (Decay, NoControl) from a `Hamiltonian` struct with decay.
raw"""
function Lindblad(
    ham::Hamiltonian,
    tspan::AbstractVector,
    decay::AbstractVector;
    dyn_method::Union{Symbol,String} = :Ode,
)
    p = ham.params
    return LindbladDynamics{typeof(ham),Decay,NonControl,DYN_METHOD_MAP[Symbol(dyn_method)],Some}(
        LindbladData(ham, tspan; decay = decay),
        p,
    )
end

raw"""
    Lindblad(H0::H, dH::D, tspan, decay; dyn_method=:Ode) where {H,D}

Construct Lindblad dynamics (Decay, NoControl) from raw ``H_0`` and ``\partial H`` with decay.
raw"""
function Lindblad(
    H0::H,
    dH::D,
    tspan::AbstractVector,
    decay::AbstractVector;
    dyn_method::Union{Symbol,String} = :Ode,
) where {H,D}
    ham = Hamiltonian(H0, dH)
    return LindbladDynamics{typeof(ham),Decay,NonControl,DYN_METHOD_MAP[Symbol(dyn_method)],Nothing}(
        LindbladData(ham, tspan; decay = decay),
        nothing,
    )
end

# NonDecay, Control
raw"""
    Lindblad(H0::H, dH::D, tspan, Hc::Vector{M}; ctrl=ZeroCTRL(), dyn_method=:Ode) where {H,D,M<:AbstractMatrix}

Construct Lindblad dynamics (NoDecay, Control) from raw ``H_0`` and ``\partial H`` with control Hamiltonians.
"""
function Lindblad(
    H0::H,
    dH::D,
    tspan::AbstractVector,
    Hc::Vector{M};
    ctrl = ZeroCTRL(),
    dyn_method::Union{Symbol,String} = :Ode,
) where {H,D,M<:AbstractMatrix}
    ham = Hamiltonian(H0, dH)
    return LindbladDynamics{typeof(ham),NonDecay,Control,DYN_METHOD_MAP[Symbol(dyn_method)],Nothing}(
        LindbladData(ham, tspan; Hc = complex.(Hc), ctrl = init_ctrl(Hc, tspan, ctrl)),
        nothing,
    )
end

"""
    Lindblad(ham::Hamiltonian, tspan, Hc::Vector{M}; ctrl=ZeroCTRL(), dyn_method=:Ode) where {M<:AbstractMatrix}

Construct Lindblad dynamics (NoDecay, Control) from a `Hamiltonian` struct with control Hamiltonians.
raw"""
function Lindblad(
    ham::Hamiltonian,
    tspan::AbstractVector,
    Hc::Vector{M};
    ctrl = ZeroCTRL(),
    dyn_method::Union{Symbol,String} = :Ode,
) where {M<:AbstractMatrix}
    p = ham.params
    return LindbladDynamics{typeof(ham),NonDecay,Control,DYN_METHOD_MAP[Symbol(dyn_method)],Some}(
        LindbladData(ham, tspan; Hc = complex.(Hc), ctrl = init_ctrl(Hc, tspan, ctrl)),
        p,
    )
end

# Decay, Control
raw"""
    Lindblad(H0::H, dH::D, tspan, Hc::Vector{M}, decay; ctrl=ZeroCTRL(), dyn_method=:Ode) where {H,D,M<:AbstractMatrix}

Construct Lindblad dynamics (Decay, Control) from raw ``H_0`` and ``\partial H`` with control and decay.
"""
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
    return LindbladDynamics{typeof(ham),Decay,Control,DYN_METHOD_MAP[Symbol(dyn_method)],Nothing}(
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


"""
    Lindblad(ham::Hamiltonian, tspan, Hc::Vector{M}, decay; ctrl=ZeroCTRL(), dyn_method=:Ode) where {M<:AbstractMatrix}

Construct Lindblad dynamics (Decay, Control) from a `Hamiltonian` struct with control and decay.
"""
function Lindblad(
    ham::Hamiltonian,
    tspan::AbstractVector,
    Hc::Vector{M},
    decay::AbstractVector;
    ctrl = ZeroCTRL(),
    dyn_method::Union{Symbol,String} = :Ode,
) where {M<:AbstractMatrix}
    p = ham.params
    return LindbladDynamics{typeof(ham),Decay,Control,DYN_METHOD_MAP[Symbol(dyn_method)],Some}(
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

"""
    init_ctrl(Hc, tspan, ctrl::AbstractVector)

Return the control sequence directly (already a vector of coefficient values).
"""
init_ctrl(Hc, tspan, ctrl::AbstractVector) = ctrl
"""
    init_ctrl(Hc, tspan, ::ZeroCTRL)

Generate an all-zero control sequence: ``c(t) = 0``.
raw"""
init_ctrl(Hc, tspan, ::ZeroCTRL) = [zero(tspan[1:end-1]) for _ in eachindex(Hc)]
@doc raw"""
    init_ctrl(Hc, tspan, ctrl::LinearCTRL)

Generate linear-in-time control coefficients: ``c(t) = k t + c_0``.
raw"""
init_ctrl(Hc, tspan, ctrl::LinearCTRL) =
    [[ctrl.k * t .+ ctrl.c0 for t in tspan[1:end-1]] for _ in eachindex(Hc)]
@doc raw"""
    init_ctrl(Hc, tspan, ctrl::SineCTRL)

Generate sinusoidal control coefficients: ``c(t) = A \sin(\omega t + \phi)``.
"""
init_ctrl(Hc, tspan, ctrl::SineCTRL) =
    [[ctrl.A * sin(ctrl.ω * t .+ ctrl.ϕ) for t in tspan[1:end-1]] for _ in eachindex(Hc)]
"""
    init_ctrl(Hc, tspan, ctrl::SawCTRL)

Generate sawtooth-wave control coefficients over `tspan`.
"""
function init_ctrl(Hc, tspan, ctrl::SawCTRL)
    ramp = (tspan[end] - tspan[1]) / ctrl.n
    return [
        [2 * ctrl.k * (t / ramp - floor(1 / 2 + t / ramp)) for t in tspan[1:end-1]] for
        _ in eachindex(Hc)
    ]
end
"""
    init_ctrl(Hc, tspan, ctrl::TriangleCTRL)

Generate triangle-wave control coefficients over `tspan`.
raw"""
function init_ctrl(Hc, tspan, ctrl::TriangleCTRL)
    ramp = (tspan[end] - tspan[1]) / ctrl.n
    return [
        [
            2 * abs(2 * ctrl.k * (t / ramp - floor(1 / 2 + t / ramp))) - 1 for
            t in tspan[1:end-1]
        ] for _ in eachindex(Hc)
    ]
end

@doc raw"""
    init_ctrl(Hc, tspan, ctrl::GaussianCTRL)

Generate Gaussian-pulse control coefficients: ``c(t) = A \exp(-(t-\mu)^2 / (2\sigma))``, where ``\sigma`` is interpreted as the variance.
raw"""
init_ctrl(Hc, tspan, ctrl::GaussianCTRL) = [
    [ctrl.A * exp(-((t - ctrl.μ)^2) / (2 * ctrl.σ)) for t in tspan[1:end-1]] for
    _ in eachindex(Hc)
]
@doc raw"""
    init_ctrl(Hc, tspan, ctrl::GaussianEdgeCTRL)

Generate double-sided Gaussian-edge control coefficients:
``c(t) = A \bigl(1 - e^{-t^2/\sigma} - e^{-(t - T)^2/\sigma}\bigr)``,
where ``T`` is the total evolution time and ``\sigma`` is the width parameter.
"""
init_ctrl(Hc, tspan, ctrl::GaussianEdgeCTRL) = [
    [
        ctrl.A * (1 - exp(-t^2 / ctrl.σ) - exp(-(t - (tspan[end] - tspan[1]))^2 / ctrl.σ)) for
        t in tspan[1:end-1]
    ] for _ in eachindex(Hc)
]


"""
    para_type(::LindbladDynamics{...})

Return `:single_para` if the Hamiltonian has exactly one parameter (``N=1``), otherwise `:multi_para`.
"""
para_type(::LindbladDynamics{Hamiltonian{T,D,1},DC,C,S,P}) where {T,D,DC,C,S,P} = :single_para
para_type(::LindbladDynamics{Hamiltonian{T,D,N},DC,C,S,P}) where {T,D,N,DC,C,S,P} = :multi_para

"""
    get_param_num(::Type{LindbladDynamics{H,D,C,S,P}})

Extract the number of parameters from the `LindbladDynamics` type (delegates to the Hamiltonian type).
"""
get_param_num(::Type{LindbladDynamics{H,D,C,S,P}}) where {H,D,C,S,P} = get_param_num(H)
"""
    get_param_num(::Type{Hamiltonian{H,D,N}})

Extract the number of parameters `N` from the `Hamiltonian` type.
"""
get_param_num(::Type{Hamiltonian{H,D,N}}) where {H,D,N} = N
"""
    get_param_num(::Scheme{S,L,M,E}) where {L<:AbstractDynamics}

Extract the number of parameters from a `Scheme` containing Lindblad dynamics.
"""
get_param_num(::Scheme{S,L,M,E}) where {S,L<:AbstractDynamics,M,E} = get_param_num(L)

"""
    get_dim(ham::Hamiltonian{<:AbstractMatrix})

Return the Hilbert space dimension from the size of ``H_0`` (matrix case).
"""
get_dim(ham::Hamiltonian{H,DH,N}) where {H<:AbstractMatrix,DH,N} = size(ham.H0, 1)
"""
    get_dim(data::Hamiltonian{<:Function})

Return the Hilbert space dimension by evaluating ``H_0`` at an arbitrary argument (function case).
"""
get_dim(data::Hamiltonian{F,DF,N}) where {F<:Function,DF,N}= size(data.H0(0.0), 1)
"""
    get_dim(data::LindbladData)

Return the Hilbert space dimension from the stored `LindbladData`.
"""
get_dim(data::LindbladData) = get_dim(data.hamiltonian)
"""
    get_dim(dynamics::LindbladDynamics)

Return the Hilbert space dimension, delegating to the dynamics data.
"""
get_dim(dynamics::LindbladDynamics) = get_dim(dynamics.data.hamiltonian)

"""
    get_ctrl_num(data::LindbladData)

Return the number of control Hamiltonians (length of `Hc`).
"""
get_ctrl_num(data::LindbladData) = length(data.Hc)
"""
    get_ctrl_num(dynamics::LindbladDynamics)

Return the number of control Hamiltonians, delegating to the dynamics data.
"""
get_ctrl_num(dynamics::LindbladDynamics) = get_ctrl_num(dynamics.data)
"""
    get_ctrl_num(scheme::Scheme)

Extract the number of control Hamiltonians from a `Scheme`.
"""
get_ctrl_num(scheme::Scheme) = get_ctrl_num(scheme.Parameterization)

"""
    get_ctrl_length(data::LindbladData)

Return the length of the control coefficient sequence (number of time steps).
"""
get_ctrl_length(data::LindbladData) = length(data.ctrl[1])
"""
    get_ctrl_length(dynamics::LindbladDynamics)

Return the control sequence length, delegating to the dynamics data.
"""
get_ctrl_length(dynamics::LindbladDynamics) = get_ctrl_length(dynamics.data)
"""
    get_ctrl_length(scheme::Scheme)

Extract the control sequence length from a `Scheme`.
"""
get_ctrl_length(scheme::Scheme) = get_ctrl_length(scheme.Parameterization)
