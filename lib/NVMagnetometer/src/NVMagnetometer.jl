module NVMagnetometer
export NVMagnetometerScheme, NVMagnetometerData, NVMagnetometerScheme
# export nv_dynamics_hooks, nv_state_hooks, nv_measurement_hooks, nv_control_hooks, nv_measurement_hooks, nv_state_hooks, nv_dynamics_hooks
using QuanEstimationBase
using UnPack
using LinearAlgebra


const sx = [0.0 1.0; 1.0 0.0]
const sy = [0.0 -im; im 0.0]
const sz = [1.0 0.0; 0.0 -1.0]
const s1 = [0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0] / sqrt(2)
const s2 = [0.0 -im 0.0; im 0.0 -im; 0.0 im 0.0] / sqrt(2)
const s3 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0]
const Is = [kron(I(3), sx), kron(I(3), sy), kron(I(3), sz)]
const S = [kron(s1, I(2)), kron(s2, I(2)), kron(s3, I(2))]

struct NVMagnetometerScheme <: AbstractScheme
    data::Any
    io_hooks::Any
end

struct NVMagnetometerData
    D::Float64##coefficient_D
    gS::Float64 ##coefficient_gS
    gI::Float64##coefficient_gI
    A1::Float64##coefficient_A1
    A2::Float64 ##coefficient_A2
    B1::Float64 ##magnetic_field_B1
    B2::Float64 ##magnetic_field_B2
    B3::Float64 ##magnetic_field_B3
    γ::Float64 ##decay_rate_γ
    decay_opt::Vector{Matrix{ComplexF64}}##decay_operator
    init_state::Vector{ComplexF64}##ρ0
    Hc::Vector{Matrix{ComplexF64}} ##control_Hamiltonians
    ctrl::Union{Nothing,Vector{Vector{Float64}}} ##control_coefficients
    tspan::Union{Vector{Float64},StepRangeLen} ##time_span
    M::Union{Nothing,Vector{Matrix{ComplexF64}}} ##meassurments
end

# Base.keys(t::NVMagnetometer{names...}) where {names...} = [names...]
include("show.jl")

function NVMagnetometerScheme(;
    D = 2pi * 2870, # MHz
    gS = 2pi * 28.03, # MHz/mT
    gI = 2pi * 4.32 * 1e-3, # MHz/mT
    A1 = 2pi * 3.65, # MHz
    A2 = 2pi * 3.03, # MHz
    B1 = 0.5, # mT
    B2 = 0.5, # mT
    B3 = 0.5, # mT
    γ = 2pi, # MHz
    decay_opt = [S[3]],
    init_state = [1, 0, 0, 0, 1, 0] / sqrt(2),
    Hc = S,
    ctrl = nothing,
    tspan = 0.0:0.01:2.0,
    M = nothing,
    io_hooks = :verbose,
)
    data = NVMagnetometerData(
        D,
        gS,
        gI,
        A1,
        A2,
        B1,
        B2,
        B3,
        γ,
        decay_opt,
        init_state,
        Hc,
        ctrl,
        tspan,
        M,
    )
    return NVMagnetometerScheme(data, io_hooks)
end

function getscheme(
    nv::NVMagnetometerData;
    dynamics_hooks = nv_dynamics_hooks,
    state_hooks = nv_state_hooks,
    measurement_hooks = nv_measurement_hooks,
    kwargs...,
)
    @unpack init_state, M = nv

    return QuanEstimationBase.GeneralScheme(
        probe = state_hooks(init_state),
        param = dynamics_hooks(nv),
        measurement = measurement_hooks(M),
        kwargs...,
    )
end

function nv_dynamics_hooks(nv::NVMagnetometerData)
    @unpack D, gS, gI, A1, A2, B1, B2, B3, γ, init_state, ctrl, tspan, M = nv

    B = [B1, B2, B3]
    H0 = sum([
        D * kron(s3^2, I(2)),
        sum(gS * B .* S),
        sum(gI * B .* Is),
        A1 * (kron(s1, sx) + kron(s2, sy)),
        A2 * kron(s3, sz),
    ])

    # derivatives of the free Hamiltonian on B1, B2 and B3
    dH = gS * S + gI * Is
    # control Hamiltonians 
    Hc = control_Hamiltonians_hook()
    # dissipation
    decay_opt = S[3]

    ctrl0 = nv_control_hooks(ctrl, tspan)
    decay = [[decay_opt, γ]]

    return Lindblad(H0, dH, tspan, Hc, decay; ctrl = ctrl0, dyn_method = :Expm)
end

function nv_state_hooks(init_state::Vector{T}) where {T<:Number}
    return complex(init_state * init_state')
end

function nv_state_hooks(init_state::Matrix{T}) where {T<:Number}
    return complex(init_state)
end

function nv_control_hooks(ctrl::Nothing, tspan)
    nc = length(tspan) - 1
    return [[0.0 for _ = 1:nc] for _ = 1:3]
end

function nv_control_hooks(ctrl, tspan)
    return ctrl
end

function nv_measurement_hooks(::Nothing)
    return QuanEstimationBase.SIC(6)
end

function nv_measurement_hooks(M::Vector{T}) where {T<:Matrix}
    return M
end

function control_Hamiltonians_hook()
    sx = [0.0 1.0; 1.0 0.0]
    sy = [0.0 -im; im 0.0]
    sz = [1.0 0.0; 0.0 -1.0]
    s1 = [0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0] / sqrt(2)
    s2 = [0.0 -im 0.0; im 0.0 -im; 0.0 im 0.0] / sqrt(2)
    s3 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0]
    Is = [kron(I(3), sx), kron(I(3), sy), kron(I(3), sz)]
    S = [kron(s1, I(2)), kron(s2, I(2)), kron(s3, I(2))]
    return S
end

## 
QuanEstimationBase.QFIM(nv::NVMagnetometerScheme; kwargs...) =
    QFIM(getscheme(nv.data); kwargs...)
QuanEstimationBase.CFIM(nv::NVMagnetometerScheme; kwargs...) =
    CFIM(getscheme(nv.data); kwargs...)
QuanEstimationBase.HCRB(nv::NVMagnetometerScheme; kwargs...) =
    HCRB(getscheme(nv.data); kwargs...)

function QuanEstimationBase.optimize!(
    nv::NVMagnetometerScheme,
    opt;
    algorithm = autoGRAPE(),
    objective = QFIM_obj(),
    savefile = false,
)
    QuanEstimationBase.optimize!(
        getscheme(nv.data),
        opt;
        algorithm = algorithm,
        objective = objective,
        savefile = savefile,
    )
end


end # NVMagnetometer
