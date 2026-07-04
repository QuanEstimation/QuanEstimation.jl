@doc raw"""
    NVMagnetometer

Module for NV-center magnetometry based on `QuanEstimationBase`.

Provides a high-level interface for constructing NV-center estimation schemes
and evaluating quantum/classical Fisher information, Holevo Cramér-Rao bound,
and control optimization.

The NV center Hamiltonian includes:
- Zero-field splitting ``D S_z^2``
- Electron Zeeman ``g_S \mathbf{B}\cdot\mathbf{S}``
- Nuclear Zeeman ``g_I \mathbf{B}\cdot\mathbf{I}``
- Hyperfine coupling ``A_1(S_x I_x + S_y I_y) + A_2 S_z I_z``

The six-level system (``3\otimes 2``) undergoes dephasing via ``S_z``.

**Coupling**: This module has a hard dependency on `QuanEstimationBase`
([deps] in Project.toml). It extends six `QuanEstimationBase.*` functions
on its native `NVMagnetometerScheme` type (method specialization, not type
piracy). The thin root package `QuanEstimation.jl` loads both modules
simultaneously.
"""
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

@doc raw"""
    NVMagnetometerScheme <: AbstractScheme

High-level NV magnetometer scheme bundling physical parameters and I/O hooks.

# Fields

- `data::Any`: [`NVMagnetometerData`](@ref) containing all physical parameters.
- `io_hooks::Any`: I/O verbosity setting (e.g., `:verbose`).
raw"""
struct NVMagnetometerScheme <: AbstractScheme
    data::Any
    io_hooks::Any
end

@doc raw"""
    NVMagnetometerData

Physical parameters for an NV-center magnetometer.

# Fields

- `D::Float64`: Zero-field splitting (MHz, default ``2\pi\times 2870``).
- `gS::Float64`: Electron gyromagnetic ratio (MHz/mT).
- `gI::Float64`: Nuclear gyromagnetic ratio (MHz/mT).
- `A1::Float64`: Transverse hyperfine coupling (MHz).
- `A2::Float64`: Longitudinal hyperfine coupling (MHz).
- `B1::Float64`: Magnetic field ``B_x`` (mT).
- `B2::Float64`: Magnetic field ``B_y`` (mT).
- `B3::Float64`: Magnetic field ``B_z`` (mT).
- `γ::Float64`: Dephasing rate (MHz).
- `decay_opt::Vector{Matrix{ComplexF64}}`: Decay operators.
- `init_state::Vector{ComplexF64}`: Initial state ``|\psi_0\rangle``.
- `Hc::Vector{Matrix{ComplexF64}}`: Control Hamiltonians.
- `ctrl`: Control coefficients (or `nothing` for zeros).
- `tspan`: Time span for evolution.
- `M`: POVM measurement (or `nothing` for SIC-POVM).
"""
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
    ctrl::Union{Nothing, Vector{Vector{Float64}}} ##control_coefficients
    tspan::Union{Vector{Float64}, StepRangeLen} ##time_span
    M::Union{Nothing, Vector{Matrix{ComplexF64}}} ##meassurments
end

# Base.keys(t::NVMagnetometer{names...}) where {names...} = [names...]
include("show.jl")

@doc raw"""
    NVMagnetometerScheme(; D, gS, gI, A1, A2, B1, B2, B3, γ, decay_opt, init_state, Hc, ctrl, tspan, M, io_hooks)

Construct an [`NVMagnetometerScheme`](@ref) with keyword arguments for all
NV-center physical parameters. See [`NVMagnetometerData`](@ref) for field
descriptions and default values.
"""
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

@doc raw"""
    getscheme(nv::NVMagnetometerData; dynamics_hooks, state_hooks, measurement_hooks, kwargs...)

Convert NV magnetometer data into a `QuanEstimationBase.GeneralScheme`.
"""
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

"""
    getscheme(nv::NVMagnetometerScheme)

Convenience wrapper calling `getscheme(nv.data)`.
"""
getscheme(nv::NVMagnetometerScheme) = getscheme(nv.data)

@doc raw"""
    nv_dynamics_hooks(nv::NVMagnetometerData)

Build the Lindblad dynamics for the NV center.
"""
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

"""
    nv_state_hooks(init_state)

Convert the initial state to a density matrix ``\rho_0``.
"""
function nv_state_hooks(init_state::Vector{T}) where {T<:Number}
    return complex(init_state * init_state')
end

"""
    nv_state_hooks(init_state::Matrix{T}) where {T<:Number}

Density matrix input — return as-is (complex cast).
"""
function nv_state_hooks(init_state::Matrix{T}) where {T<:Number}
    return complex(init_state)
end

"""
    nv_control_hooks(ctrl, tspan)

Create control coefficient sequences; returns zeros if `ctrl` is `nothing`.
"""
function nv_control_hooks(ctrl::Nothing, tspan)
    nc = length(tspan) - 1
    return [[0.0 for _ = 1:nc] for _ = 1:3]
end

"""
    nv_control_hooks(ctrl, tspan)

Pass through pre-computed control coefficients.
"""
function nv_control_hooks(ctrl, tspan)
    return ctrl
end

"""
    nv_measurement_hooks(M)

Return the POVM measurement; defaults to SIC-POVM of dimension 6 if `nothing`.
"""
function nv_measurement_hooks(::Nothing)
    return QuanEstimationBase.SIC(6)
end

"""
    nv_measurement_hooks(M::Vector{T}) where {T<:Matrix}

Pass through pre-computed POVM measurement.
raw"""
function nv_measurement_hooks(M::Vector{T}) where {T<:Matrix}
    return M
end

@doc raw"""
    control_Hamiltonians_hook()

Return the standard control Hamiltonians ``\mathbf{S} = (S_1, S_2, S_3)``
in the ``3\otimes 2`` representation.
"""
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

## Delegation to QuanEstimationBase

"""
    QuanEstimationBase.QFIM(nv::NVMagnetometerScheme; kwargs...)

Compute the quantum Fisher information for an NV magnetometer scheme.
"""
QuanEstimationBase.QFIM(nv::NVMagnetometerScheme; kwargs...) =
    QFIM(getscheme(nv); kwargs...)

"""
    QuanEstimationBase.CFIM(nv::NVMagnetometerScheme; kwargs...)

Compute the classical Fisher information for an NV magnetometer scheme.
"""
QuanEstimationBase.CFIM(nv::NVMagnetometerScheme; kwargs...) =
    CFIM(getscheme(nv); kwargs...)

"""
    QuanEstimationBase.HCRB(nv::NVMagnetometerScheme; kwargs...)

Compute the Holevo Cramér-Rao bound for an NV magnetometer scheme.
"""
QuanEstimationBase.HCRB(nv::NVMagnetometerScheme; kwargs...) =
    HCRB(getscheme(nv); kwargs...)
    
"""
    QuanEstimationBase.optimize!(nv::NVMagnetometerScheme, opt; algorithm, objective, savefile)

Run control optimization for an NV magnetometer scheme.
Defaults to `autoGRAPE()` with `QFIM_obj()`.
"""
function QuanEstimationBase.optimize!(
    nv::NVMagnetometerScheme,
    opt;
    algorithm = autoGRAPE(),
    objective = QFIM_obj(),
    savefile = false,
)
    QuanEstimationBase.optimize!(getscheme(nv.data), opt; algorithm = algorithm, objective = objective, savefile = savefile)
end


"""
    QuanEstimationBase.error_evaluation(nv::NVMagnetometerScheme; kwargs...)

Evaluate estimation error for an NV magnetometer scheme.
"""
function QuanEstimationBase.error_evaluation(nv::NVMagnetometerScheme; kwargs...)
    QuanEstimationBase.error_evaluation(getscheme(nv); kwargs...)
end

"""
    QuanEstimationBase.error_control(nv::NVMagnetometerScheme; kwargs...)

Perform error control for an NV magnetometer scheme.
"""
function QuanEstimationBase.error_control(nv::NVMagnetometerScheme; kwargs...)
    QuanEstimationBase.error_control(getscheme(nv); kwargs...)
end

end
