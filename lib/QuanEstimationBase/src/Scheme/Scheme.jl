"""
    AbstractScheme

Abstract supertype for all quantum estimation schemes.
raw"""
abstract type AbstractScheme end

include("StatePreparation/StatePreparation.jl")

@doc raw"""
    Scheme{S,P,M,E} <: AbstractScheme

A complete quantum parameter estimation scheme.

Bundles the four essential components of a quantum estimation experiment:

1. **State preparation** (`S`): Initial quantum state ``\rho_0`` or ``|\psi_0\rangle``.
2. **Parameterization** (`P`): Dynamics encoding the unknown parameters ``\mathbf{x}``
   into the quantum state ``\rho(\mathbf{x})``.
3. **Measurement** (`M`): POVM ``\{\Pi_y\}`` describing the measurement.
4. **Estimation strategy** (`E`): Estimator ``\hat{\mathbf{x}}(y)`` or adaptive protocol.

# Type Parameters

- `S`: State type (`Ket` or `DensityMatrix`).
- `P`: Parameterization type (e.g., `LindbladDynamics`, `Kraus`).
- `M`: Measurement POVM type (`SIC_POVM` or `POVM`).
- `E`: Estimation strategy type (`Strategy`, `AdaptiveStrategy`).

# Accessors

- [`state_data`](@ref): Probe state.
- [`param_data`](@ref): Dynamics parameterization.
- [`meas_data`](@ref): Measurement POVM.
- [`strat_data`](@ref): Estimation strategy.
raw"""
struct Scheme{S,P,M,E} <: AbstractScheme
    StatePreparation::GeneralState{S}
    Parameterization::P
    Measurement::M
    EstimationStrategy::E
end

include("Measurement/Measurement.jl")
include("EstimationStrategy/EstimationStrategy.jl")
include("Parameterization/Parameterization.jl")
include("error_evaluation.jl")
include("error_control.jl")
include("show.jl")

@doc raw"""
    GeneralScheme(; probe, param, measurement, strat, x, p, dp)

Construct a complete estimation scheme with sensible defaults.

If `measurement` is not provided, defaults to a rank-1 SIC-POVM.
If `strat` is not provided, a [`GeneralEstimation`](@ref) is constructed from
the prior parameters `x`, `p`, `dp`.

# Arguments

- `probe`: Initial quantum state (density matrix ``\rho_0`` or ket ``|\psi_0\rangle``).
- `param`: Dynamics parameterization (e.g., `Lindblad(H0, dH, tspan)`).
- `measurement`: POVM elements ``\{\Pi_y\}`` (optional; defaults to SIC-POVM).
- `strat`: Estimation strategy (optional; defaults to `GeneralEstimation(x, p, dp)`).
- `x`: Parameter grid points.
- `p`: Prior distribution.
- `dp`: Derivatives of the prior.

# Returns

- `Scheme`: A fully specified quantum estimation scheme.
"""
function GeneralScheme(;
    probe = nothing,
    param = nothing,
    measurement = nothing,
    strat = nothing,
    x = nothing,
    p = nothing,
    dp = nothing,
)
    return Scheme(
        GeneralState(probe),
        param,
        isnothing(measurement) ? GeneralMeasurement(SIC(get_dim(param))) :
        GeneralMeasurement(measurement),
        isnothing(strat) ? GeneralEstimation(x, p, dp) : strat,
    )
end

"""
    state_data(scheme::Scheme)

Return the probe state data.
"""
state_data(scheme::Scheme) = scheme.StatePreparation.data

"""
    param_data(scheme::Scheme)

Return the dynamics parameterization data.
"""
param_data(scheme::Scheme) = scheme.Parameterization.data

"""
    meas_data(scheme::Scheme)

Return the measurement POVM data.
"""
meas_data(scheme::Scheme) = scheme.Measurement.data

"""
    strat_data(scheme::Scheme)

Return the estimation strategy.
"""
strat_data(scheme::Scheme) = scheme.EstimationStrategy

"""
    set_ctrl!(scheme, ctrl)

Mutating update of control coefficients.
"""
set_ctrl!(scheme, ctrl) = set_ctrl!(scheme.Parameterization, ctrl)

"""
    set_ctrl(scheme, ctrl)

Non-mutating copy with updated control coefficients.
"""
set_ctrl(scheme, ctrl) = set_ctrl(scheme.Parameterization, ctrl)

"""
    set_state!(scheme, state)

Mutating update of probe state data.
"""
set_state!(scheme, state) = @set scheme.StatePreparation.data = state

"""
    get_dim(scheme)

Return the Hilbert space dimension of the scheme.
"""
get_dim(scheme) = get_dim(scheme.Parameterization)