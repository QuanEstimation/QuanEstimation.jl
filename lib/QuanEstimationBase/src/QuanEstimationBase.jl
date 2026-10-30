@doc raw"""
    QuanEstimationBase

Core computational engine for quantum parameter estimation. Provides:

- **Dynamics**: Lindblad, Hamiltonian, and Kraus evolution.
- **Objective functions**: QFIM, CFIM, HCRB, NHB, and various Bayesian/von Neumann bounds.
- **Optimization**: GRAPE, auto-GRAPE, AD, PSO, DE, DDPG, Nelder-Mead, and reverse-iterative algorithms for control, state, and measurement optimization.
- **Adaptive estimation**: FOP, MI, and MZI-based adaptive schemes with online/offline modes.
- **Utilities**: SU(N) generators, SIC-POVM, spin squeezing, error control, and evaluation tools.
"""
module QuanEstimationBase

export GeneralScheme,
    GeneralEstimation,
    GeneralMeasurement,
    GeneralState,
    AbstractScheme
export optimize!, init_opt
export ControlOpt,
    StateOpt,
    MeasurementOpt,
    StateMeasurementOpt,
    ControlMeasurementOpt,
    StateControlOpt,
    StateControlMeasurementOpt
export Copt, Sopt, Mopt, SMopt, CMopt, SCopt, SCMopt
export Mopt_Projection, Mopt_LinearComb, Mopt_Rotation
export evolve
export Lindblad, Hamiltonian, LindbladDynamics, Kraus, QubitDephasing
export QFIM, CFIM, HCRB, NHB, SLD, SLD_liouville, SLD_qr, FIM, FI_Expt, QFIM_Gauss, QFIM_Bloch, QFIM_Kraus, RLD, LLD
export QFIM_obj, CFIM_obj, HCRB_obj
export VTB, QVTB, QZZB, BCRB, BQCRB
export BCFIM, BQFIM, OBB, BCB, BayesCost, BayesInput
export basis, SIC, SpinSqueezing, TargetTime
export autoGRAPE, GRAPE, PSO, DE, AD, NM, RI
export Scheme, DensityMatrix, Decay, Control, Expm, Ode, Strategy, POVM
export error_evaluation, error_control, error_control_param, error_control_eps
export PrecisionSpec,
    NumericalDiagnostics,
    precision_spec,
    default_guard,
    spectral_floor,
    probability_floor,
    stop_threshold,
    guarded_pinv,
    guarded_objective,
    lyapunov_residual
export state_data, param_data, meas_data, strat_data
export expm, ode
export expm_py, ode_py, liouville_commu_py, liouville_dissip_py
export Htot
export SigmaX, SigmaY, SigmaZ, σx, σy, σz
export PlusState, MinusState, BellState
export ZeroCTRL, LinearCTRL, SineCTRL, SawCTRL, TriangleCTRL, GaussianCTRL, GaussianEdgeCTRL
export AdaptiveStrategy, adapt!, Adapt_MZI, online, offline, Bayes, MLE
export DE_deltaphiOpt, PSO_deltaphiOpt

using Random
using LinearAlgebra
using Zygote
using SparseArrays
using DelimitedFiles
using StatsBase
using JLD2
# using ReinforcementLearning
using SCS
using Trapz
using Interpolations
using Distributions
using QuadGK
using OrdinaryDiffEq
const pkgpath = @__DIR__

const GLOBAL_RNG = MersenneTwister(1234)
const GLOBAL_EPS = 1e-8

# PyExt function stubs — methods defined in ext/QuanEstimationBasePyExt.jl
@doc raw"""
    expm_py(tspan, ρ0, H0, dH, decay_opt, γ, Hc, ctrl)

Python-facing matrix-exponential wrapper provided by `QuanEstimationBasePyExt`
when `PythonCall` is loaded. It converts the Python arguments and forwards to
[`expm`](@ref). `dH` is a list of derivative Hamiltonians, `decay_opt` and `γ`
are the decay operators and rates, and `Hc`/`ctrl` are the control Hamiltonians
and coefficient sequences. Returns `(ρt, ∂ρt)` with the shapes produced by
`expm`. Calling it without the extension loaded raises a `MethodError`.
"""
function expm_py end

@doc raw"""
    ode_py(tspan, ρ0, H0, dH, Γ, γ, Hc, ctrl)

Python-facing ODE evolution wrapper provided by `QuanEstimationBasePyExt` when
`PythonCall` is loaded. It converts the Python arguments and calls the Julia
Lindblad ODE integration with `Tsit5`, returning `(ρt, ∂ρt)`. `dH` may be a single
derivative matrix or a vector of derivative matrices; `Γ` and `γ` are the decay
operators and rates, and `Hc`/`ctrl` are the control Hamiltonians and
coefficient sequences. For one parameter `∂ρt` is a vector of matrices; for
multiple parameters it is a vector (one entry per time step) of vectors (one
entry per parameter). Calling it without the extension loaded raises a
`MethodError`.
"""
function ode_py end

@doc raw"""
    liouville_commu_py(A)

Python-facing Liouville commutator provided by `QuanEstimationBasePyExt` when
`PythonCall` is loaded. `A` is a square complex matrix; the result is the
`dim^2 × dim^2` matrix representing `I ⊗ A - A^T ⊗ I`, with `dim = size(A, 1)`.
Calling it without the extension loaded raises a `MethodError`.
"""
function liouville_commu_py end

@doc raw"""
    liouville_dissip_py(A)

Python-facing Liouville dissipator provided by `QuanEstimationBasePyExt` when
`PythonCall` is loaded. `A` is a square complex decay operator; the result is
the `dim^2 × dim^2` dissipator matrix for `A`, with entries below `1e-10` in
absolute value set to zero. Calling it without the extension loaded raises a
`MethodError`.
"""
function liouville_dissip_py end

include("OptScenario/OptScenario.jl")
include("Common/Common.jl")
include("Scheme/Scheme.jl")
include("ObjectiveFunc/ObjectiveFunc.jl")
include("output.jl")
include("Algorithm/Algorithm.jl")
include("io.jl")
include("Resource/Resource.jl")

end
