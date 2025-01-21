module QuanEstimationBase

export GeneralScheme,
    GeneralEstimation,
    GeneralMeasurement,
    GeneralState,
    GeneralParameterization,
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
export evolve
export Lindblad, Hamiltonian, LindbladDynamics, Kraus, QubitDephasing
export QFIM,
    CFIM,
    HCRB,
    NHB,
    SLD,
    SLD_liouville,
    SLD_qr,
    FIM,
    FI_Expt,
    QFIM_Gauss,
    QFIM_Bloch,
    RLD,
    LLD
export QFIM_obj, CFIM_obj, HCRB_obj
export VTB, QVTB, QZZB, BCRB, BQCRB
export basis, SIC, SpinSqueezing
export autoGRAPE, GRAPE, PSO, DE, AD, NM, RI
export Scheme, DensityMatrix, Decay, Control, Expm, Ode, Strategy, POVM
export error_evaluation, error_control, error_control_param, error_control_eps
export state_data, param_data, meas_data, strat_data
export expm, ode
export SigmaX, SigmaY, SigmaZ, σx, σy, σz
export PlusState, MinusState, BellState
export ZeroCTRL, LinearCTRL, SineCTRL, SawCTRL, TriangleCTRL, GaussianCTRL, GaussianEdgeCTRL
export AdaptiveStrategy, adapt!, Adapt_MZI, online, offline, Bayes, MLE

using Random
using LinearAlgebra
using Zygote
using SparseArrays
using DelimitedFiles
using StatsBase
using Flux
# using ReinforcementLearning
using SCS
using Trapz
using Interpolations
using Distributions
using QuadGK
using OrdinaryDiffEq
using JLD2
const pkgpath = @__DIR__

const GLOBAL_RNG = MersenneTwister(1234)
const GLOBAL_EPS = 1e-8
include("OptScenario/OptScenario.jl")
include("Common/Common.jl")
include("Scheme/Scheme.jl")
include("ObjectiveFunc/ObjectiveFunc.jl")
include("output.jl")
include("Algorithm/Algorithm.jl")
include("io.jl")
include("Resource/Resource.jl")

end
