module QuanEstimationBase
export ControlOpt, StateOpt, MeasurementOpt, ControlMeasurementOpt, CMopt, StateMeasurementOpt, SMopt, StateControlMeasurementOpt, SCMopt, opt_target, Htot
export evolve
export QFIM, CFIM, HCRB
export QFIM_obj, CFIM_obj, HCRB_obj
export AbstractScheme
export basis
export isCtrl, isNoisy
export autoGRAPE, GRAPE, PSO, DE
export GeneralScheme,Lindblad, Hamiltonian
export GeneralEstimation, GeneralMeasurement, GeneralState, GeneralParameterization
export AdaptiveStrategy, adapt!, Adapt_MZI
export Output
export Scheme
export DensityMatrix, Decay, Control, Expm, Ode, Strategy, POVM
export error_evaluation, error_control, error_control_param, error_control_eps
export optimize!
export state_data, param_data, meas_data, strat_data
export expm, ode
export Bayes
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
include("run.jl")
include("io.jl")
include("Resource/Resource.jl")

end
