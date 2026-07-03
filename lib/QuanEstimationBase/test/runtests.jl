using QuanEstimationBase
using Test
using Trapz
using LinearAlgebra
using SparseArrays

include("utils.jl")

@testset verbose = true "Operator Definitions" begin
    include("Common/operator_definitions.jl")
end

@testset verbose = true "Analytic QFIM Benchmarks" begin
    include("ObjectiveFunc/AsymptoticBound/test_analytic_qfim.jl")
end

@testset verbose = true "Objective Functions" begin
    @testset "Fisher information" begin
        include("ObjectiveFunc/AsymptoticBound/test_fisher_information_matrix.jl")
    end

    @testset "Analog CramerRao" begin
        include("ObjectiveFunc/AsymptoticBound/test_hcrb.jl")
        include("ObjectiveFunc/AsymptoticBound/test_nhb.jl")
    end

    @testset "Bayesian Bounds" begin
        include("ObjectiveFunc/BayesianBound/test_ziv_zakai.jl")
    end

    @testset "Objective Wrapper" begin
        include("ObjectiveFunc/test_objective_wrapper.jl")
    end
end


@testset verbose = true "Dynamics" begin
    @testset "Lindblad Wrapper" begin
        include("Scheme/Parameterization/Lindblad/test_init_scheme.jl")
    end
    @testset "Qubit Dephasing" begin
        include("Parameterization/test_QubitDephasing.jl")
    end
    @testset "Htot" begin
        include("Scheme/Parameterization/Lindblad/test_Htot.jl")
    end
end


@testset "Adaptive" begin
    include("Scheme/EstimationStrategy/AdaptiveEstimation/test_adaptive_estimation.jl")
end

@testset "Schedule" begin
    include("Scheme/test_error_evaluation.jl")
end

@testset "Algorithm" begin
    @testset "AD" begin
        include("Algorithm/test_AD.jl")
    end
end

@testset "Optimization" begin
    @testset "AD Compat" begin
        include("OptScenario/test_ad_compat.jl")
    end

    @testset "Opt Scenario" begin
        include("OptScenario/test_opt_scenario.jl")
    end
end

@testset "Utils" begin
    @testset "SIC" begin
        include("Resource/test_sic.jl")
    end
end

# --- Root-sinked integration tests ---

@testset verbose = true "Scheme" begin
    @testset "Parameterization" begin
        include("Scheme/test_scheme.jl")
    end
    @testset "Error" begin
        include("Scheme/test_error.jl")
    end
end

@testset "Bayesian Estimation" begin
    include("Scheme/EstimationStrategy/test_bayesian_estimation.jl")
end

@testset "Adaptive Integration" begin
    include("Scheme/EstimationStrategy/AdaptiveEstimation/test_adaptive_integration.jl")
end

@testset "Resource" begin
    include("Resource/test_resource.jl")
end

@testset "Exceptions" begin
    include("Common/test_exceptions.jl")
end

@testset "Extended / Quantum States" begin
    include("Common/test_extended.jl")
end

@testset "Cramer-Rao Bounds (Integration)" begin
    include("ObjectiveFunc/AsymptoticBound/test_cramer_rao_bound.jl")
end

@testset "Bayesian Cramer-Rao Bounds (Integration)" begin
    include("ObjectiveFunc/BayesianBound/test_bayesian_cramer_rao_bound.jl")
end

@testset "Control Optimization (Integration)" begin
    include("OptScenario/test_control_optimization.jl")
end

@testset "State Optimization (Integration)" begin
    include("OptScenario/test_state_optimization.jl")
end

@testset "Measurement Optimization (Integration)" begin
    include("OptScenario/test_measurement_optimization.jl")
end

@testset "Comprehensive Optimization (Integration)" begin
    include("OptScenario/test_comprehensive_optimization.jl")
end

@testset verbose = true "Algorithm Execution" begin
    include("Algorithm/test_algorithm_exec.jl")
end
