using QuanEstimation, Test, LinearAlgebra, Random, Trapz, SparseArrays
using Suppressor: @suppress

include("utils.jl")

# Main test suite for QuanEstimation
@testset "QuanEstimation Tests" begin

    # Optimization tests
    @testset "Optimization" begin
        @testset "Control Optimization" begin
            include("optimization/test_control_optimization.jl")
        end

        @testset "State Optimization" begin
            include("optimization/test_state_optimization.jl")
        end

        @testset "Measurement Optimization" begin
            include("optimization/test_measurement_optimization.jl")
        end

        @testset "Comprehensive Optimization" begin
            include("optimization/test_comprehensive_optimization.jl")
        end
    end

    # Bayesian estimation tests
    @testset "Bayesian Estimation" begin
        @testset "Bayesian Estimation" begin
            include("test_bayesian_estimation.jl")
        end
        @testset "Adaptive Estimation" begin
            include("test_adaptive_estimation.jl")
        end
    end

    # Objective function tests
    @testset "Objective Functions" begin
        @testset "Cramer-Rao Bounds" begin
            include("objective/test_cramer_rao_bound.jl")
        end

        @testset "Bayesian Cramer-Rao Bounds" begin
            include("objective/test_bayesian_cramer_rao_bound.jl")
        end
    end

    # Core scheme tests
    @testset "Scheme" begin
        include("test_scheme.jl")
    end

    # Resource tests
    @testset "Resource" begin
        include("test_resource.jl")
    end

    # NV magnetometer tests
    @testset "NV" begin
        include("test_nv.jl")
    end

    # Error handling tests
    @testset "Error Evaluation and Control" begin
        include("test_error.jl")
    end

end # End of QuanEstimation Tests
