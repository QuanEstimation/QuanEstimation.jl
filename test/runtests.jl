using QuanEstimation, Test, LinearAlgebra, Random, Trapz, SparseArrays
using Suppressor: @suppress

include("utils.jl")

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

@testset "Bayesian Estimation" begin
    @testset "Bayesian Estimation" begin
        include("test_bayesian_estimation.jl")
    end
    @testset "Adaptive Estimation" begin
        include("test_adaptive_estimation.jl")
    end
end

@testset "Objective Function" begin
    @testset "Cramer-Rao Bounds" begin
        include("objective/test_cramer_rao_bound.jl")
    end

    @testset "Bayesian Cramer-Rao Bounds" begin
        include("objective/test_bayesian_cramer_rao_bound.jl")
    end
end

@testset "Scheme" begin
    include("test_scheme.jl")
end

@testset "Resource" begin
    include("test_resource.jl")
end

@testset "NV" begin
    include("test_nv.jl")
end

@testset "Error evaluation and control" begin
    include("test_error.jl")
end