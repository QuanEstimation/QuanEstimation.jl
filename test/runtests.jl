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

@testset "Resource" begin
    include("test_resource.jl")
end

@testset "NV" begin
    include("test_nv.jl")
end