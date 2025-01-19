using QuanEstimation, Test, LinearAlgebra, Random, SparseArrays
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
