using Test
using LinearAlgebra

using QuanEstimationBase:
    QubitDephasing,
    get_dim

@testset "QubitDephasing" begin
    tspan = range(0.0, 2.0, length = 20)

    @testset "construct with default axis" begin
        r = [1.0, 0.0, 0.0]
        result = QubitDephasing(r, "z", 0.1, tspan)
        @test get_dim(result) > 0
    end

    @testset "all three axes" begin
        r = [1.0, 1.0, 1.0]
        for axis in ["x", "y", "z"]
            result = QubitDephasing(r, axis, 0.01, tspan)
            @test get_dim(result) > 0
        end
    end

    @testset "invalid axis errors" begin
        r = [1.0, 0.0, 0.0]
        @test_throws UndefVarError QubitDephasing(r, "invalid", 0.1, tspan)
    end
end
