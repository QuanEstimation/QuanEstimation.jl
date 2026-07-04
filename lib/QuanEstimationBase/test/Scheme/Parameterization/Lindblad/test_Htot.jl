using Test
using LinearAlgebra
using QuanEstimationBase: Htot

@testset "Htot" begin
    @testset "two-time-slice qubit with control" begin
        H0 = ComplexF64[1.0 0.0; 0.0 -1.0]
        Hc = [ComplexF64[0.0 1.0; 1.0 0.0]]
        ctrl = [[0.5, 1.0]]
        result = Htot(H0, Hc, ctrl)
        @test length(result) == 2
        @test size(result[1]) == (2, 2)
        @test size(result[2]) == (2, 2)
        @test result[1] ≈ H0 + 0.5 * Hc[1]
        @test result[2] ≈ H0 + 1.0 * Hc[1]
    end

    @testset "zero control" begin
        H0 = ComplexF64[1.0 0.0; 0.0 -1.0]
        Hc = ComplexF64[]
        ctrl = [[]]
        result = Htot(H0, Hc, ctrl)
        @test result == ComplexF64[]
    end
end
