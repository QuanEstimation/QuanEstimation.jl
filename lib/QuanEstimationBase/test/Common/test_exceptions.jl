using Test
using QuanEstimationBase

@testset "Exception handling" begin
    @testset "BellState DomainError" begin
        @test_throws DomainError BellState(0)
        @test_throws DomainError BellState(5)
        @test_throws DomainError BellState(-1)
    end

    @testset "BellState valid inputs" begin
        for n in 1:4
            result = BellState(n)
            @test result isa Vector{ComplexF64}
            @test length(result) == 4
        end
    end

    @testset "BayesInput invalid channel" begin
        x = [[0.0]]
        func(x) = [1.0 0.0; 0.0 0.0]
        dfunc(x) = [[0.0 1.0; 1.0 0.0]]
        @test_throws ArgumentError QuanEstimationBase.BayesInput(x, func, dfunc; channel="invalid")
    end
end
