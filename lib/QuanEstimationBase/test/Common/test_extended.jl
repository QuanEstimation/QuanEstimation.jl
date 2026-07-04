using Test
using LinearAlgebra

using QuanEstimationBase:
    BellState,
    PlusState,
    MinusState,
    SigmaX, SigmaY, SigmaZ,
    σx, σy, σz

@testset "BellState" begin
    @test BellState() == BellState(1)
    @test BellState(1) ≈ [1.0, 0.0, 0.0, 1.0] / sqrt(2)
    @test BellState(2) ≈ [1.0, 0.0, 0.0, -1.0] / sqrt(2)
    @test BellState(3) ≈ [0.0, 1.0, 1.0, 0.0] / sqrt(2)
    @test BellState(4) ≈ [0.0, 1.0, -1.0, 0.0] / sqrt(2)
    @test_throws DomainError BellState(0)
    @test_throws DomainError BellState(5)
end

@testset "PlusState / MinusState" begin
    psi_p = PlusState()
    psi_m = MinusState()
    @test psi_p ≈ [1.0, 1.0] / sqrt(2)
    @test psi_m ≈ [1.0, -1.0] / sqrt(2)
    # orthogonality
    @test psi_p' * psi_m ≈ 0.0 atol = 1e-10
    # eigenstates of σx
    @test SigmaX() * psi_p ≈ psi_p
    @test SigmaX() * psi_m ≈ -psi_m
end

@testset "Pauli matrices" begin
    sx = SigmaX()
    sy = SigmaY()
    sz = SigmaZ()
    @test sx == σx()
    @test sy == σy()
    @test sz == σz()
    @test sx' == sx
    @test sy' == sy
    @test sz' == sz
    @test isapprox(sx * sx, I(2); atol = 1e-10)
    @test isapprox(sy * sy, I(2); atol = 1e-10)
    @test isapprox(sz * sz, I(2); atol = 1e-10)
end
