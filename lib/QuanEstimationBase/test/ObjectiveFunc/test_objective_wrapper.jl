using Test
using LinearAlgebra: I
using QuanEstimationBase:
    QFIM_obj, CFIM_obj, HCRB_obj,
    SIC, Objective, Lindblad, GeneralScheme

function _build_scheme(; M=nothing)
    sz = [1.0 0.0; 0.0 -1.0]
    H0 = 0.5 * sz
    dH = [0.5 * sz]
    sp = [0.0 1.0; 0.0 0.0im]
    sm = [0.0 0.0; 1.0 0.0im]
    dynamics = Lindblad(H0, dH, 0:0.1:10, [[sp, 0.0], [sm, 0.1]])
    rho0 = 0.5 * ones(ComplexF64, 2, 2)
    return GeneralScheme(; probe=rho0, param=dynamics, measurement=M)
end

@testset "Objective with QFIM_obj" begin
    scheme = _build_scheme()
    obj = QFIM_obj(W=nothing, eps=0.01)
    result = Objective(scheme, obj)
    @test result.eps == 0.01
    @test result.W == I
end

@testset "Objective with CFIM_obj" begin
    M = SIC(2)
    scheme = _build_scheme(; M=M)
    obj = CFIM_obj(W=nothing, M=nothing, eps=0.01)
    result = Objective(scheme, obj)
    @test result.eps == 0.01
    @test result.W == I
    @test result.M == M
end

@testset "Objective with HCRB_obj" begin
    scheme = _build_scheme()
    obj = HCRB_obj(W=nothing, eps=0.01)
    result = Objective(scheme, obj)
    @test result.eps == 0.01
    @test result.W == I
end
