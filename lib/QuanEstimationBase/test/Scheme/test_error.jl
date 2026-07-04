using Test, LinearAlgebra
using QuanEstimationBase:
    SigmaX, SigmaY, SigmaZ,
    Lindblad, GeneralScheme,
    error_evaluation, error_control


function test_error_evaluation()
    scheme = generate_qubit_scheme()
    @test_nowarn error_evaluation(scheme)
end

function test_error_control()
    scheme = generate_qubit_scheme()
    @test_nowarn error_control(scheme)
    @test_throws ArgumentError error_control(scheme, objective="HCRB")
end

function test_error()
    @testset "error_evaluation" begin test_error_evaluation() end
    @testset "error_control" begin test_error_control() end
end

test_error()
