using Test
using NVMagnetometer, QuanEstimationBase, LinearAlgebra

@testset "NVMagnetometer.jl" begin

    scheme = NVMagnetometerScheme()
    @test isposdef(QFIM(scheme))
    @test isposdef(CFIM(scheme))
    @test HCRB(scheme) > 0
end # NVMagnetometer.jl tests
