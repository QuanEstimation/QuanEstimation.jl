using QuanEstimation
using Test

@testset "QuanEstimation.jl" begin

    @testset "QuanEstimationBase" begin
        include("../src/QuanEstimationBase/test/runtests.jl")
    end
    @testset "NVMagnetometer" begin
        include("../src/NVMagnetometer/test/runtests.jl")
    end
    
end # QuanEstimation.jl tests