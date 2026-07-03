using Test
using LinearAlgebra
using QuanEstimationBase

@testset "Sys-E: Williamson decomposition" begin
    c = QuanEstimationBase.Williamson_form(Matrix{Float64}(I, 2, 2) * 3.0)[2]
    @test length(c) == 1
    @test isapprox(c[1], 3.0, rtol=1e-10)

    c5 = QuanEstimationBase.Williamson_form(Matrix{Float64}(I, 2, 2) * 5.0)[2]
    @test length(c5) == 1
    @test isapprox(c5[1], 5.0, rtol=1e-10)
end
