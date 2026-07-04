using Zygote
using ChainRulesCore
using LinearAlgebra, Test

function test_comprehension_qfim()
    ρ = rand_ρ(3)
    ∂ρ = [rand_∂ρ(3), rand_∂ρ(3)]

    F_SLD = QuanEstimationBase.QFIM_SLD(ρ, ∂ρ; eps = 1e-8)
    F_RLD = QuanEstimationBase.QFIM_RLD(ρ, ∂ρ; eps = 1e-8)
    F_LLD = QuanEstimationBase.QFIM_LLD(ρ, ∂ρ; eps = 1e-8)

    @test size(F_SLD) == (2, 2)
    @test size(F_RLD) == (2, 2)
    @test size(F_LLD) == (2, 2)
    @test all(isfinite.(F_SLD))
    @test all(isfinite.(F_RLD))
    @test all(isfinite.(F_LLD))
end

function test_rrule_sld_basic()
    ρ = rand_ρ(2)
    A = randn(ComplexF64, 2, 2)
    ∂ρ = ComplexF64[0.5 0.1+0.0im; 0.1+0.0im -0.5]
    L = QuanEstimationBase.SLD(ρ, ∂ρ; eps = 1e-8)
    @test ishermitian(L) || norm(L - L') < 1e-10
    grad = Zygote.gradient(ρ -> real(tr(QuanEstimationBase.SLD(ρ, ∂ρ; eps = 1e-8))), ρ)
    @test !isnothing(grad)
    @test all(isfinite.(grad[1]))
end

function test_rrule_sld_pullback()
    ρ = rand_ρ(2)
    ∂ρ = ComplexF64[0.5 0.1+0.0im; 0.1+0.0im -0.5]
    L, pb = ChainRulesCore.rrule(QuanEstimationBase.SLD, ρ, ∂ρ; eps = 1e-8)
    @test ishermitian(L) || norm(L - L') < 1e-10
    L̄ = rand_ρ(2)
    fresult = pb(L̄)
    @test fresult isa NTuple{3, Any}
end

@testset "test_ad_compat" begin
    @testset "comprehension QFIM" begin
        test_comprehension_qfim()
    end
    @testset "rrule SLD basic" begin
        test_rrule_sld_basic()
    end
    @testset "rrule SLD pullback" begin
        test_rrule_sld_pullback()
    end
end
