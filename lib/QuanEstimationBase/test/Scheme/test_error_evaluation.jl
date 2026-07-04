using LinearAlgebra, Test

function test_QFI_with_error_scheme()
    (; tspan, rho0, H0, dH) = generate_qubit_dynamics_base()
    dyn = Lindblad(H0, dH, tspan; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dyn)

    F, deltaF = QuanEstimationBase.QFIM_with_error(scheme; eps = 1e-8)
    @test isfinite(F[1, 1])
end

function test_SLD_with_error_hermitian()
    ρ = rand_ρ(3)
    ∂ρ = rand_∂ρ(3)

    SLD_tp, SLD_err = QuanEstimationBase.SLD_with_error(ρ, ∂ρ; eps = 1e-8)
    @test norm(SLD_tp - SLD_tp') < 1e-10
    @test norm(SLD_err - SLD_err') < 1e-10
end

function test_RLD_LLD_with_error_formula()
    ρ = rand_ρ(3)
    ∂ρ = rand_∂ρ(3)

    F_RLD = QuanEstimationBase.QFIM_RLD_with_error(ρ, ∂ρ; eps = 1e-8)
    F_LLD = QuanEstimationBase.QFIM_LLD_with_error(ρ, ∂ρ; eps = 1e-8)
    @test isfinite(F_RLD)
    @test isfinite(F_LLD)
    @test F_RLD ≈ F_LLD
end

function test_SLD_with_error_anti_hermitian_perturbation()
    ρ = rand_ρ(3)
    ∂ρ = rand_∂ρ(3)
    # Add small anti-Hermitian perturbation to ρ
    rho_pert = ρ + 1e-6 * 1.0im * (rand_∂ρ(3))
    rho_pert = (rho_pert + rho_pert') / 2
    rho_pert ./= tr(rho_pert)

    SLD_tp, _ = QuanEstimationBase.SLD_with_error(rho_pert, ∂ρ; eps = 1e-8)
    @test norm(SLD_tp - SLD_tp') < 1e-10
end

function test_multi_QFIM_with_error()
    ρ = rand_ρ(3)
    ∂ρ = [rand_∂ρ(3), rand_∂ρ(3)]

    F_SLD = QuanEstimationBase.QFIM_SLD_with_error(ρ, ∂ρ; eps = 1e-8)
    F_RLD = QuanEstimationBase.QFIM_RLD_with_error(ρ, ∂ρ; eps = 1e-8)
    F_LLD = QuanEstimationBase.QFIM_LLD_with_error(ρ, ∂ρ; eps = 1e-8)

    @test size(F_SLD) == (2, 2)
    @test size(F_RLD) == (2, 2)
    @test size(F_LLD) == (2, 2)
    @test all(isfinite.(F_SLD))
    @test all(isfinite.(F_RLD))
    @test all(isfinite.(F_LLD))
end

@testset "test_error_evaluation" begin
    @testset "QFI_with_error scheme" begin
        test_QFI_with_error_scheme()
    end
    @testset "SLD_with_error hermiticity" begin
        test_SLD_with_error_hermitian()
    end
    @testset "RLD/LLD with error formula" begin
        test_RLD_LLD_with_error_formula()
    end
    @testset "SLD_with_error anti-hermitian perturbation" begin
        test_SLD_with_error_anti_hermitian_perturbation()
    end
    @testset "Multi-param QFIM with error" begin
        test_multi_QFIM_with_error()
    end
end
