using Test
using LinearAlgebra
using QuanEstimationBase

@testset "Sys-D: Two-qubit XX coupling (pure state)" begin
    omega1, omega2, g = 1.0, 1.0, 0.1

    for t in [0.5, 1.0]
        rho, drho, F_exact = analytic_sysD(t, omega1, omega2, g)

        F_pure = QuanEstimationBase.QFIM_pure(rho, drho)
        @test isapprox(F_pure, real.(F_exact), rtol=1e-10)

        F_sld = QFIM(rho, drho; LDtype=:SLD)
        @test isapprox(F_sld, real.(F_exact), rtol=1e-10)

        Ls = [SLD(rho, d; rep="original") for d in drho]
        @test all(L -> norm(L - L') < 1e-14, Ls)

        rho_h = (rho + rho') / 2
        F_h = QFIM(rho_h, drho; LDtype=:SLD)
        @test isapprox(F_h, F_sld, rtol=1e-12)
    end
end
