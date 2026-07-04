using Test
using LinearAlgebra
using QuanEstimationBase

@testset "Sys-B: Single-qubit spontaneous emission (mixed state)" begin
    omega = 1.0
    gamma_minus = 0.1

    for t in [0.01, 0.1, 0.5, 1.0]
        rho, drho, F_exact = analytic_sysB(t, omega, gamma_minus)

        F_sld = QFIM(rho, drho; LDtype=:SLD)
        @test isapprox(F_sld[1], F_exact, rtol=1e-8)

        rho_h = (rho + rho') / 2
        F_h = QFIM(rho_h, drho; LDtype=:SLD)
        @test isapprox(F_h[1], F_sld[1], rtol=1e-12)

        R = RLD(rho, drho[1]; rep="original")
        @test norm(rho * R - drho[1]) < 1e-8

        L_lld = LLD(rho, drho[1]; rep="original")
        @test norm(L_lld * rho - drho[1]) < 1e-8

        F_rld = QFIM(rho, drho; LDtype=:RLD)
        F_lld = QFIM(rho, drho; LDtype=:LLD)
        @test isapprox(F_rld[1], F_lld[1], rtol=1e-10)
    end
end
