using Test
using LinearAlgebra
using QuanEstimationBase

@testset "Sys-A: Single-qubit rotation (pure state, noiseless)" begin
    omega = 1.5
    for t in [0.1, 0.5, 1.0, 2.0]
        rho, drho, F_exact = analytic_sysA(t, omega)

        F_sld = QFIM(rho, drho; LDtype=:SLD)
        @test F_sld[1] ≈ F_exact rtol=1e-10

        F_pure = QuanEstimationBase.QFIM_pure(rho, drho)
        @test F_pure[1] ≈ F_exact rtol=1e-10

        L = SLD(rho, drho[1])
        @test norm(L - L') < 1e-14
        @test norm(2 * drho[1] - L * rho - rho * L) < 1e-10

        rho_h = (rho + rho') / 2
        @test norm(rho_h - rho) < 1e-14
    end
end
