using Test
using LinearAlgebra
using QuanEstimationBase

@testset "Sys-C: Single-qubit pure dephasing" begin
    omega = 1.0
    for gamma in [0.01, 0.1, 0.5]
        for t in [0.1, 0.5, 1.0]
            rho, drho, F_exact = analytic_sysC(t, omega, gamma)

            F_sld = QFIM(rho, drho[1]; LDtype=:SLD)
            @test isapprox(F_sld, F_exact, rtol=1e-10)

            F_rld = QFIM(rho, drho[1]; LDtype=:RLD)
            F_lld = QFIM(rho, drho[1]; LDtype=:LLD)
            @test isapprox(F_rld, F_lld, rtol=1e-10)
        end
    end

    gamma = 0.0
    for t in [0.1, 0.5, 1.0]
        rho, drho, F_exact = analytic_sysC(t, omega, gamma)
        F_sld = QFIM(rho, drho[1]; LDtype=:SLD)
        @test isapprox(F_sld, F_exact, rtol=1e-10)
    end
end
