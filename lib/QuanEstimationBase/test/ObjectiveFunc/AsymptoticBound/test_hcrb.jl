using QuanEstimationBase
using LinearAlgebra
using Test

function test_hcrb()
    psi0 = [1.0, 0.0, 0.0, 1.0] / sqrt(2)
    rho0 = psi0 * psi0'
    omega1, omega2, g = 1.0, 1.0, 0.1
    sx = [0.0 1.0; 1.0 0.0im]
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = omega1 * kron(sz, I(2)) + omega2 * kron(I(2), sz) + g * kron(sx, sx)
    dH = [kron(I(2), sz), kron(sx, sx)]
    decay = [[kron(sz, I(2)), 0.05], [kron(I(2), sz), 0.05]]
    W = one(zeros(2, 2))
    tspan = range(0.0, 5.0, length = 20)
    rho, drho = QuanEstimationBase.expm(tspan, rho0, H0, dH; decay=decay)

    f_HCRB, f_QFIM = Float64[], Float64[]
    for ti = 2:length(tspan)
        push!(f_HCRB, QuanEstimationBase.HCRB(rho[ti], drho[ti], W))
        push!(f_QFIM, tr(inv(QuanEstimationBase.QFIM(rho[ti], drho[ti]))))
    end

    @testset "HCRB" begin
        @test all(>=(0), f_HCRB)
        @test all(isfinite.(f_HCRB))
        @test all(f_HCRB .>= f_QFIM .- 1e-10)
    end
end

test_hcrb()
