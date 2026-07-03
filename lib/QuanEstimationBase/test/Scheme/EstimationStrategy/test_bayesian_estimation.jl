using Test
using LinearAlgebra
using Random
using Trapz: trapz

using QuanEstimationBase:
    BCB,
    SIC,
    SigmaX, SigmaY, SigmaZ,
    expm,
    Bayes,
    MLE


function test_bayes()
    (; rho0, x, p, dp, H0_func, dH_func) = generate_bayes()
    M = SIC(2)
    tspan = range(0.0, stop = 1.0, length = 1000)
    rho = Vector{Matrix{ComplexF64}}(undef, length(x))
    for i in eachindex(x)
        H0_tp = H0_func(x[i])
        dH_tp = dH_func(x[i])
        rho_tp, _ = expm(tspan, rho0, H0_tp, dH_tp)
        rho[i] = rho_tp[end]
    end

    # Generation of the experimental results
    Random.seed!(1234)
    y = [rand() > 0.7 ? 1 : 0 for _ = 1:500]

    #===============Maximum a posteriori estimation===============#
    pout, xout = Bayes([x], p, rho, y; M = M, savefile = false)
    @test all(pout .>= 0)
    pout, xout = Bayes([x], p, rho, y; M = M, savefile = true)
    pout, xout = Bayes([x], p, rho, y; M = M, estimator = "MAP", savefile = false)
    pout, xout = Bayes([x], p, rho, y; M = M, estimator = "MAP", savefile = true)

    #===============Maximum likelihood estimation===============#
    Lout, xout = MLE([x], rho, y, M = M; savefile = false)
    @test length(Lout) > 0
    Lout, xout = MLE([x], rho, y, M = M; savefile = true)

    BCB([x], p, rho)
    isfile("bayes.dat") && rm("bayes.dat")
    isfile("bayes.csv") && rm("bayes.csv")
    isfile("MLE.dat") && rm("MLE.dat")
    isfile("MLE.csv") && rm("MLE.csv")    

    # Bug #56: Near-zero eigenvalue truncation in BayesEstimation
    @testset "#56 Near-zero eigenvalue truncation" begin
        rho_sing = ComplexF64[1.0 0.0; 0.0 1e-16]
        rho_sing = (rho_sing + rho_sing') / 2
        rho_sing ./= tr(rho_sing)
        M_sic = SIC(2)
        y_test = [rand() >= 0.5 ? 0 : 1 for _ = 1:10]
        pout_test, xout_test = Bayes([x], p, [rho_sing for _ in x], y_test; M = M_sic, estimator = "MAP", savefile = false)
        @test all(isfinite, pout_test)
        @test !any(isnan, pout_test)
    end
end

@testset "Bayesian estimation" begin test_bayes() end