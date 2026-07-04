using Test
using LinearAlgebra
using Suppressor: @suppress
using SparseArrays: sparse
using Trapz: trapz

using QuanEstimationBase: 
    Adapt_MZI,
    DE,
    PSO,
    adapt!,
    offline,
    online,
    basis,
    Lindblad,
    AdaptiveStrategy,
    GeneralScheme,
    DE_deltaphiOpt,
    PSO_deltaphiOpt,
    SigmaX, SigmaY, SigmaZ


function test_adaptive_estimation_MZI()
    N = 3
    # probe state
    psi =
        sum([
            sin(k * pi / (N + 2)) * kron(basis(N + 1, k), basis(N + 1, N - k + 2)) for
            k = 1:(N+1)
        ]) |> sparse
    psi = psi * sqrt(2 / (2 + N))
    rho0 = psi * psi'
    # prior distribution
    x = range(-pi, pi, length = 100)
    p = (1.0 / (x[end] - x[1])) * ones(length(x))
    apt = Adapt_MZI(x, p, rho0)

    #================online strategy=========================#
    res_phi = zeros(2)
    online(apt; target="sharpness", output="phi", res=res_phi)
    @test all(isfinite, res_phi)
    res_phi_mi = zeros(2)
    online(apt; target="MI", output="phi", res=res_phi_mi)
    @test all(isfinite, res_phi_mi)
    res_dphi = zeros(2)
    online(apt; target="sharpness", output="dphi", res=res_dphi)
    @test all(isfinite, res_dphi)
    res_dphi_mi = zeros(2)
    online(apt; target="MI", output="dphi", res=res_dphi_mi)
    @test all(isfinite, res_dphi_mi)

    #================offline strategy=========================#
    alg = DE(p_num = 3, ini_population = nothing, max_episode = 3, c = 1.0, cr = 0.5)
    out = offline(apt, alg, target = :sharpness, seed = 1234)
    @test all(isfinite, out)
    out_mi = offline(apt, alg, target = :MI, seed = 1234)
    @test all(isfinite, out_mi)

    alg = PSO(p_num=3, ini_particle=nothing, max_episode=[3,3], c0=1.0, c1=2.0, c2=2.0)
    out_ps = offline(apt, alg, target=:sharpness, seed=1234)
    @test all(isfinite, out_ps)

    isfile("f.csv") && rm("f.csv")
    isfile("deltaphi.csv") && rm("deltaphi.csv")
    isfile("adaptive.dat") && rm("adaptive.dat")
    isfile("adaptive.csv") && rm("adaptive.csv")
end

function test_adaptive_estimation()
    scheme = generate_scheme_adaptive()

    res_fop = zeros(10)
    @suppress adapt!(scheme; res=res_fop, method="FOP", max_episode=3)
    @test all(isfinite, res_fop)
    res_mi = zeros(10)
    @suppress adapt!(scheme; res=res_mi, method="MI", max_episode=3)
    @test all(isfinite, res_mi)

    isfile("adaptive.dat") && rm("adaptive.dat")
    isfile("adaptive.csv") && rm("adaptive.csv")
end


function test_deltaphi_opt()
    N = 3
    psi = sum([
        sin(k * pi / (N + 2)) * kron(basis(N + 1, k), basis(N + 1, N - k + 2)) for
        k = 1:(N+1)
    ]) |> sparse
    psi = psi * sqrt(2 / (2 + N))
    rho0 = psi * psi'
    x = range(-pi, pi, length = 100)
    p = (1.0 / (x[end] - x[1])) * ones(length(x))
    apt = Adapt_MZI(x, p, rho0)

    @testset "DE_deltaphiOpt" begin
        out = offline(apt, DE(p_num=3, max_episode=3), target=:sharpness, seed=1234)
        @test all(isfinite, out)
        @test DE_deltaphiOpt isa Function
    end
    @testset "PSO_deltaphiOpt" begin
        out = offline(apt, PSO(p_num=3, max_episode=[3, 3]), target=:sharpness, seed=1234)
        @test all(isfinite, out)
        @test PSO_deltaphiOpt isa Function
    end
    isfile("f.csv") && rm("f.csv")
    isfile("deltaphi.csv") && rm("deltaphi.csv")
end

@testset "Adaptive Estimation" begin
    test_adaptive_estimation_MZI()
    test_adaptive_estimation()
    test_deltaphi_opt()
end