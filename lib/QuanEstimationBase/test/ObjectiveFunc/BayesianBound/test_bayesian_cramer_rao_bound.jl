using Test
using LinearAlgebra
using QuadGK: quadgk

using QuanEstimationBase: 
    BCRB, 
    VTB, 
    BQCRB, 
    QVTB, 
    QZZB,
    Lindblad,
    GeneralScheme,
    SigmaX, SigmaY, SigmaZ


function test_bayesian_cramer_rao_bounds_singleparameter()
    scheme = generate_scheme_bayes_singleparameter()

    # Classical Bayesian bounds
    f_bcrb1 = BCRB(scheme; btype = 1)
    expected_bcrb1 = 0.654654507602925
    @test f_bcrb1 ≈ expected_bcrb1 atol=1e-3

    f_bcrb2 = BCRB(scheme; btype = 2)
    expected_bcrb2 = 0.651778484577857
    @test f_bcrb2 ≈ expected_bcrb2 atol=1e-3

    f_bcrb3 = BCRB(scheme; btype = 3)
    expected_bcrb3 = 0.16522254719803486
    @test f_bcrb3 ≈ expected_bcrb3 atol=1e-3

    f_vtb = VTB(scheme)
    expected_vtb = 0.03768712089828974
    @test f_vtb ≈ expected_vtb atol=1e-3

    # Quantum Bayesian bounds
    f_bqcrb1 = BQCRB(scheme; btype = 1)
    expected_bqcrb1 = 0.5097987285760552
    @test f_bqcrb1 ≈ expected_bqcrb1 atol=1e-3

    f_bqcrb2 = BQCRB(scheme; btype = 2)
    expected_bqcrb2 = 0.5094351484343563
    @test f_bqcrb2 ≈ expected_bqcrb2 atol=1e-3

    f_bqcrb3 = BQCRB(scheme; btype = 3)
    expected_bqcrb3 = 0.14347116223111836
    @test f_bqcrb3 ≈ expected_bqcrb3 atol=1e-3

    f_qvtb = QVTB(scheme)
    expected_qvtb = 0.037087918374800306
    @test f_qvtb ≈ expected_qvtb atol=1e-3

    f_qzzb = QZZB(scheme)
    expected_qzzb = 0.028521709437588784 
    @test f_qzzb ≈ expected_qzzb atol=1e-3

end

# Plan-4.18: extracted from deprecated test_bayesian_cramer_rao.jl (Bug #48)
@testset "#48 BQFIM with plain Vector prior" begin
    x = range(-0.5 * pi, stop = 0.5 * pi, length = 10) |> Vector
    p = fill(0.1, length(x))
    sx = SigmaX()
    sz = SigmaZ()
    H0(xi) = 0.5 * pi * (sx * cos(xi) + sz * sin(xi))
    dH(xi) = [0.5 * pi * (-sx * sin(xi) + sz * cos(xi))]
    rho_ev = [
        QuanEstimationBase.expm(range(0.0, 1.0, length = 10),
            0.5 * ones(2, 2), H0(xi), dH(xi))
        for xi in x
    ]
    rho = [r[1][end] for r in rho_ev]
    drho = [r[2][end] for r in rho_ev]
    result = QuanEstimationBase.BQFIM([x], p, rho, drho)
    @test result >= 0
end

test_bayesian_cramer_rao_bounds_singleparameter()
