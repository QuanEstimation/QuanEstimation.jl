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

if !@isdefined generate_scheme_bayes_singleparameter
    include("../utils.jl")
end

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

end

test_bayesian_cramer_rao_bounds_singleparameter()
