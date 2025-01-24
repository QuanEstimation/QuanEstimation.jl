function test_bayesian_cramer_rao_bounds()
    scheme = generate_scheme_bayes()

    # Classical Bayesian bounds
    f_BCRB1 = BCRB(scheme; btype = 1)
    f_BCRB2 = BCRB(scheme; btype = 2)
    f_BCRB3 = BCRB(scheme; btype = 3)
    f_VTB = VTB(scheme)

    # Quantum Bayesian bounds
    f_BQCRB1 = BQCRB(scheme; btype = 1)
    f_BQCRB2 = BQCRB(scheme; btype = 2)
    f_BQCRB3 = BQCRB(scheme; btype = 3)
    f_QVTB = QVTB(scheme)
    f_QZZB = QZZB(scheme)

    @test f_BCRB1 >= 0
    @test f_BCRB2 >= 0
    @test f_BCRB3 >= 0

    @test f_BQCRB1 >= 0
    @test f_BQCRB2 >= 0
    @test f_BQCRB3 >= 0

    @test f_VTB >= 0
    @test f_QVTB >= 0
    @test f_QZZB >= 0
end

test_bayesian_cramer_rao_bounds()
