function test_nv_magnetometer()
    scheme = NVMagnetometerScheme()
    @test isposdef(QFIM(scheme))
    @test isposdef(CFIM(scheme))
    @test isposdef(HCRB(scheme))
        
    alg = autoGRAPE(;max_episode = 10)
    @suppress optimize!(scheme, ControlOpt(); algorithm = alg)
    rm("f.csv")
    rm("controls.dat")

    error_evaluation(scheme)
    error_control(scheme)
end

test_nv_magnetometer()