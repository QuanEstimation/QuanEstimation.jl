function test_sopt_autogrape_qfi(;savefile=false)
    (; tspan, psi, H0, dH, decay,) = generate_LMG1_dynamics()

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
    scheme = GeneralScheme(; probe=psi, param=dynamics,)

    obj = QFIM_obj()
    f0 = QFIM(scheme)[1]

    opt = StateOpt(psi=psi, seed=1234) 
    alg = AD(Adam=true, max_episode=10, epsilon=0.01, beta1=0.90, beta2=0.99)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)

    f1 = QFIM(scheme)[1]
    @test f1 >= f0
    rm("f.csv")
    rm("states.dat")
end

function test_sopt_autogrape_qfim(;savefile=false)
    (; tspan, psi, H0, dH, decay,W) = generate_LMG2_dynamics()

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
    scheme = GeneralScheme(; probe=psi, param=dynamics,)

    obj = QFIM_obj(W=W)
    f0 = tr(pinv(QFIM(scheme)))

    opt = StateOpt(psi=psi, seed=1234) 
    alg = AD(Adam=true, max_episode=10, epsilon=0.01, beta1=0.90, beta2=0.99)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)

    f1 = tr(pinv(QFIM(scheme)))
    @test f1 <= f0
    rm("f.csv")
    rm("states.dat")
end

function test_sopt_autogrape_cfi(;savefile=false)
    (; tspan, psi, H0, dH, decay,) = generate_LMG1_dynamics()

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
    scheme = GeneralScheme(; probe=psi, param=dynamics,)

    obj = CFIM_obj()
    f0 = CFIM(scheme)[1]

    opt = StateOpt(psi=psi, seed=1234) 
    alg = AD(Adam=true, max_episode=10, epsilon=0.01, beta1=0.90, beta2=0.99)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)

    f1 = CFIM(scheme)[1]
    @test f1 >= f0

    rm("f.csv")
    rm("states.dat")
end

function test_sopt_autogrape_cfim(;savefile=false)
    (; tspan, psi, H0, dH, decay,W) = generate_LMG2_dynamics()

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
    scheme = GeneralScheme(; probe=psi, param=dynamics,)

    obj = CFIM_obj(W=W)
    f0 = tr(pinv(CFIM(scheme)))

    opt = StateOpt(psi=psi, seed=1234) 
    alg = AD(Adam=true, max_episode=10, epsilon=0.01, beta1=0.90, beta2=0.99)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)

    f1 = tr(pinv(CFIM(scheme)))
    @test f1 <= f0
    rm("f.csv")
    rm("states.dat")
end


function test_sopt()
    @testset "State Optimization AD QFIM" begin
        test_sopt_autogrape_qfi()
        test_sopt_autogrape_qfi(savefile=true)
        test_sopt_autogrape_qfim()
        test_sopt_autogrape_qfim(savefile=true)
    end

    @testset "State Optimization AD CFIM" begin
        test_sopt_autogrape_cfi()
        test_sopt_autogrape_cfi(savefile=true)
        test_sopt_autogrape_cfim()
        test_sopt_autogrape_cfim(savefile=true)
    end
end

test_sopt()