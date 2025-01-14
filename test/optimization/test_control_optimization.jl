function test_copt_autogrape_qfi(; savefile = false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl = ctrl, dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = QFIM_obj()
    f0 = QFIM(scheme)[1]

    opt = ControlOpt(ctrl = ctrl, ctrl_bound = ctrl_bound, seed = 1234)
    alg =
        autoGRAPE(Adam = true, max_episode = 10, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    f1 = QFIM(scheme)[1]
    @test f1 >= f0
    rm("f.csv")
    rm("controls.dat")
end

function test_copt_autogrape_qfim(; savefile = false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl = ctrl, dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = QFIM_obj()
    f0 = tr(pinv(QFIM(scheme)))

    opt = ControlOpt(ctrl = ctrl, ctrl_bound = ctrl_bound, seed = 1234)
    alg =
        autoGRAPE(Adam = true, max_episode = 10, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    f1 = tr(pinv(QFIM(scheme)))
    @test f1 <= f0

    rm("f.csv")
    rm("controls.dat")
end

function test_copt_autogrape_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound, M) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl = ctrl, dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics, measurement = M)

    obj = CFIM_obj(M = M)
    f0 = CFIM(scheme)[1]

    opt = ControlOpt(ctrl = ctrl, ctrl_bound = ctrl_bound, seed = 1234)
    alg =
        autoGRAPE(Adam = true, max_episode = 10, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    f1 = CFIM(scheme)[1]
    @test f1 >= f0

    rm("f.csv")
    rm("controls.dat")
end

function test_copt_autogrape_cfim(; savefile = false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound, M) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl = ctrl, dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics, measurement = M)

    obj = CFIM_obj(M = M)
    f0 = tr(pinv(CFIM(scheme)))

    opt = ControlOpt(ctrl = ctrl, ctrl_bound = ctrl_bound, seed = 1234)
    alg =
        autoGRAPE(Adam = true, max_episode = 10, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    f1 = tr(pinv(CFIM(scheme)))
    @test f1 <= f0

    rm("f.csv")
    rm("controls.dat")
end


function test_copt()
    @testset "Control Optimization autoGRAPE QFIM" begin
        test_copt_autogrape_qfi()
        test_copt_autogrape_qfi(savefile = true)
        test_copt_autogrape_qfim()
        test_copt_autogrape_qfim(savefile = true)
    end

    @testset "Control Optimization autoGRAPE CFIM" begin
        test_copt_autogrape_cfi()
        test_copt_autogrape_cfi(savefile = true)
        test_copt_autogrape_cfim()
        test_copt_autogrape_cfim(savefile = true)
    end
end

test_copt()
