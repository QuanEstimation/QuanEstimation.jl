function test_scopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = QFIM_obj()
    opt = SCopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = AD(max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")
end

function test_scopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = QFIM_obj()
    opt = SCopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = AD(max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")
end

function test_cmopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = CMopt(ctrl_bound = [-2.0, 2.0], seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("measurements.dat")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("measurements.dat")
end

function test_cmopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = CMopt(ctrl_bound = [-2.0, 2.0], seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("measurements.dat")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("measurements.dat")
end

function test_smopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = SMopt(seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("states.dat")
    rm("measurements.dat")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("states.dat")
    rm("measurements.dat")
end

function test_smopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = SMopt(seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("states.dat")
    rm("measurements.dat")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("states.dat")
    rm("measurements.dat")
end

function test_scmopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = SCMopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")
    rm("measurements.dat")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")
    rm("measurements.dat")
end

function test_scmopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = SCMopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")
    rm("measurements.dat")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    rm("f.csv")
    rm("controls.dat")
    rm("states.dat")
    rm("measurements.dat")
end

function test_comprehensive_optimization()
    test_scopt_qfi()
    test_scopt_qfim()
    test_cmopt_qfi()
    test_cmopt_qfim()
    test_smopt_qfi()
    test_smopt_qfim()
    test_scmopt_qfi()
    test_scmopt_qfim()
end

test_comprehensive_optimization()