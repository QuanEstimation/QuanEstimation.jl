using Test
using LinearAlgebra
using Suppressor: @suppress
using Random

using QuanEstimationBase:
    Lindblad,
    GeneralScheme,
    optimize!,
    SCopt,
    CMopt,
    SMopt,
    SCMopt,
    AD,
    DE,
    PSO,
    SigmaX, SigmaY, SigmaZ,
    QFIM_obj,
    CFIM_obj

    
if !@isdefined generate_qubit_dynamics
    include("../utils.jl")
end

function test_scopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = QFIM_obj()
    opt = SCopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = AD(max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat")
    isfile("states.csv") && rm("states.csv")

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat")
    isfile("states.csv") && rm("states.csv")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.csv") && rm("states.csv")
end

function test_scopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = QFIM_obj()
    opt = SCopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = AD(max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat")
    isfile("states.csv") && rm("states.csv")

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.csv") && rm("states.csv")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.csv") && rm("states.csv")
end

function test_cmopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = CMopt(ctrl_bound = [-2.0, 2.0], seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
end

function test_cmopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = CMopt(ctrl_bound = [-2.0, 2.0], seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.csv") && rm("measurements.csv")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
end

function test_smopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = SMopt(seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
end

function test_smopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = SMopt(seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.csv") && rm("measurements.csv")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
end

function test_scmopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = SCMopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat")
    isfile("states.csv") && rm("states.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat")
    isfile("states.csv") && rm("states.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
end

function test_scmopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    obj = CFIM_obj()
    opt = SCMopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = DE(p_num=3, max_episode=10)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat")
    isfile("states.csv") && rm("states.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")

    alg = PSO(p_num=3, max_episode=[10, 10])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat")
    isfile("states.csv") && rm("states.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
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