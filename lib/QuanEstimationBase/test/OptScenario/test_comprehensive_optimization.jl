using Test
using LinearAlgebra
using Suppressor: @suppress
using Random

using QuanEstimationBase:
    Lindblad,
    GeneralScheme,
    optimize!,
    SCopt, CMopt, SMopt, SCMopt,
    AD, DE, PSO,
    SigmaX, SigmaY, SigmaZ,
    QFIM_obj, CFIM_obj,
    QFIM,
    objective

    

function test_scopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)
    obj = QFIM_obj()
    opt = SCopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    f_ini, _ = objective(obj, scheme)

    alg = AD(max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_ad, _ = objective(obj, scheme)
    @test isfinite(f_ad)
    @test f_ad > 0
    @test f_ad >= f_ini - 1e-10

    alg = DE(p_num=3, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_de = tr(QFIM(scheme))
    @test isfinite(f_de)
    @test f_de > 0

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_pso = tr(QFIM(scheme))
    @test isfinite(f_pso)
    @test f_pso > 0

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat"); isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat"); isfile("states.csv") && rm("states.csv")
end

function test_scopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    obj = QFIM_obj()
    opt = SCopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    f_ini, _ = objective(obj, scheme)

    alg = AD(max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_ad, _ = objective(obj, scheme)
    @test isfinite(f_ad)
    @test f_ad > 0
    @test f_ad >= f_ini - 1e-10

    alg = DE(p_num=3, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_de = tr(QFIM(scheme))
    @test isfinite(f_de)
    @test f_de > 0

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_pso = tr(QFIM(scheme))
    @test isfinite(f_pso)
    @test f_pso > 0

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat"); isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat"); isfile("states.csv") && rm("states.csv")
end

function test_cmopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    obj = CFIM_obj()
    opt = CMopt(ctrl_bound = [-2.0, 2.0], seed = 1234)

    alg = DE(p_num=3, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_de = tr(QFIM(scheme))
    @test isfinite(f_de)
    @test f_de > 0

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_pso = tr(QFIM(scheme))
    @test isfinite(f_pso)
    @test f_pso > 0

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat"); isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat"); isfile("measurements.csv") && rm("measurements.csv")
end

function test_cmopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    obj = CFIM_obj()
    opt = CMopt(ctrl_bound = [-2.0, 2.0], seed = 1234)

    alg = DE(p_num=3, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_de = tr(QFIM(scheme))
    @test isfinite(f_de)
    @test f_de > 0

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_pso = tr(QFIM(scheme))
    @test isfinite(f_pso)
    @test f_pso > 0

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat"); isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat"); isfile("measurements.csv") && rm("measurements.csv")
end

function test_smopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    obj = CFIM_obj()
    opt = SMopt(seed = 1234)

    alg = DE(p_num=3, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_de = tr(QFIM(scheme))
    @test isfinite(f_de)
    @test f_de > 0

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_pso = tr(QFIM(scheme))
    @test isfinite(f_pso)
    @test f_pso > 0

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat"); isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat"); isfile("measurements.csv") && rm("measurements.csv")
end

function test_smopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    obj = CFIM_obj()
    opt = SMopt(seed = 1234)

    alg = DE(p_num=3, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_de = tr(QFIM(scheme))
    @test isfinite(f_de)
    @test f_de > 0

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_pso = tr(QFIM(scheme))
    @test isfinite(f_pso)
    @test f_pso > 0

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat"); isfile("controls.csv") && rm("controls.csv")
    isfile("measurements.dat") && rm("measurements.dat"); isfile("measurements.csv") && rm("measurements.csv")
end

function test_scmopt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    obj = CFIM_obj()
    opt = SCMopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = DE(p_num=3, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_de = tr(QFIM(scheme))
    @test isfinite(f_de)
    @test f_de > 0

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_pso = tr(QFIM(scheme))
    @test isfinite(f_pso)
    @test f_pso > 0

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat"); isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat"); isfile("states.csv") && rm("states.csv")
    isfile("measurements.dat") && rm("measurements.dat"); isfile("measurements.csv") && rm("measurements.csv")
end

function test_scmopt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_NV_dynamics()
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    obj = CFIM_obj()
    opt = SCMopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

    alg = DE(p_num=3, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_de = tr(QFIM(scheme))
    @test isfinite(f_de)
    @test f_de > 0

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_pso = tr(QFIM(scheme))
    @test isfinite(f_pso)
    @test f_pso > 0

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat"); isfile("controls.csv") && rm("controls.csv")
    isfile("states.dat") && rm("states.dat"); isfile("states.csv") && rm("states.csv")
    isfile("measurements.dat") && rm("measurements.dat"); isfile("measurements.csv") && rm("measurements.csv")
end

function test_comprehensive_optimization()
    @testset "SCopt QFI" begin test_scopt_qfi() end
    @testset "SCopt QFIM" begin test_scopt_qfim() end
    @testset "CMopt QFI" begin test_cmopt_qfi() end
    @testset "CMopt QFIM" begin test_cmopt_qfim() end
    @testset "SMopt QFI" begin test_smopt_qfi() end
    @testset "SMopt QFIM" begin test_smopt_qfim() end
    @testset "SCMopt QFI" begin test_scmopt_qfi() end
    @testset "SCMopt QFIM" begin test_scmopt_qfim() end
end

test_comprehensive_optimization()
