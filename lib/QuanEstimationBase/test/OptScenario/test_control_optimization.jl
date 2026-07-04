using Test
using LinearAlgebra
using Suppressor: @suppress
using Random
using JSON

using QuanEstimationBase:
    Lindblad, GeneralScheme, QFIM, CFIM, QFIM_obj, CFIM_obj,
    ControlOpt, autoGRAPE, optimize!, GRAPE, DE, PSO,
    SigmaX, SigmaY, SigmaZ, PlusState,
    Output, Objective, init_opt


const REFERENCES_DIR = joinpath(@__DIR__, "..", "references")

function load_golden_controlopt(name)
    return JSON.parsefile(joinpath(REFERENCES_DIR, name))
end

function capture_de_pso(alg, obj, opt, scheme; seed=1234)
    Random.seed!(seed)
    opt = init_opt(opt, scheme)
    ow = Objective(scheme, obj)
    out = Output(opt; save=false)
    @suppress optimize!(opt, alg, ow, scheme, out)
    return out.f_list[end]
end

function test_copt_qfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=PlusState(), param=dynamics)

    f_ini = QFIM(scheme)[1]

    # autoGRAPE + QFI
    Random.seed!(1234)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = autoGRAPE(Adam=true, max_episode=3, epsilon=0.01, beta1=0.90, beta2=0.99)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=QFIM_obj(), savefile=savefile)
    f_post = QFIM(scheme)[1]
    @test isfinite(f_post) && f_post >= f_ini - 1e-10
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # GRAPE_Adam + QFI
    Random.seed!(1234)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = GRAPE(Adam=true, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=QFIM_obj(), savefile=savefile)
    f_post = QFIM(scheme)[1]
    @test isfinite(f_post) && f_post >= f_ini - 1e-10
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # GRAPE(Adam=false) + QFI
    Random.seed!(1234)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = GRAPE(Adam=false, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=QFIM_obj(), savefile=savefile)
    f_post = QFIM(scheme)[1]
    @test isfinite(f_post) && f_post >= f_ini - 1e-10
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # DE + QFI (ode dynamics)
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Ode)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = DE(p_num=3, max_episode=3)
    f_de = capture_de_pso(alg, QFIM_obj(), opt, scheme)
    @test isfinite(f_de) && f_de > 0
    rm("f.csv", force=true)

    # PSO + QFI (ode)
    alg = PSO(p_num=3, max_episode=[3,3])
    f_pso = capture_de_pso(alg, QFIM_obj(), opt, scheme)
    @test isfinite(f_pso) && f_pso > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)
end

function test_copt_qfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)

    # autoGRAPE + QFI (NV multi-para)
    Random.seed!(1234)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = autoGRAPE(Adam=true, max_episode=3, epsilon=0.01, beta1=0.90, beta2=0.99)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=QFIM_obj(), savefile=savefile)
    f1 = tr(pinv(QFIM(scheme)))
    @test isfinite(f1) && f1 >= 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # PSO (NV) 
    alg = PSO(p_num=3, max_episode=[3,3])
    f_pso = capture_de_pso(alg, QFIM_obj(), opt, scheme)
    @test isfinite(f_pso) && f_pso > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # DE (NV)
    alg = DE(p_num=3, max_episode=3)
    f_de = capture_de_pso(alg, QFIM_obj(), opt, scheme)
    @test isfinite(f_de) && f_de > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # GRAPE_Adam + QFI (NV)
    alg = GRAPE(Adam=true, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=QFIM_obj(), savefile=savefile)
    f_post = tr(pinv(QFIM(scheme)))
    @test isfinite(f_post) && f_post >= 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # GRAPE(Adam=false) + QFI (NV)
    alg = GRAPE(Adam=false, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=QFIM_obj(), savefile=savefile)
    f_post = QFIM(scheme)[1]
    @test isfinite(f_post) && f_post > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)
end

function test_copt_cfi(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound, M) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics, measurement=M)

    obj = CFIM_obj(M=M)
    f_ini = CFIM(scheme)[1]

    # autoGRAPE + CFI
    Random.seed!(1234)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = autoGRAPE(Adam=true, max_episode=3, epsilon=0.01, beta1=0.90, beta2=0.99)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_post = CFIM(scheme)[1]
    @test isfinite(f_post) && f_post >= f_ini - 1e-10
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # PSO (CFI)
    alg = PSO(p_num=3, max_episode=[3,3])
    f_pso = capture_de_pso(alg, obj, opt, scheme)
    @test isfinite(f_pso) && f_pso > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # DE (CFI)
    f_de = capture_de_pso(DE(p_num=3, max_episode=3), obj, opt, scheme)
    @test isfinite(f_de) && f_de > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # GRAPE_Adam + CFI
    Random.seed!(1234)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = GRAPE(Adam=true, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_post = QFIM(scheme)[1]
    @test isfinite(f_post) && f_post > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # GRAPE(Adam=false) + CFI
    Random.seed!(1234)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = GRAPE(Adam=false, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_post = QFIM(scheme)[1]
    @test isfinite(f_post) && f_post > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)
end

function test_copt_cfim(; savefile=false)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound, M) = generate_NV_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics, measurement=M)

    obj = CFIM_obj(M=M)

    # autoGRAPE + CFI (NV multi-para)
    Random.seed!(1234)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    alg = autoGRAPE(Adam=true, max_episode=3, epsilon=0.01, beta1=0.90, beta2=0.99)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f1 = tr(pinv(CFIM(scheme)))
    @test isfinite(f1) && f1 >= 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # PSO (NV CFI)
    alg = PSO(p_num=3, max_episode=[3,3])
    f_pso = capture_de_pso(alg, obj, opt, scheme)
    @test isfinite(f_pso) && f_pso > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # DE (NV CFI)
    f_de = capture_de_pso(DE(p_num=3, max_episode=3), obj, opt, scheme)
    @test isfinite(f_de) && f_de > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # GRAPE_Adam + CFI (NV)
    alg = GRAPE(Adam=true, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_post = tr(pinv(QFIM(scheme)))
    @test isfinite(f_post) && f_post >= 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

    # GRAPE(Adam=false) + CFI (NV)
    alg = GRAPE(Adam=false, max_episode=3)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_post = QFIM(scheme)[1]
    @test isfinite(f_post) && f_post > 0
    rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)
end


function test_copt()
    @testset "Control Optimization QFIM" begin
        test_copt_qfi()
    end
    @testset "Control Optimization CFIM" begin
        test_copt_cfi()
    end
end

test_copt()
