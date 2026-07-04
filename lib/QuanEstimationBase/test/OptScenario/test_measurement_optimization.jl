using Test
using LinearAlgebra
using Suppressor: @suppress
using Random

using QuanEstimationBase: 
    SIC,
    QFIM,
    Lindblad,
    GeneralScheme,
    CFIM_obj,
    MeasurementOpt,
    DE,
    PSO,
    AD,
    optimize!,
    SigmaX, SigmaY, SigmaZ


function test_mopt_lc_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)
    F_qf = QFIM(scheme)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :LC, POVM_basis = POVM_basis, M_num = 2, seed = 1234)
    
    alg = DE(p_num = 3, ini_population = nothing, max_episode = 3, c = 1.0, cr = 0.5)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    F_cf = CFIM(scheme)
    @test isfinite(tr(F_cf))
    @test tr(inv(F_cf)) >= tr(inv(F_qf)) - 1e-10

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    F_cf = CFIM(scheme)
    @test isfinite(tr(F_cf))
    @test tr(inv(F_cf)) >= tr(inv(F_qf)) - 1e-10
    
    alg = AD(max_episode = 3)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    F_cf = CFIM(scheme)
    @test isfinite(tr(F_cf))
    @test tr(inv(F_cf)) >= tr(inv(F_qf)) - 1e-10

    isfile("f.csv") && rm("f.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
end

function test_mopt_projection_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)
    F_qf = QFIM(scheme)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype=:Projection, seed = 1234)
    
    alg = DE(p_num = 3, max_episode = 3)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    F_cf = CFIM(scheme)
    @test isfinite(tr(F_cf))
    @test tr(inv(F_cf)) >= tr(inv(F_qf)) - 1e-10

    alg = PSO(p_num = 3, max_episode = [3, 3])
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    F_cf = CFIM(scheme)
    @test isfinite(tr(F_cf))
    @test tr(inv(F_cf)) >= tr(inv(F_qf)) - 1e-10

    isfile("f.csv") && rm("f.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
end

function test_mopt_rotation_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)
    F_qf = QFIM(scheme)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :Rotation, POVM_basis = POVM_basis, seed = 1234)
    
    alg = DE(p_num = 3, max_episode = 3)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    F_cf = CFIM(scheme)
    @test isfinite(tr(F_cf))
    @test tr(inv(F_cf)) >= tr(inv(F_qf)) - 1e-10

    alg = PSO(p_num=3, max_episode=[3, 3])
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    F_cf = CFIM(scheme)
    @test isfinite(tr(F_cf))
    @test tr(inv(F_cf)) >= tr(inv(F_qf)) - 1e-10
    
    alg = AD(max_episode = 3)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    F_cf = CFIM(scheme)
    @test isfinite(tr(F_cf))
    @test tr(inv(F_cf)) >= tr(inv(F_qf)) - 1e-10

    isfile("f.csv") && rm("f.csv")
    isfile("measurements.dat") && rm("measurements.dat")
    isfile("measurements.csv") && rm("measurements.csv")
end



function test_mopt()
    @testset "Measurement Optimization CFIM" begin
        test_mopt_lc_cfi()
        test_mopt_projection_cfi()
        test_mopt_rotation_cfi()

    end
end

test_mopt()
