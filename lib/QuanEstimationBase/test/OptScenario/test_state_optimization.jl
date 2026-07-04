using Test
using LinearAlgebra
using Suppressor: @suppress
using Random

using QuanEstimationBase: 
    Lindblad,
    GeneralScheme,
    QFIM_obj,
    QFIM,
    StateOpt,
    AD,
    optimize!,
    PSO,
    DE,
    NM,
    RI


function test_sopt_qfi(; savefile = false)
    (; tspan, psi, H0, dH, decay) = generate_LMG1_dynamics()

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = psi, param = dynamics)

    obj = QFIM_obj()
    f0 = QFIM(scheme)[1]

    opt = StateOpt(psi = psi, seed = 1234)
    alg = AD(Adam = true, max_episode = 3, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    f1 = QFIM(scheme)[1]
    @test f1 >= f0 || isapprox(f1, f0; atol=1e-5)
    
    # Ensure cleanup happens even if tests fail
    try
        isfile("f.csv") && rm("f.csv")
        isfile("states.dat") && rm("states.dat")
    catch e
        @warn "Cleanup failed: $e"
    end

    # Test other algorithms
    for alg in [
        PSO(p_num=3, max_episode=[3, 3]),
        DE(p_num=3, max_episode=3),
        NM(p_num=5, max_episode=3)
    ]
        @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
        f_post = QFIM(scheme)[1]
        @test isfinite(f_post)
        try
            isfile("f.csv") && rm("f.csv")
            isfile("states.dat") && rm("states.dat")
        catch e
            @warn "Cleanup failed: $e"
        end
    end

    # Test RI algorithm separately
    alg = RI()
    scheme = generate_scheme_kraus()
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_post = tr(QFIM(scheme))
    @test isfinite(f_post)
    @test f_post > 0
    try
        isfile("f.csv") && rm("f.csv")
        isfile("states.dat") && rm("states.dat")
    catch e
        @warn "Cleanup failed: $e"
    end
end

function test_sopt_qfim(; savefile = false)
    (; tspan, psi, H0, dH, decay, W) = generate_LMG2_dynamics()

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = psi, param = dynamics)

    obj = QFIM_obj(W = W)
    f0 = tr(pinv(QFIM(scheme)))

    opt = StateOpt(psi = psi, seed = 1234)
    alg = AD(Adam = true, max_episode = 3, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    f1 = tr(pinv(QFIM(scheme)))
    @test f1 <= f0 || isapprox(f1, f0; atol=1e-5)
    
    try
        isfile("f.csv") && rm("f.csv")
        isfile("states.dat") && rm("states.dat")
        isfile("states.csv") && rm("states.csv")
    catch e
        @warn "Cleanup failed: $e"
    end

    # Test other algorithms
    for alg in [
        PSO(p_num=3, max_episode=[3, 3]),
        DE(p_num=3, max_episode=3),
        NM(p_num=5, max_episode=3)
    ]
        @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
        f_post = QFIM(scheme)[1]
        @test isfinite(f_post)
        try
            isfile("f.csv") && rm("f.csv")
            isfile("states.dat") && rm("states.dat")
            isfile("states.csv") && rm("states.csv")
        catch e
            @warn "Cleanup failed: $e"
        end
    end
end

function test_sopt_cfi(; savefile = false)
    (; tspan, psi, H0, dH) = generate_LMG1_dynamics()

    dynamics = Lindblad(H0, dH, tspan; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = psi, param = dynamics)

    obj = CFIM_obj()
    f0 = CFIM(scheme)[1]

    opt = StateOpt(psi = psi, seed = 1234)
    alg = AD(Adam = true, max_episode = 3, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    f1 = CFIM(scheme)[1]
    @test f1 >= f0 || isapprox(f1, f0; atol=1e-5)
    
    try
        isfile("f.csv") && rm("f.csv")
        isfile("states.dat") && rm("states.dat")
        isfile("states.csv") && rm("states.csv")
    catch e
        @warn "Cleanup failed: $e"
    end

    # Test other algorithms
    for alg in [
        PSO(p_num=3, max_episode=[3, 3]),
        DE(p_num=3, max_episode=3),
        NM(p_num=5, max_episode=3)
    ]
        @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
        f_post = QFIM(scheme)[1]
        @test isfinite(f_post)
        try
            isfile("f.csv") && rm("f.csv")
            isfile("states.dat") && rm("states.dat")
            isfile("states.csv") && rm("states.csv")
        catch e
            @warn "Cleanup failed: $e"
        end
    end
end

function test_sopt_cfim(; savefile = false)
    (; tspan, psi, H0, dH, decay, W) = generate_LMG2_dynamics()

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = psi, param = dynamics)

    obj = CFIM_obj(W = W)
    f0 = LinearAlgebra.tr(pinv(CFIM(scheme)))

    opt = StateOpt(psi = psi, seed = 1234)
    alg = AD(Adam = true, max_episode = 3, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    f1 = LinearAlgebra.tr(pinv(CFIM(scheme)))
    @test f1 <= f0 || isapprox(f1, f0; atol=1e-5)
    
    try
        isfile("f.csv") && rm("f.csv")
        isfile("states.dat") && rm("states.dat")
        isfile("states.csv") && rm("states.csv")
    catch e
        @warn "Cleanup failed: $e"
    end

    # Test other algorithms
    for alg in [
        PSO(p_num=3, max_episode=[3, 3]),
        DE(p_num=3, max_episode=3),
        NM(p_num=5, max_episode=3)
    ]
        @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
        f_post = QFIM(scheme)[1]
        @test isfinite(f_post)
        try
            isfile("f.csv") && rm("f.csv")
            isfile("states.dat") && rm("states.dat")
            isfile("states.csv") && rm("states.csv")
        catch e
            @warn "Cleanup failed: $e"
        end
    end
end


function test_sopt_kraus_cfi(; savefile = false)
    scheme = generate_scheme_kraus()
    obj = CFIM_obj()
    opt = StateOpt(seed = 1234)
    alg = RI()
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=savefile)
    f_post = tr(CFIM(scheme))
    @test isfinite(f_post)
    @test f_post > 0
    try
        isfile("f.csv") && rm("f.csv")
        isfile("states.dat") && rm("states.dat")
    catch e
        @warn "Cleanup failed: $e"
    end
end

function test_sopt_nm_qubit_ode()
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    psi = rho0[:, 1] / norm(rho0[:, 1])
    for dyn_method in (:Expm, :Ode)
        dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=dyn_method)
        scheme = GeneralScheme(; probe=psi, param=dynamics)
        obj = QFIM_obj()
        opt = StateOpt(psi=psi, seed=1234)
        alg = NM(p_num=5, max_episode=3)
        @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
        f_post = QFIM(scheme)[1]
        @test isfinite(f_post)
        @test f_post > 0
        try
            isfile("f.csv") && rm("f.csv")
            isfile("states.dat") && rm("states.dat")
        catch e
            @warn "Cleanup failed: $e"
        end
    end
end


function test_sopt()
    @testset "State Optimization QFIM" begin
        test_sopt_qfi()
        test_sopt_qfim()
    end

    @testset "State Optimization CFIM" begin
        test_sopt_cfi()
        test_sopt_cfim()
        test_sopt_kraus_cfi()
    end

    @testset "State Optimization NM Qubit" begin
        test_sopt_nm_qubit_ode()
    end
end

test_sopt()
