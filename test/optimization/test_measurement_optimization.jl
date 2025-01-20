
function test_mopt_lc_de_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :LC, POVM_basis = POVM_basis, M_num = 2, seed = 1234)
    alg = DE(p_num = 3, ini_population = nothing, max_episode = 10, c = 1.0, cr = 0.5)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    rm("f.csv")
    rm("measurements.dat")
    @test true
end

function test_mopt_lc_pso_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :LC, POVM_basis = POVM_basis, M_num = 2, seed = 1234)
    alg =
        alg = PSO(
            p_num = 3,
            ini_particle = nothing,
            max_episode = [10, 10],
            c0 = 1.0,
            c1 = 2.0,
            c2 = 2.0,
        )
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    rm("f.csv")
    rm("measurements.dat")
    @test true
end

function test_mopt_lc_ad_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :LC, POVM_basis = POVM_basis, M_num = 2, seed = 1234)
    alg = AD(Adam = false, max_episode = 10, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)

    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    rm("f.csv")
    rm("measurements.dat")
    @test true
end

function test_mopt_projection_de_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :Rotation, POVM_basis = POVM_basis, seed = 1234)
    alg = DE(p_num = 10, ini_population = nothing, max_episode = 1000, c = 1.0, cr = 0.5)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    rm("f.csv")
    rm("measurements.dat")
    @test true
end

function test_mopt_projection_pso_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :Rotation, POVM_basis = POVM_basis, seed = 1234)
    alg = PSO(
        p_num = 3,
        ini_particle = nothing,
        max_episode = [10, 10],
        c0 = 1.0,
        c1 = 2.0,
        c2 = 2.0,
    )
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)
    rm("f.csv")
    rm("measurements.dat")
    @test true
end

function test_mopt_rotation_de_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :Rotation, POVM_basis = POVM_basis, seed = 1234)
    alg = DE(p_num = 3, ini_population = nothing, max_episode = 10, c = 1.0, cr = 0.5)
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    rm("f.csv")
    rm("measurements.dat")
    @test true
end

function test_mopt_rotation_pso_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :Rotation, POVM_basis = POVM_basis, seed = 1234)
    alg = PSO(
            p_num = 3,
            ini_particle = nothing,
            max_episode = [10, 10],
            c0 = 1.0,
            c1 = 2.0,
            c2 = 2.0,
        )
    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    rm("f.csv")
    rm("measurements.dat")
    @test true
end

function test_mopt_rotation_ad_cfi(; savefile = false)
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dim = size(rho0, 1)
    POVM_basis = SIC(dim)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    obj = CFIM_obj()
    opt = MeasurementOpt(mtype = :Rotation, POVM_basis = POVM_basis, seed = 1234)
    alg = AD(Adam = false, max_episode = 10, epsilon = 0.01, beta1 = 0.90, beta2 = 0.99)

    @suppress optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = savefile)

    rm("f.csv")
    rm("measurements.dat")
    @test true
end

function test_mopt()
    @testset "Measurement Optimization DE CFIM" begin
        test_mopt_lc_de_cfi()
        test_mopt_lc_de_cfi(savefile = true)
        test_mopt_projection_de_cfi()
        test_mopt_projection_de_cfi(savefile = true)
        test_mopt_rotation_de_cfi()
        test_mopt_rotation_de_cfi(savefile = true)

    end

    @testset "Measurement Optimization PSO CFIM" begin
        test_mopt_lc_pso_cfi()
        test_mopt_lc_pso_cfi(savefile = true)
        test_mopt_projection_pso_cfi()
        test_mopt_projection_pso_cfi(savefile = true)
        test_mopt_rotation_pso_cfi()
        test_mopt_rotation_pso_cfi(savefile = true)
    end

    @testset "Measurement Optimization AD CFIM" begin
        test_mopt_lc_ad_cfi()
        test_mopt_lc_ad_cfi(savefile = true)
        test_mopt_rotation_ad_cfi()
        test_mopt_rotation_ad_cfi(savefile = true)
    end
end

test_mopt()
