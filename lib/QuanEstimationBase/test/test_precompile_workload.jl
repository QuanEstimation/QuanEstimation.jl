# Semantic gates for src/precompile_workload.jl. These are the same canonical
# cells executed by PrecompileTools; an API or numerical regression therefore
# fails both package precompilation and the explicit runtime test suite.

const QEB = QuanEstimationBase

@testset "Precompile workload: Fisher queries" begin
    F = QEB._pcw_qfim()
    @test size(F) == (1, 1)
    @test F ≈ reshape([1.0], 1, 1) rtol = 1e-10

    I = QEB._pcw_cfim()
    @test size(I) == (1, 1)
    @test all(isfinite, I)
    @test all(>=(0), I)
    @test I[1, 1] <= 1 + 1e-8

    K = QEB._pcw_kraus()
    @test size(K) == (1, 1)
    @test all(isfinite, K)
end

@testset "Precompile workload: Tsit5 ODE query" begin
    F = QEB._pcw_ode()
    @test size(F) == (1, 1)
    @test all(isfinite, F)
    @test all(>(0), F)
    # Do not impose the 1.0 self-hosted RK45 analytic oracle on 0.5: this
    # legacy Tsit5 trajectory returns about 1.0533 for the same fixture.
    @test F ≈ reshape([1.0533231496810913], 1, 1) rtol = 1e-4
end

@testset "Precompile workload: control optimization" begin
    for alg in (
        GRAPE(; max_episode = 2),
        autoGRAPE(; max_episode = 2),
        DE(; max_episode = 2, p_num = 4),
        PSO(; max_episode = [2, 2], p_num = 4),
    )
        F = QEB._pcw_optimize(alg)
        @test size(F) == (1, 1)
        @test all(isfinite, F)
        @test all(>=(0), F)
    end

    F_ode = QEB._pcw_optimize(GRAPE(; max_episode = 2); dyn_method = :Ode)
    @test all(isfinite, F_ode)

    F_waveform = QEB._pcw_optimize(
        GRAPE(; max_episode = 2);
        dyn_method = :Ode,
        ctrl = SineCTRL(; A = 0.3, ω = 2pi, ϕ = 0.0),
    )
    @test all(isfinite, F_waveform)
end

@testset "Precompile workload: measurement optimization" begin
    coefficients = QEB._pcw_optimize_meas()
    @test length(coefficients) == 8
    @test all(isfinite, coefficients)

    initial_opt = MeasurementOpt(;
        mtype = :LC,
        POVM_basis = SIC(2),
        M_num = 2,
        seed = 42,
    )
    QEB.init_opt(initial_opt, QEB._pcw_lindblad_scheme())
    @test !isapprox(coefficients, reduce(vcat, initial_opt.B))
end

@testset "Precompile workload: MZI and NHB" begin
    result = QEB._pcw_mzi()
    @test result.y == [0, 1]
    @test length(result.x) == 3
    @test all(isfinite, result.x)
    @test result.x[1] == 0.0
    @test any(!iszero, result.x[2:end])

    bound = QEB._pcw_nhb()
    @test bound isa Real
    @test isfinite(bound)
    @test bound > 0
end
