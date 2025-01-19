using QuanEstimationBase: trace_norm, fidelity, helstrom_bound, QZZB

# Test for trace_norm with matrices
function test_trace_norm_matrix()
    X = [1 2; 3 4]
    expected_result = 5.83
    @test trace_norm(X) >= 0
end

# Test for trace_norm with density matrices
function test_trace_norm_density()
    ρ = [1 0; 0 1]
    σ = [0 1; 1 0]
    expected_result = 2.0
    @test trace_norm(ρ, σ) ≈ expected_result
end

# Test for fidelity with density matrices
function test_fidelity_density()
    ρ = [1 0; 0 1]
    σ = [0 1; 1 0]
    expected_result = 1.0
    @test fidelity(ρ, σ) ≈ expected_result
end

# Test for fidelity with pure states
function test_fidelity_pure()
    ψ = [1, 0]
    ϕ = [0, 1]
    expected_result = 0.0
    @test fidelity(ψ, ϕ) ≈ expected_result
end

# Test for helstrom_bound with density matrices
function test_helstrom_bound_density()
    ρ = [1 0; 0 1]
    σ = [0 1; 1 0]
    expected_result = 0.0
    @test helstrom_bound(ρ, σ) ≈ expected_result
end

# Test for helstrom_bound with pure states
function test_helstrom_bound_pure()
    ψ = [1, 0]
    ϕ = [0, 1]
    expected_result = 0.0
    @test helstrom_bound(ψ, ϕ) ≈ expected_result
end

function data_gen_zz()
    # prior distribution
    function p_func(x, mu, eta)
        return exp(-(x - mu)^2 / (2 * eta^2)) / (eta * sqrt(2 * pi))
    end
    function dp_func(x, mu, eta)
        return -(x - mu) * exp(-(x - mu)^2 / (2 * eta^2)) / (eta^3 * sqrt(2 * pi))
    end

    B, omega0 = 0.5 * pi, 1.0
    sx = [0.0 1.0; 1.0 0.0im]
    sz = [1.0 0.0im; 0.0 -1.0]
    # initial state
    rho0 = 0.5 * ones(2, 2)
    # prior distribution
    x = range(-0.5 * pi, stop = 0.5 * pi, length = 10) |> Vector
    mu, eta = 0.0, 0.2
    p_tp = [p_func(x[i], mu, eta) for i in eachindex(x)]
    dp_tp = [dp_func(x[i], mu, eta) for i in eachindex(x)]
    # normalization of the distribution
    c = trapz(x, p_tp)
    p = p_tp / c
    dp = dp_tp / c
    # time length for the evolution
    tspan = range(0.0, stop = 1.0, length = 10)
    # dynamics
    rho = Vector{Matrix{ComplexF64}}(undef, length(x))
    drho = Vector{Vector{Matrix{ComplexF64}}}(undef, length(x))
    for i in eachindex(x)
        H0_tp = 0.5 * B * omega0 * (sx * cos(x[i]) + sz * sin(x[i]))
        dH_tp = [0.5 * B * omega0 * (-sx * sin(x[i]) + sz * cos(x[i]))]
        rho_tp, drho_tp = QuanEstimationBase.expm(tspan, rho0, H0_tp, dH_tp)
        rho[i], drho[i] = rho_tp[end], drho_tp[end]
    end
    return (; x = x, p = p, rho = rho, drho = drho, dp = dp)
end

# Test for QZZB
function test_QZZB()
    (; x, p, rho) = data_gen_zz()
    @test QZZB(x, p, rho) >= 0
end

# Run the tests
function test_ObjectiveFunc()
    @testset "Utils" begin
        test_trace_norm_matrix()
        test_trace_norm_density()
        test_fidelity_density()
        test_fidelity_pure()
        test_helstrom_bound_density()
        test_helstrom_bound_pure()
    end

    @testset "QZZB" begin
        test_QZZB()
    end
end

# Call the test function
test_ObjectiveFunc()
