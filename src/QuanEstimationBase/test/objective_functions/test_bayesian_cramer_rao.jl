using Test
using QuanEstimationBase: BCFIM, BQFIM, BQCRB, BCRB

function data_gen_bayes()
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

# Test for BCFIM
function test_BCFIM()
    (; x, p, rho, drho) = data_gen_bayes()
    M = QuanEstimationBase.SIC(2)
    result = BCFIM([x], p, rho, drho, M = M)
    # Add assertions here
end

# Test for BQFIM
function test_BQFIM()
    (; x, p, rho, drho) = data_gen_bayes()
    LDtype = :SLD
    eps = 1e-6
    result = BQFIM([x], p, rho, drho, LDtype = LDtype, eps = eps)
    # Add assertions here
end

# Test for BQCRB
function test_BQCRB()
    (; x, p, dp, rho, drho) = data_gen_bayes()
    LDtype = :SLD
    btypes = [1, 2, 3]
    eps = 1e-6
    [
        BQCRB([x], p, dp, rho, drho, LDtype = LDtype, btype = btype, eps = eps) for
        btype in btypes
    ]
    # Add assertions here
end

# Test for BCRB
function test_BCRB()
    (; x, p, dp, rho, drho) = data_gen_bayes()
    btype = 1
    eps = 1e-6
    btypes = [1, 2, 3]
    [BCRB([x], p, dp, rho, drho, btype = btype, eps = eps) for btype in btypes]
    # Add assertions here
end

# Run the tests
function test_BayesianCramerRao()
    @testset "BCFIM" begin
        test_BCFIM()
    end

    @testset "BQFIM" begin
        test_BQFIM()
    end

    @testset "BQCRB" begin
        test_BQCRB()
    end

    @testset "BCRB" begin
        test_BCRB()
    end
end

# Call the test function
test_BayesianCramerRao()
