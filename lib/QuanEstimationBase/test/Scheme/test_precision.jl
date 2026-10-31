using LinearAlgebra, Test
using QuanEstimationBase

function _pauli_matrices()
    sx = ComplexF64[0 1; 1 0]
    sy = ComplexF64[0 -im; im 0]
    sz = ComplexF64[1 0; 0 -1]
    return sx, sy, sz
end

function _bloch_qubit(s, θ, φ)
    sx, sy, sz = _pauli_matrices()
    n = [sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ)]
    ρ = Matrix((I + s * (n[1] * sx + n[2] * sy + n[3] * sz)) / 2)
    dθn = [cos(θ) * cos(φ), cos(θ) * sin(φ), -sin(θ)]
    dφn = [-sin(θ) * sin(φ), sin(θ) * cos(φ), 0.0]
    dρθ = s * (dθn[1] * sx + dθn[2] * sy + dθn[3] * sz) / 2
    dρφ = s * (dφn[1] * sx + dφn[2] * sy + dφn[3] * sz) / 2
    return ρ, [dρθ, dρφ]
end

function test_mixed_state_analytic_qfim()
    s, θ, φ = 0.7, 0.9, 0.4
    ρ, dρ = _bloch_qubit(s, θ, φ)
    F = QFIM(ρ, dρ)
    Fana = [s^2 0.0; 0.0 s^2 * sin(θ)^2]
    @test norm(F - Fana) / norm(Fana) < 1e-12
end

function test_near_pure_analytic_qfim()
    s, θ, φ = 1 - 2e-10, 0.9, 0.4
    ρ, dρ = _bloch_qubit(s, θ, φ)
    λmin = minimum(real.(eigvals(Hermitian(ρ))))
    @test 1e-11 < λmin < 1e-9

    F, diag = QFIM(ρ, dρ; diagnostics = true)
    Fana = [s^2 0.0; 0.0 s^2 * sin(θ)^2]
    @test norm(F - Fana) / norm(Fana) < 1e-10
    @test diag isa NumericalDiagnostics
    @test diag.quantity == :QFIM
    @test diag.spectral_floor ≈ default_guard(Float64)
end

function test_rld_support_condition()
    ρ = ComplexF64[1 0; 0 0]
    dρ = ComplexF64[0 0; 0 1]
    @test_throws ArgumentError RLD(ρ, dρ)
    @test_throws ArgumentError QuanEstimationBase.QFIM_RLD(ρ, dρ)
end

function test_rld_lld_qfim_equality()
    ρ = ComplexF64[0.5 0.0 0.0; 0.0 0.3 0.0; 0.0 0.0 0.2]
    dρ = [
        ComplexF64[0.1 0.2 0.0; 0.2 -0.1 0.1; 0.0 0.1 0.0],
        ComplexF64[0.0 0.1 0.2; 0.1 0.0 -0.1; 0.2 -0.1 0.0],
    ]
    F_RLD = QuanEstimationBase.QFIM_RLD(ρ, dρ)
    F_LLD = QuanEstimationBase.QFIM_LLD(ρ, dρ)
    @test F_RLD ≈ F_LLD
end

function test_cfim_resolution_floor()
    δ = 1e-10
    ρ = ComplexF64[1-δ 0; 0 δ]
    dρ = ComplexF64[0 0; 0 1]
    M = [ComplexF64[1 0; 0 0], ComplexF64[0 0; 0 1]]

    F, diag = CFIM(ρ, [dρ], M; guard = 1.5e-10, diagnostics = true)
    @test :resolution_floor in diag.flags
    @test diag.numerical_rank == 1
    @test diag.deleted_mass ≈ δ atol = 1e-14
    @test diag.truncation_bias_bound ≈ δ atol = 1e-14
    @test isfinite(diag.upper_bound)
    @test diag.upper_bound >= F[1, 1]
    reference = 1 / δ
    @test F[1, 1] <= reference
    @test reference <= diag.upper_bound
end

function test_guarded_objective()
    F = [2.0 0.0; 0.0 4.0]
    W = Matrix{Float64}(I, 2, 2)
    value, diag = guarded_objective(F, W)
    @test value ≈ 0.75
    @test diag.numerical_rank == 2

    F0 = [1.0 0.0; 0.0 0.0]
    @test_throws ArgumentError guarded_objective(F0, W)
    value0, _ = guarded_objective(F0, [1.0 0.0; 0.0 0.0])
    @test value0 ≈ 1.0
end

function test_nonhermitian_input()
    ρ = ComplexF64[0.6 0.1; 0.1 0.4]
    dρ = ComplexF64[0.2 0.3; 0.3 -0.2]
    B = ComplexF64[0 1; 0 0]
    ρ_nh = ρ + 1e-6 * B

    L, diag = SLD(ρ_nh, dρ; diagnostics = true)
    @test :hermiticity_corrected in diag.flags
    Lref = SLD((ρ_nh + ρ_nh') / 2, (dρ + dρ') / 2)
    @test L ≈ Lref
end

function test_low_temperature_thermometry()
    ε, T = 1.0, 0.05
    p(T) = 1 / (1 + exp(ε / T))
    K(T) = [
        sqrt(1 - p(T)) * ComplexF64[1 0; 0 0],
        sqrt(1 - p(T)) * ComplexF64[0 1; 0 0],
        sqrt(p(T)) * ComplexF64[0 0; 1 0],
        sqrt(p(T)) * ComplexF64[0 0; 0 1],
    ]
    function dK(T)
        q = p(T)
        dq = q * (1 - q) * ε / T^2
        d1 = -dq / (2 * sqrt(1 - q))
        d2 = dq / (2 * sqrt(q))
        return [
            [d1 * ComplexF64[1 0; 0 0]],
            [d1 * ComplexF64[0 1; 0 0]],
            [d2 * ComplexF64[0 0; 1 0]],
            [d2 * ComplexF64[0 0; 0 1]],
        ]
    end
    M = [ComplexF64[1 0; 0 0], ComplexF64[0 0; 0 1]]

    scheme = GeneralScheme(;
        probe = ComplexF64[1.0, 0.0],
        param = Kraus(K, dK, T),
        measurement = M,
    )
    F_ana = p(T) * (1 - p(T)) * (ε / T^2)^2
    F_Q = QFIM(scheme)[1, 1]
    F_C = CFIM(scheme)[1, 1]
    @test isapprox(F_Q, F_ana; rtol = 1e-14)
    @test isapprox(F_C, F_ana; rtol = 1e-14)

    rho = ComplexF64[1-p(T) 0; 0 p(T)]
    dp = p(T) * (1 - p(T)) * ε / T^2
    drho = ComplexF64[-dp 0; 0 dp]
    @test isapprox(QFIM(rho, drho), F_ana; rtol = 1e-14)
    @test isapprox(CFIM(rho, drho, M), F_ana; rtol = 1e-14)
end

function test_precision_spec_validation()
    @test_throws ArgumentError PrecisionSpec(Float64; tol = 0)
    @test_throws ArgumentError PrecisionSpec(Float64; tol = NaN)
    @test_throws ArgumentError PrecisionSpec(Float64; input_error = -1)
    @test_throws ArgumentError PrecisionSpec(Float64; guard = -1)
    @test PrecisionSpec(Float64; guard = 0).guard == 0
end

function test_eps_deprecation_path()
    s, θ, φ = 0.7, 0.9, 0.4
    ρ, dρ = _bloch_qubit(s, θ, φ)
    F_default = QFIM(ρ, dρ)
    F_eps = @test_deprecated QFIM(ρ, dρ; eps = 1e-8)
    @test F_eps ≈ F_default
end

function test_float32_relative_guard()
    ρ = ComplexF32[1 0; 0 1.0f-8]
    dρ = ComplexF32[0 0; 0 1]
    spec32 = PrecisionSpec(Float32)

    F32, _, diag32 = QuanEstimationBase._qfim_sld_solve(ρ, [dρ]; spec = spec32)
    @test diag32.spectral_floor == Float64(default_guard(Float32))
    @test all(isfinite, F32)
    @test maximum(abs, F32) < 1.0

    F64, _, _ = QuanEstimationBase._qfim_sld_solve(ρ, [dρ]; spec = PrecisionSpec(Float64))
    @test F64[1, 1] > 1e6
end

@testset "test_precision" begin
    @testset "Mixed-state analytic QFIM" begin
        test_mixed_state_analytic_qfim()
    end
    @testset "Near-pure analytic QFIM" begin
        test_near_pure_analytic_qfim()
    end
    @testset "RLD support condition" begin
        test_rld_support_condition()
    end
    @testset "RLD/LLD QFIM equality" begin
        test_rld_lld_qfim_equality()
    end
    @testset "CFIM resolution floor" begin
        test_cfim_resolution_floor()
    end
    @testset "Guarded objective" begin
        test_guarded_objective()
    end
    @testset "Non-Hermitian input" begin
        test_nonhermitian_input()
    end
    @testset "Low-temperature thermometry" begin
        test_low_temperature_thermometry()
    end
    @testset "PrecisionSpec validation" begin
        test_precision_spec_validation()
    end
    @testset "eps deprecation path" begin
        test_eps_deprecation_path()
    end
    @testset "Float32 relative guard" begin
        test_float32_relative_guard()
    end
end
