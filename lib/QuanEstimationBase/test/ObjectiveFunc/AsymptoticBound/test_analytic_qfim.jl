using QuanEstimationBase
using Test
using LinearAlgebra
using StableRNGs
rng = StableRNG(1234)

# =========== §0.4-A: Single-qubit pure state QFIM (Eq. 574-576) ===========
@testset "A. Single-qubit pure state QFIM" begin
    for _ in 1:5
        θ = rand(rng) * π
        φ = rand(rng) * 2π
        ψ = [cos(θ); sin(θ) * exp(im * φ)]
        ρ = ψ * ψ'
        ∂θψ = [-sin(θ); cos(θ) * exp(im * φ)]
        ∂φψ = [0.0; im * sin(θ) * exp(im * φ)]
        ∂θρ = ∂θψ * ψ' + ψ * ∂θψ'
        ∂φρ = ∂φψ * ψ' + ψ * ∂φψ'
        F = QFIM(ρ, [∂θρ, ∂φρ]; LDtype=:SLD)
        F_expected = [4.0 0.0; 0.0 sin(2θ)^2]
        @test isapprox(F, F_expected, rtol=1e-10)
    end
end

# =========== §0.4-B: Single-qubit mixed state QFIM (Eq. 602-605) ===========
@testset "B. Single-qubit mixed state QFIM" begin
    for _ in 1:5
        ρ = rand_ρ(2)
        ∂ρ₁ = rand_∂ρ(2)
        ∂ρ₂ = rand_∂ρ(2)
        F_num = QFIM(ρ, [∂ρ₁, ∂ρ₂]; LDtype=:SLD)
        F_exact = zeros(2, 2)
        pairs = [(∂ρ₁, ∂ρ₁, 1, 1), (∂ρ₁, ∂ρ₂, 1, 2), (∂ρ₂, ∂ρ₁, 2, 1), (∂ρ₂, ∂ρ₂, 2, 2)]
        for (da, db, i, j) in pairs
            F_exact[i, j] = real(tr(da * db) + tr(ρ * da * ρ * db) / det(ρ))
        end
        @test isapprox(F_num, F_exact, rtol=1e-8)
    end
end

# =========== §0.4-C: Commuting generators — covariance formula (Eq. 553-564) ===========
@testset "C. Commuting generators covariance formula" begin
    σz = [1.0 0.0im; 0.0 -1.0]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    H1 = kron(σz, I2)
    H2 = kron(I2, σz)
    ψ0 = [1.0; 0.0; 0.0; 1.0] / sqrt(2)

    for t in [0.1, 0.5, 1.0, 2.0]
        U = exp(-im * (H1 + H2) * t)
        ψt = U * ψ0
        ρ = ψt * ψt'
        ∂₁ρ = -im * t * (H1 * ρ - ρ * H1)
        ∂₂ρ = -im * t * (H2 * ρ - ρ * H2)
        F = QuanEstimationBase.QFIM_pure(ρ, [∂₁ρ, ∂₂ρ])
        F_expected = 4 * t^2 * [1.0 1.0; 1.0 1.0]
        @test isapprox(F, F_expected, rtol=1e-10)
    end
end

# ===== §0.4-D: Dephasing qubit dual-parameter QFIM (Eq. 617-635) =====
# Master eq: ∂_t ρ = -i[Bσ_z, ρ] + (γ/2)(σ_z ρ σ_z - ρ)
# With B = ω/2, γ_lit = 2γ_SysC.  Initial state |+⟩: ρ₀₀=ρ₁₁=ρ₀₁=½.
@testset "D. Dephasing qubit dual-parameter QFIM" begin
    for γ in [0.01, 0.1, 0.5]
        for t in [0.1, 0.5, 1.0]
            B = 1.0
            ρ01_0 = 0.5
            ρ01_t = ρ01_0 * exp(-2im * B * t - γ * t)
            ρ = ComplexF64[0.5 ρ01_t; conj(ρ01_t) 0.5]

            ∂Bρ01 = -im * t * exp(-2im * B * t - γ * t)
            ∂Bρ = ComplexF64[0.0 ∂Bρ01; conj(∂Bρ01) 0.0]

            ∂γρ01 = -t/2 * exp(-2im * B * t - γ * t)
            ∂γρ = ComplexF64[0.0 ∂γρ01; conj(∂γρ01) 0.0]

            F_num = QFIM(ρ, [∂Bρ, ∂γρ]; LDtype=:SLD)
            F_BB_exact = 16 * abs2(ρ01_0) * exp(-2γ * t) * t^2
            F_γγ_exact = t^2 / (exp(2γ * t) - 1)
            @test isapprox(F_num[1,1], F_BB_exact, rtol=1e-8)
            @test isapprox(F_num[2,2], F_γγ_exact, rtol=1e-8)
            @test abs(F_num[1,2]) < 1e-10
            @test abs(F_num[2,1]) < 1e-10
        end
    end
end

# ===== §0.4-E: Thermal-state temperature QFI (Eq. 1210) =====
# Two-level system H = ωσ_z.  F_TT = C_v/T², C_v = (⟨H²⟩-⟨H⟩²)/T².
@testset "E. Thermal-state temperature QFI" begin
    ω = 1.0
    I2 = Matrix{ComplexF64}(I, 2, 2)
    for T in [0.5, 1.0, 2.0, 5.0]
        β = 1/T
        Z = 2cosh(β * ω)
        ρ = ComplexF64[exp(-β*ω)/Z 0; 0 exp(β*ω)/Z]
        dρ00_dβ = -ω / (2cosh(β*ω)^2)
        dρ11_dβ = ω / (2cosh(β*ω)^2)
        dρ_dβ = ComplexF64[dρ00_dβ 0; 0 dρ11_dβ]
        ∂ρ_∂T = dρ_dβ * (-β^2)

        F_code = QFIM(ρ, ∂ρ_∂T; LDtype=:SLD)
        F_exact = ω^2 / (T^4 * cosh(β*ω)^2)
        @test isapprox(F_code, F_exact, rtol=1e-8)
    end
end

# =========== §0.4-F: Pure state SLD explicit form (Eq. 543-544) ===========
@testset "F. Pure state SLD explicit form" begin
    σz = [1.0 0.0; 0.0 -1.0]
    ψ0 = [1.0; 1.0] / sqrt(2)
    H = σz / 2
    t = 0.5
    U = exp(-im * H * t)
    ψt = U * ψ0
    ρ = ψt * ψt'
    ∂ψt = -im * H * ψt
    ∂ρ = ∂ψt * ψt' + ψt * ∂ψt'

    L_code = SLD(ρ, ∂ρ; rep="original")
    L_pure = 2 * ∂ρ
    L_explicit = 2 * (ψt * ∂ψt' + ∂ψt * ψt')

    @test norm(L_code - L_pure) < 1e-12
    @test norm(L_code - L_explicit) < 1e-12
end

# =========== §0.4-G: QFIM mathematical properties ===========
@testset "G. QFIM mathematical properties" begin
    # G.1: Unitary invariance
    for _ in 1:5
        N = 2
        ρ = rand_ρ(N)
        ∂ρ = rand_∂ρ(N)
        U = qr(randn(ComplexF64, N, N) + im * randn(ComplexF64, N, N)).Q
        ρU = U * ρ * U'
        ∂ρU = U * ∂ρ * U'
        F = QuanEstimationBase.QFIM_SLD(ρ, ∂ρ)
        FU = QuanEstimationBase.QFIM_SLD(ρU, ∂ρU)
        @test isapprox(F, FU, rtol=1e-12)
    end

    # G.2: Reparametrization invariance
    σz = [1.0 0.0; 0.0 -1.0]
    ω = 1.0
    γ_val = 0.1
    t = 0.5
    ρt = [0.5 (0.5 * exp(-im * ω * t - 2 * γ_val * t));
          (0.5 * exp(im * ω * t - 2 * γ_val * t)) 0.5]
    dρ01_dω = -im * t * 0.5 * exp(-im * ω * t - 2 * γ_val * t)
    ∂ωρ = [0.0 dρ01_dω; conj(dρ01_dω) 0.0]
    F_omega = QuanEstimationBase.QFIM_SLD(ρt, ∂ωρ)
    ∂ηρ = ∂ωρ / 2.0
    F_eta = QuanEstimationBase.QFIM_SLD(ρt, ∂ηρ)
    @test isapprox(F_eta, F_omega / 4.0, rtol=1e-10)

    # G.3: Direct sum property
    for _ in 1:3
        ρ1 = rand_ρ(2)
        ∂ρ1 = rand_∂ρ(2)
        ρ2 = rand_ρ(3)
        ∂ρ2 = rand_∂ρ(3)
        ρ_ds = [ρ1 zeros(ComplexF64, 2, 3); zeros(ComplexF64, 3, 2) ρ2]
        ∂ρ_ds = [∂ρ1 zeros(ComplexF64, 2, 3); zeros(ComplexF64, 3, 2) ∂ρ2]
        F1 = QuanEstimationBase.QFIM_SLD(ρ1, ∂ρ1)
        F2 = QuanEstimationBase.QFIM_SLD(ρ2, ∂ρ2)
        F_ds = QuanEstimationBase.QFIM_SLD(ρ_ds, ∂ρ_ds)
        @test isapprox(F_ds, F1 + F2, rtol=1e-10)
    end

    # G.4: Convexity — F(λρ₁+(1-λ)ρ₂) ≤ λF(ρ₁) + (1-λ)F(ρ₂)
    for _ in 1:10
        ρ1 = rand_ρ(2)
        ∂ρ1 = rand_∂ρ(2)
        ρ2 = rand_ρ(2)
        ∂ρ2 = rand_∂ρ(2)
        λ = rand(rng)
        ρ_mix = λ * ρ1 + (1 - λ) * ρ2
        ∂ρ_mix = λ * ∂ρ1 + (1 - λ) * ∂ρ2
        F_mix = QuanEstimationBase.QFIM_SLD(ρ_mix, ∂ρ_mix)
        F_bound = λ * QuanEstimationBase.QFIM_SLD(ρ1, ∂ρ1) +
                  (1 - λ) * QuanEstimationBase.QFIM_SLD(ρ2, ∂ρ2)
        @test F_mix ≤ F_bound + 1e-8
    end

    # G.5: RLD ≥ SLD for single-parameter full-rank ρ
    # SLD gives the tightest CR bound: 1/F_SLD ≥ 1/F_RLD → F_RLD ≥ F_SLD
    for _ in 1:20
        N = rand(2:5)
        ρ = rand_ρ(N)
        ∂ρ = rand_∂ρ(N)
        F_s = QuanEstimationBase.QFIM_SLD(ρ, ∂ρ)
        F_r = QuanEstimationBase.QFIM_RLD(ρ, ∂ρ)
        @test F_r ≥ F_s - 1e-10
    end
end
