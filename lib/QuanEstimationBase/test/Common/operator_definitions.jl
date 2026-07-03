using QuanEstimationBase
using Test
using LinearAlgebra

@testset "Operator Definitions — Random Matrices N=2:5" begin
    for N in 2:5, _ in 1:20
        ρ = rand_ρ(N)
        ∂ρ = rand_∂ρ(N)

        # --- SLD: 2∂ρ = Lρ + ρL ---
        # Hermiticity check for both reps
        for rep in ("original", "eigen")
            L = SLD(ρ, ∂ρ; rep=rep)
            @test norm(L - L') < 1e-11
        end
        # Definition check in original basis only
        L_orig = SLD(ρ, ∂ρ; rep="original")
        @test norm(2 * ∂ρ - L_orig * ρ - ρ * L_orig) < 1e-10

        # --- RLD: ∂ρ = ρR (original basis only) ---
        R_orig = RLD(ρ, ∂ρ; rep="original")
        R_eig = RLD(ρ, ∂ρ; rep="eigen")
        @test norm(∂ρ - ρ * R_orig) < 1e-10

        # --- LLD: ∂ρ = Lρ (original basis only, L = R†) ---
        L_orig = LLD(ρ, ∂ρ; rep="original")
        L_eig = LLD(ρ, ∂ρ; rep="eigen")
        @test norm(∂ρ - L_orig * ρ) < 1e-10

        # --- QFI cross-consistency (F_RLD ≈ F_LLD since LLD = RLD') ---
        F_RLD = QFIM(ρ, ∂ρ; LDtype=:RLD)
        F_LLD = QFIM(ρ, ∂ρ; LDtype=:LLD)
        @test isapprox(F_RLD, F_LLD, rtol=1e-10)
    end
end
