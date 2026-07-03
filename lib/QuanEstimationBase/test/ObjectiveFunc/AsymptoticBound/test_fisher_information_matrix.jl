using QuanEstimationBase
using Test
using LinearAlgebra

# initial state
rho0 = 0.5 * ones(2, 2)
# free Hamiltonian
omega = 1.0
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -im; im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]
θ = pi / 3
H0 = 0.5 * omega * sz * cos(θ)
# derivative of the free Hamiltonian on omega
dH = [0.5 * sz * cos(θ), -0.5 * omega * sz * sin(θ)]
# dissipation
sp = [0.0 1.0; 0.0 0.0im]
sm = [0.0 0.0; 1.0 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# measurement
M1 = 0.5 * [1.0+0.0im 1.0; 1.0 1.0]
M2 = 0.5 * [1.0+0.0im -1.0; -1.0 1.0]
M = [M1, M2]
# time length for the evolution
tspan = range(0.0, 50.0, length = 200)
# dynamics
rho, drho = expm(tspan, rho0, H0, dH;decay=decay)
# calculation of the CFI and QFI
Im, F = Matrix{Float64}[], Matrix{Float64}[]
for ti in eachindex(tspan)
    # CFI
    I_tp = CFIM(rho[ti], drho[ti], M)
    append!(Im, [I_tp])
    # QFI
    F_tp = QFIM(rho[ti], drho[ti])
    append!(F, [F_tp])
end


@testset "CFIM/QFIM trajectory properties" begin
@test tr(Im[1]) ≈ 0
@test tr(F[1]) ≈ 0
# positivity
@test all(x -> real(tr(x)) >= (0), Im[1])
@test all(x -> real(tr(x)) >= (0), F[1])

# Properties of the QFIM & CFIM
# real symmetric
@test issymmetric(F[end])
@test issymmetric(Im[end])

# positive semi-definite
@test F[end] |> x -> round.(x; digits = 5) |> isposdef
@test Im[end] |> x -> round.(x; digits = 5) |> isposdef

# parameter-independent unitary operation invariant
U = exp(im * pi / 8 * sx)
ρ_0 = rho[end]
ρ_1 = U * ρ_0 * U'
@test QFIM(ρ_1, [U * dr * U' for dr in drho[end]]) ≈ F[end]

# convexity
p = 0.3
@test (QFIM(
    p * ρ_0 + (1 - p) * ρ_1,
    [p * dr + (1 - p) * U * dr * U' for dr in drho[end]],
)) |> x -> tr(x) >= 0
end

# ============ Plan-2 bug-specific tests ============

# Bug #38: Hermiticity enforcement
@testset "#38 Hermiticity enforcement" begin
    for _ in 1:10
        N = rand(2:4)
        ρ = rand_ρ(N)
        ∂ρ = rand_∂ρ(N)
        A = randn(ComplexF64, N, N)
        A_anti = A - A'
        ε = rand() * 1e-12
        ρ_pert = ρ + ε * A_anti
        F = QFIM(ρ_pert, [∂ρ]; LDtype=:SLD)
        @test abs(imag(F[1,1])) < 1e-12
    end
end

# Bug #39: SLD Hermiticity
@testset "#39 SLD Hermiticity" begin
    for _ in 1:10
        N = rand(2:4)
        ρ = rand_ρ(N)
        ∂ρ = rand_∂ρ(N)
        L_orig = SLD(ρ, ∂ρ; rep="original")
        L_eig = SLD(ρ, ∂ρ; rep="eigen")
        @test norm(L_orig - L_orig') < 5e-13
        @test norm(L_eig - L_eig') < 5e-13
    end
end

# Bug #8: QFIM multi-param correct dimensions
@testset "#8 Multi-param QFIM dimensions" begin
    N = 2
    ρ = rand_ρ(N)
    dρ = [rand_∂ρ(N), rand_∂ρ(N), rand_∂ρ(N)]
    for LDtype in [:SLD, :RLD, :LLD]
        F = QFIM(ρ, dρ; LDtype=LDtype)
        @test size(F) == (3, 3)
    end
    ρ_pure = rand_ρ(N)
    dρ_pure = [rand_∂ρ(N), rand_∂ρ(N)]
    Fp = QuanEstimationBase.QFIM_pure(ρ_pure, dρ_pure)
    @test size(Fp) == (2, 2)
end

# Bug #10: Williamson eigenvalue sorting
@testset "#10 Williamson eigenvalue sorting" begin
    c_single = QuanEstimationBase.Williamson_form(Matrix{Float64}(I, 2, 2) * 3.0)[2]
    @test length(c_single) == 1
    @test isapprox(c_single[1], 3.0, rtol=1e-10)
end

# Bug #56: Near-zero eigenvalue truncation
@testset "#56 Near-zero eigenvalue truncation" begin
    rho_sing = ComplexF64[1.0 0.0; 0.0 1e-16]
    rho_sing = (rho_sing + rho_sing') / 2
    rho_sing ./= tr(rho_sing)
    drho_sing = ComplexF64[0.0 0.0; 0.0 0.0]
    F = QFIM(rho_sing, [drho_sing]; LDtype=:SLD)
    @test all(isfinite, F)
    @test !any(isnan, F)
end
