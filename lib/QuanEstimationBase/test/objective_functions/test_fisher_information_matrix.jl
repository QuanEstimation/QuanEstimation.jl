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
rho, drho = expm(tspan, rho0, H0, dH; decay = decay)
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
