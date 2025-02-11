using Random
using StatsBase

# free Hamiltonian
function H0_func(x)
    return 0.5 * B * omega0 * (sx * cos(x) + sz * sin(x))
end

# derivative of the free Hamiltonian on x
function dH_func(x)
    return [0.5 * B * omega0 * (-sx * sin(x) + sz * cos(x))]
end

B, omega0 = pi / 2.0, 1.0
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -im; im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]

# initial state
rho0 = 0.5 * ones(2, 2)

# measurement 
M1 = 0.5 * [1.0+0.0im 1.0; 1.0 1.0]
M2 = 0.5 * [1.0+0.0im -1.0; -1.0 1.0]
M = [M1, M2]

# prior distribution
x = range(0.0, stop = 0.5 * pi, length = 5) |> Vector
p = (1.0 / (x[end] - x[1])) * ones(length(x))

# time length for the evolution
tspan = range(0.0, stop = 1.0, length = 5)

# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
for i in eachindex(x)
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = QuanEstimationBase.expm(tspan, rho0, H0_tp, dH_tp)
    rho[i] = rho_tp[end]
end

# Generation of the experimental results
Random.seed!(1234)
y = [rand() >= 0.5 ? 0 : 1 for _ = 1:5]

#===============Maximum a posteriori estimation===============#
pout, xout =
    QuanEstimationBase.Bayes([x], p, rho, y; M = M, estimator = "MAP", savefile = false)

#===============Maximum likelihood estimation===============#
Lout, xout = QuanEstimationBase.MLE([x], rho, y; M = M, savefile = false)
