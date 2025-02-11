using Random
# free Hamiltonian
function H0_func(x)
    return 0.5 * B * omega0 * (sx * cos(x[1]) + sz * sin(x[1]))
end
# derivative of free Hamiltonian in x
function dH_func(x)
    return [0.5 * B * omega0 * (-sx * sin(x[1]) + sz * cos(x[1]))]
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
# time length for the evolution
tspan = range(0.0, stop = 1.0, length = 100) |> Vector
# prior distribution
x = range(-0.25 * pi + 0.1, stop = 3.0 * pi / 4.0 - 0.1, length = 10) |> Vector
p = (1.0 / (x[end] - x[1])) * ones(length(x))
# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
for i in eachindex(x)
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = QuanEstimationBase.expm(tspan, rho0, H0_tp, dH_tp)
    rho[i] = rho_tp[end]
end
# Bayesian estimation
Random.seed!(1234)
y = [rand() >= 0.5 ? 0 : 1 for _ = 1:500]
pout, xout = QuanEstimationBase.Bayes([x], p, rho, y, M = M, savefile = false)
# generation of H and dH
H, dH = QuanEstimationBase.BayesInput([x], H0_func, dH_func; channel = "dynamics")
# adaptive measurement
QuanEstimationBase.Adapt(
    [x],
    pout,
    rho0,
    tspan,
    H,
    dH;
    M = M,
    max_episode = 100,
    dyn_method = :Expm,
    method = "FOP",
)
