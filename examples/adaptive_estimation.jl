using Revise
using QuanEstimationBase
using Random
using StatsBase

B, omega0 = pi/2.0, 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]

# free Hamiltonian
H0_func(x) = 0.5*(sx*cos(x[1])+sz*sin(x[1]))
# derivative of free Hamiltonian in x
dH_func(x) = [0.5*(-sx*sin(x[1])+sz*cos(x[1]))]

# initial state
rho0 = 0.5*ones(2, 2)
# measurement 
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., stop=1., length=1000) |> Vector
# prior distribution
x = range(-pi/4+0.1, stop=3.0*pi/4.0-0.1, length=100) |> Vector
p = (1.0/(x[end]-x[1]))*ones(length(x))
# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
for i = 1:length(x) 
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = expm(tspan, rho0, H0_tp, dH_tp)
    rho[i] = rho_tp[end]
end
# pre-estimation
Random.seed!(1234)
y = [rand() > 0.7 ? 0 : 1 for _ in 1:500] 
pout, xout = Bayes([x], p, rho, y, M=M, savefile=false)

dynamics = Lindblad(Hamiltonian(H0_func, dH_func), tspan; dyn_method=:Expm)
strategy = AdaptiveStrategy([x], pout)
scheme = GeneralScheme(;probe=rho0,param=dynamics,measurement=M, strat=strategy)
# adaptive measurement
adapt!(scheme; max_episode=100, method="FOP")