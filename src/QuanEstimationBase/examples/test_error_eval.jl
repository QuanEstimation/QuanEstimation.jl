using Revise
using QuanEstimation

# initial state
rho0 = 0.5*ones(ComplexF64, 2, 2)
# free Hamiltonian
omega = 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
H0 = 0.5*omega*sz
# derivative of the free Hamiltonian on omega
dH = [0.5*sz]
# control Hamiltonians 
Hc = [sx, sy, sz]
# dissipation
sp = [0. 1.; 0. 0.0im]
sm = [0. 0.; 1. 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# measurement
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., 10., length=2500)

dynamics = Lindblad(H0, dH, tspan, Hc, decay, dyn_method=:Expm)
scheme = GeneralScheme(;probe=rho0,param=dynamics,measurement=M)

error_evaluation(scheme;input_error_scaling=1e-8, eps_scaling=1e-8)
error_evaluation(scheme;input_error_scaling=1e-6, eps_scaling=1e-8)

dynamics = Lindblad(H0, dH, tspan, Hc, decay, dyn_method=:Ode)
scheme = GeneralScheme(;probe=rho0,param=dynamics,measurement=M)

error_evaluation(scheme;input_error_scaling=1e-8, eps_scaling=1e-8)

