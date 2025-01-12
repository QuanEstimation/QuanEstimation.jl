using QuanEstimation
using Random

# initial state
rho0 = 0.5 * ones(2, 2)
# free Hamiltonian
omega = 1.0
sx = [0.0 1.0; 1.0 0.0im]
sy = [0.0 -im; im 0.0]
sz = [1.0 0.0im; 0.0 -1.0]
H0 = 0.5 * omega * sz
# derivative of the free Hamiltonian on omega
dH = [0.5 * sz]
# control Hamiltonians 
Hc = [sx, sy, sz]
# dissipation
sp = [0.0 1.0; 0.0 0.0im]
sm = [0.0 0.0; 1.0 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# measurement
M1 = 0.5 * [1.0+0.0im 1.0; 1.0 1.0]
M2 = 0.5 * [1.0+0.0im -1.0; -1.0 1.0]
M = [M1, M2]
# time length for the evolution
tspan = range(0.0, 10.0, length=2500)
# guessed control coefficients
cnum = length(tspan) - 1
ctrl = [zeros(cnum) for _ in eachindex(Hc)]
ctrl_bound = [-2.0, 2.0]

##==========choose measurement optimization algorithm==========##
##-------------algorithm: auto-GRAPE---------------------##
alg = autoGRAPE(Adam=true, max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99)

##-------------algorithm: GRAPE---------------------##
# alg = GRAPE(Adam=true, max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99)


##-------------algorithm: PSO---------------------##
# alg = PSO(p_num=10, ini_particle=([ctrl],), max_episode=[1000, 100], c0=1.0, c1=2.0, c2=2.0)

##-------------algorithm: DE---------------------##
# alg = DE(p_num=10, ini_population=([ctrl],), 
#                         max_episode=1000, c=1.0, cr=0.5)

##-------------algorithm: DDPG---------------------##
# alg = DDPG(max_episode=500, layer_num=4, layer_dim=220)

##===================choose objective function===================##
##-------------objective function: QFI---------------------##
# objective function: QFI
obj = QFIM_obj()
# input the dynamics data
dynamics = Lindblad(H0, dH, tspan, Hc, decay; dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dynamics,)

# choose the optimization type
opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)

# run the control optimization problem
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=true)

##-------------objective function: CFI---------------------##
# # objective function: CFI
# obj = CFIM_obj(M=M)
# # input the dynamics data
# dynamics = Lindblad(H0, dH, tspan, Hc, decay; dyn_method=:Expm)
# # run the control optimization problem
# optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=true)


