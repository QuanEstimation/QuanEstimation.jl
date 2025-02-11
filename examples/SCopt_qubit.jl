using QuanEstimation

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
tspan = range(0.0, 10.0, length = 2500)

##==========choose comprehensive optimization algorithm==========##
##-------------algorithm: DE---------------------##
alg = DE(p_num = 10, max_episode = 1000, c = 1.0, cr = 0.5)

##-------------algorithm: PSO---------------------##
# alg = PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
#                          c1=2.0, c2=2.0)

##-------------algorithm: AD---------------------##
# alg = AD(Adam=true, max_episode=1000, epsilon=0.01, 
#                         beta1=0.90, beta2=0.99)

##===================choose objective function===================##
##-------------objective function: tr(WF^{-1})---------------------##
obj = QFIM_obj()

##-------------objective function: tr(WI^{-1})---------------------##
# obj = CFIM_obj(M=M)

dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl = ZeroCTRL(), dyn_method = :Expm)
scheme = GeneralScheme(; probe = rho0, param = dynamics)

opt = SCopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = false)
