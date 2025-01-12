using QuanEstimation, Random, LinearAlgebra

# initial state
rho0 = 0.5*ones(2, 2)
# free Hamiltonian
omega = 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
H0 = 0.5*omega*sz
# derivative of the free Hamiltonian on omega
dH = [0.5*sz]
# dissipation
sp = [0. 1.; 0. 0.0im]
sm = [0. 0.; 1. 0.0im]
decay = [[sp, 0.], [sm, 0.1]]
# generation of a set of POVM basis
dim = size(rho0, 1)
POVM_basis = SIC(dim)
# time length for the evolution
tspan = range(0., 10., length=2500)

##==========choose measurement optimization algorithm==========##
##-------------algorithm: DE---------------------##
alg = DE(p_num=10, ini_population=nothing, max_episode=1000, c=1.0, cr=0.5)

##-------------algorithm: PSO---------------------##
# alg = PSO(p_num=10, ini_particle=nothing, max_episode=[1000,100], c0=1.0, c1=2.0, c2=2.0)

##-------------algorithm: AD---------------------##
# alg = AD( max_episode=300, epsilon=0.01, beta1=0.90, beta2=0.99)

# input the dynamics data
dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dynamics,)
# objective function: CFI
obj = CFIM_obj()

# find the optimal linear combination of an input measurement
opt = MeasurementOpt(mtype=:LC, POVM_basis=POVM_basis, M_num=2, seed=1234)

optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
