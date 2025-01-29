using QuanEstimation, Random, LinearAlgebra
# initial state
rho0 = zeros(ComplexF64, 6, 6)
rho0[1:4:5, 1:4:5] .= 0.5
# Hamiltonian
sx = [0.0 1.0; 1.0 0.0]
sy = [0.0 -im; im 0.0]
sz = [1.0 0.0; 0.0 -1.0]
s1 = [0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0] / sqrt(2)
s2 = [0.0 -im 0.0; im 0.0 -im; 0.0 im 0.0] / sqrt(2)
s3 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0]
Is = I1, I2, I3 = [kron(I(3), sx), kron(I(3), sy), kron(I(3), sz)]
S = S1, S2, S3 = [kron(s1, I(2)), kron(s2, I(2)), kron(s3, I(2))]
B = B1, B2, B3 = [5.0e-4, 5.0e-4, 5.0e-4]
# All numbers are divided by 100 in this example 
# for better calculation accurancy
cons = 100
D = (2pi * 2.87 * 1000) / cons
gS = (2pi * 28.03 * 1000) / cons
gI = (2pi * 4.32) / cons
A1 = (2pi * 3.65) / cons
A2 = (2pi * 3.03) / cons
H0 = sum([
    D * kron(s3^2, I(2)),
    sum(gS * B .* S),
    sum(gI * B .* Is),
    A1 * (kron(s1, sx) + kron(s2, sy)),
    A2 * kron(s3, sz),
])
# derivatives of the free Hamiltonian on B1, B2 and B3
dH = gS * S + gI * Is
# control Hamiltonians 
Hc = [S1, S2, S3]
# dissipation
decay = [[S3, 2pi / cons]]
# generation of a set of POVM basis
dim = size(rho0, 1)
POVM_basis = [basis(dim, i) * basis(dim, i)' for i = 1:dim]
# time length for the evolution
tspan = range(0.0, 2.0, length = 4)

##==========choose measurement optimization algorithm==========##
##-------------algorithm: DE---------------------##
alg = DE(p_num = 10, ini_population = nothing, max_episode = 1000, c = 1.0, cr = 0.5)

##-------------algorithm: PSO---------------------##
# alg = PSO(p_num=10, ini_particle=nothing, max_episode=[1000,100], c0=1.0, c1=2.0, c2=2.0)

# input the dynamics data
dynamics = Lindblad(H0, dH, tspan, Hc, decay; dyn_method = :Expm)
scheme = GeneralScheme(; probe = rho0, param = dynamics)
# objective function: CFI
obj = CFIM_obj()

opt = SCMopt(ctrl_bound = [-0.2, 0.2], seed = 1234)

optimize!(scheme, opt; algorithm = alg, objective = obj, savefile = false)
