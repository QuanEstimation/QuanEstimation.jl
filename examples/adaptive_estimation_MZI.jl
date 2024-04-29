using QuanEstimation
using SparseArrays

# the number of photons
N = 8
# probe state
psi = sum([sin(k*pi/(N+2))*kron(QuanEstimation.basis(N+1,k), 
      QuanEstimation.basis(N+1, N-k+2)) for k in 1:(N+1)]) |> sparse
psi = psi*sqrt(2/(2+N))
rho0 = psi*psi'
# prior distribution
x = range(-pi, pi, length=100)
p = (1.0/(x[end]-x[1]))*ones(length(x))
apt = Adapt_MZI(x, p, rho0)

#================online strategy=========================#
online(apt, target=:sharpness, output="phi")

#================offline strategy=========================#
# # algorithm: DE
# alg = DE(p_num=10, ini_population=missing, 
#                         max_episode=1000, c=1.0, cr=0.5)
# offline(apt, alg, target=:sharpness, seed=1234)

# # algorithm: PSO
# PSO(p_num=10, ini_particle=missing,  
#                          max_episode=[1000,100], c0=1.0, 
#                          c1=2.0, c2=2.0)
# offline(apt, alg, target=:sharpness, seed=1234)