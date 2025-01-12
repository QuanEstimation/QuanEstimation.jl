using SparseArrays

# the number of photons
N = 3

# probe state
psi = sum([
      sin(k*pi/(N+2)) * kron(
            QuanEstimationBase.basis(N+1,k), 
            QuanEstimationBase.basis(N+1, N-k+2)
            ) for k in 1:(N+1)
      ]) |> sparse
psi = psi*sqrt(2/(2+N))
rho0 = psi*psi'

# prior distribution
x = range(-pi, pi, length=5)
p = (1.0/(x[end]-x[1])) * ones(length(x))
apt = QuanEstimationBase.Adapt_MZI(x, p, rho0)

#================online strategy=========================#
# QuanEstimationBase.online(apt, target=:sharpness, output="phi")

#================offline strategy=========================#
# algorithm: DE
alg = QuanEstimationBase.DE(
      p_num=10, 
      ini_population=nothing, 
      max_episode=10, 
      c=1.0, 
      cr=0.5
      )

QuanEstimationBase.offline(
      apt, 
      alg;
      target=:sharpness, 
      seed=1234
      )

# algorithm: PSO
alg = QuanEstimationBase.PSO(;
      p_num=10, 
      ini_particle=nothing,  
      max_episode=[10,10], 
      c0=1.0, 
      c1=2.0, 
      c2=2.0)

QuanEstimationBase.offline(
      apt, 
      alg;
      target=:sharpness, 
      seed=1234
      )