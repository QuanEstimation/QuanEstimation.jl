# initial state
rho0 = 0.5 * ones(2, 2)
# Kraus operators for the amplitude damping channel
gamma = 0.1
K1 = [1.0 0.0; 0.0 sqrt(1 - gamma)]
K2 = [0.0 sqrt(gamma); 0.0 0.0]
K = [K1, K2]
# derivatives of Kraus operators on gamma
dK1 = [1.0 0.0; 0.0 -0.5/sqrt(1 - gamma)]
dK2 = [0.0 0.5/sqrt(gamma); 0.0 0.0]
dK = [[dK1], [dK2]]
# parameterization process
Kraus = QuanEstimationBase.Kraus(rho0, K, dK)
rho, drho = QuanEstimationBase.evolve(Kraus)

@test isposdef(rho)
