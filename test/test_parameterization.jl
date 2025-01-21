using QuanEstimationBase:evaluate_hamiltonian

function test_lindblad(;dyn_method=:Ode)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, M) = generate_qubit_dynamics()

    expm(tspan, rho0, H0, dH[1])
    ode(tspan, rho0, H0, dH[1])
    ode(tspan, rho0, H0, dH, decay=decay, Hc=Hc, ctrl=ctrl)

    dynamics = Lindblad(H0, dH, tspan; dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    expm(scheme);ode(scheme);evolve(scheme)

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    expm(scheme);ode(scheme);evolve(scheme)

    dynamics = Lindblad(H0, dH, tspan, Hc; dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    expm(scheme);ode(scheme);evolve(scheme)

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    expm(scheme);ode(scheme);evolve(scheme)

    evaluate_hamiltonian(scheme)
end  # function test_lindblad

function test_lindblad_pure()
    (; tspan, H0, dH, Hc, decay, ctrl, M) = generate_qubit_dynamics()

    expm(tspan, rho0, H0, dH[1])
    ode(tspan, rho0, H0, dH[1])
    ode(tspan, rho0, H0, dH, decay=decay, Hc=Hc, ctrl=ctrl)

    dynamics = Lindblad(H0, dH, tspan)
    scheme = GeneralScheme(; probe=PlusState(), param=dynamics)
    evolve(scheme)

    dynamics = Lindblad(H0, dH, tspan, decay)
    scheme = GeneralScheme(; probe=PlusState(), param=dynamics)
    evolve(scheme)

    dynamics = Lindblad(H0, dH, tspan, Hc, decay)
    scheme = GeneralScheme(; probe=PlusState(), param=dynamics)
    evolve(scheme)

    evaluate_hamiltonian(scheme)
end

function test_parameterization()
    test_lindblad(dyn_method=:Ode)
    test_lindblad(dyn_method=:Expm)
    test_lindblad_pure()
end  # function test_parameterization

test_parameterization()