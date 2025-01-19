using QuanEstimationBase
using Random, LinearAlgebra

function test_Lindblad_qubit_expm()
    (; tspan, rho0, H0, dH, Hc, ctrl, decay) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl = ctrl, dyn_method = :Expm)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    rho, drho = evolve(scheme)
    @test isposdef(rho) && ishermitian(rho)
    @test length(drho) == length(dH)
end

function test_Lindblad_qubit_ode()
    (; tspan, rho0, H0, dH, Hc, ctrl, decay) = generate_qubit_dynamics()

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl = ctrl, dyn_method = :Ode)
    scheme = GeneralScheme(; probe = rho0, param = dynamics)

    rho, drho = evolve(scheme)
    @test isposdef(rho) && ishermitian(rho)
    @test length(drho) == length(dH)
end

# Run the tests
function test_Lindblad()
    test_Lindblad_qubit_expm()
    test_Lindblad_qubit_ode()
end

test_Lindblad()
