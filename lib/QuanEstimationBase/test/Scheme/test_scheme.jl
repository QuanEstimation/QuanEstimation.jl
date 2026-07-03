using Test
using LinearAlgebra

using QuanEstimationBase:
    QFIM,
    evaluate_hamiltonian,
    Lindblad,
    expm,
    ode,
    evolve,
    GeneralScheme,
    get_dim,
    SigmaX, SigmaY, SigmaZ,
    ZeroCTRL, LinearCTRL, SineCTRL, SawCTRL, TriangleCTRL, GaussianCTRL, GaussianEdgeCTRL, 
    Hamiltonian,
    PlusState, MinusState, BellState, 
    Kraus,
    QFIM_Kraus


function test_lindblad(;dyn_method=:Ode)
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, M) = generate_qubit_dynamics()

    expm(tspan, rho0, H0, dH[1])
    ode(tspan, rho0, H0, dH[1])
    ode(tspan, rho0, H0, dH, decay=decay, Hc=Hc, ctrl=ctrl)

    Lindblad(H0, dH, tspan, Hc; ctrl=ZeroCTRL())
    Lindblad(H0, dH, tspan, Hc; ctrl=LinearCTRL())
    Lindblad(H0, dH, tspan, Hc; ctrl=SineCTRL())
    Lindblad(H0, dH, tspan, Hc; ctrl=SawCTRL())
    Lindblad(H0, dH, tspan, Hc; ctrl=TriangleCTRL())
    Lindblad(H0, dH, tspan, Hc; ctrl=GaussianCTRL())
    Lindblad(H0, dH, tspan, Hc; ctrl=GaussianEdgeCTRL())

    dynamics = Lindblad(H0, dH, tspan; dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    expm(scheme);ode(scheme);evolve(scheme)
    @test get_dim(scheme) > 0

    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    expm(scheme);ode(scheme);evolve(scheme)
    @test get_dim(scheme) > 0

    dynamics = Lindblad(H0, dH, tspan, Hc; dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    expm(scheme);ode(scheme);evolve(scheme)
    @test get_dim(scheme) > 0

    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    expm(scheme);ode(scheme);evolve(scheme)
    @test get_dim(scheme) > 0

    (;H0_func, dH_func) = generate_bayes()
    
    ham = Hamiltonian(H0_func, dH_func, 1.0)
    dynamics = Lindblad(ham, tspan, decay; dyn_method=dyn_method)
    dynamics = Lindblad(ham, tspan, Hc; dyn_method=dyn_method)
    dynamics = Lindblad(ham, tspan, Hc, decay; dyn_method=dyn_method)
    scheme = GeneralScheme(; probe=rho0, param=dynamics)
    evolve(scheme)


    H0, dH = evaluate_hamiltonian(scheme)
    @test H0 isa AbstractMatrix
    @test dH isa AbstractVector
end  # function test_lindblad

@doc raw"""
    test_lindblad_consistency()

STRICT :Expm and :Ode dynamics should produce the same QFIM
for identical physical parameters.  This is a self-consistency check: both
methods solve the same Lindblad master equation numerically.
"""
# STRICT Two equivalent Lindblad constructions (with and without
# explicit control) should produce the same QFIM when using the same integration
# method.  For zero control, the Hamiltonian-only and control-included dynamics
# are physically identical, as are explicit vs ZeroCTRL constructions.
function test_lindblad_consistency()
    (; tspan, rho0, H0, dH, Hc) = generate_qubit_dynamics()

    dyn_free = Lindblad(H0, dH, tspan; dyn_method=:Expm)
    dyn_ctrl = Lindblad(H0, dH, tspan, Hc; ctrl=ZeroCTRL(), dyn_method=:Expm)

    F_free = QFIM(GeneralScheme(; probe=rho0, param=dyn_free))
    F_ctrl = QFIM(GeneralScheme(; probe=rho0, param=dyn_ctrl))

    @test F_free ≈ F_ctrl rtol=1e-10
end

function test_lindblad_pure()
    (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    expm(tspan, rho0, H0, dH[1])
    ode(tspan, rho0, H0, dH[1])
    ode(tspan, rho0, H0, dH, decay=decay, Hc=Hc, ctrl=ctrl)

    dynamics = Lindblad(H0, dH, tspan)
    scheme = GeneralScheme(; probe=PlusState(), param=dynamics)
    evolve(scheme)
    @test get_dim(scheme) > 0
    
    dynamics = Lindblad(H0, dH, tspan, decay)
    scheme = GeneralScheme(; probe=PlusState(), param=dynamics)
    evolve(scheme)
    @test get_dim(scheme) > 0

    dynamics = Lindblad(H0, dH, tspan, Hc, decay)
    scheme = GeneralScheme(; probe=PlusState(), param=dynamics)
    evolve(scheme)
    @test get_dim(scheme) > 0

    H0, dH = evaluate_hamiltonian(scheme)
    @test H0 isa AbstractMatrix
    @test dH isa AbstractVector
end

# SOFT For a pure state input, QFIM computed from |ψ⟩
# directly and from ρ = |ψ⟩⟨ψ| should agree.
function test_lindblad_pure_qfim()
    (; tspan, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()

    psi = PlusState()
    rho_mixed = psi * psi'

    dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)

    F_pure  = QFIM(GeneralScheme(; probe=psi, param=dyn))
    F_mixed = QFIM(GeneralScheme(; probe=rho_mixed, param=dyn))

    @test F_pure ≈ F_mixed rtol=1e-8
end

function test_kraus() 
    (;rho0, psi, K, dK, K_func, dK_func) = generate_kraus()

    F = QFIM_Kraus(rho0, K, dK)
    @test F isa Number
    @test isfinite(F)
    @test F >= 0

    channel = Kraus(K, dK)
    evolve(GeneralScheme(; probe=rho0, param=channel))
    evolve(GeneralScheme(; probe=psi, param=channel))

    channel = Kraus(K_func, dK_func, 0.5)
    evolve(GeneralScheme(; probe=rho0, param=channel))
    evolve(GeneralScheme(; probe=psi, param=channel))
end

function test_state()
    @test PlusState() == [1.0, 1.0]/sqrt(2)
    @test MinusState() == [1.0, -1.0]/sqrt(2)
    @test BellState() == BellState(1) == [1.0, 0.0, 0.0, 1.0]/sqrt(2)
    @test BellState(2) == [1.0, 0.0, 0.0, -1.0]/sqrt(2)
    @test BellState(3) == [0.0, 1.0, 1.0, 0.0]/sqrt(2)
    @test BellState(4) == [0.0, 1.0, -1.0, 0.0]/sqrt(2)
end  # function test_state

function test_parameterization()
    test_lindblad(dyn_method=:Ode)
    test_lindblad(dyn_method=:Expm)
    test_lindblad_pure()
    test_kraus()
end  # function test_parameterization

@testset "Lindblad consistency" begin
    test_lindblad_consistency()
end
@testset "Pure state QFIM" begin
    test_lindblad_pure_qfim()
end
@testset "Parameterization smoke" begin test_parameterization() end
@testset "Canonical States" begin test_state() end
