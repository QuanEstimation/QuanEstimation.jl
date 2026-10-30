using Test
using LinearAlgebra
using QuanEstimation

@testset "Documentation: dynamics and resource examples" begin
    rho0 = ComplexF64[0.5 0.5; 0.5 0.5]
    sz = ComplexF64[1 0; 0 -1]
    H0 = sz / 2
    dH = [sz / 2]
    tspan = collect(range(0.0, 0.3; length=4))
    decay = [[ComplexF64[0 1; 0 0], 0.1]]

    rho, drho = expm(tspan, rho0, H0, dH; decay=decay)
    @test length(rho) == length(tspan)
    @test length(drho) == length(tspan)
    @test real(tr(rho[end])) ≈ 1.0 rtol=1e-12

    gamma = 0.1
    K = [ComplexF64[1 0; 0 sqrt(1-gamma)], ComplexF64[0 sqrt(gamma); 0 0]]
    dK = [[ComplexF64[0 0; 0 -0.5/sqrt(1-gamma)]],
          [ComplexF64[0 0.5/sqrt(gamma); 0 0]]]
    scheme = GeneralScheme(; probe=rho0, param=Kraus(K, dK))
    rho_k, drho_k = evolve(scheme)
    @test real(tr(rho_k)) ≈ 1.0 rtol=1e-12
    @test length(drho_k) == 1
    @test all(isfinite, drho_k[1])

    objective(t, scale) = scale * t^2
    target = 1.0
    expected = sqrt(target / 2.0)
    @test TargetTime(target, collect(range(0.0, 2.0; length=201)), objective, 2.0) ≈ expected atol=1e-4
end

@testset "Documentation: public bindings have API docs" begin
    documented = (
        :BCFIM, :BQFIM, :OBB, :BCB, :BayesCost, :BayesInput,
        :BellState, :Copt, :FI_Expt, :Htot, :MinusState, :Mopt,
        :PlusState, :SigmaX, :SigmaY, :SigmaZ, :Sopt, :basis,
        :expm_py, :liouville_commu_py, :liouville_dissip_py, :ode_py,
        :σx, :σy, :σz, :helstrom_bound, :bound_LC_coeff!,
    )
    for name in documented
        @test Base.Docs._doc(Base.Docs.Binding(QuanEstimationBase, name)) !== nothing
    end
end
