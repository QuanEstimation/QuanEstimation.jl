# API compatibility signature tests for Julia-documented functions.
# Run with: julia --project -e 'using Pkg; Pkg.test(; test_args=["api_compat"])'
# or: julia --project test/api_compat/run_compat.jl

using Test
using LinearAlgebra
using QuanEstimation

@testset "api_compat: paper-documented functions exist" begin
    # Scheme setup
    @test isdefined(QuanEstimation, :GeneralScheme)
    @test isdefined(QuanEstimation, :Lindblad)
    @test isdefined(QuanEstimation, :Kraus)
    @test isdefined(QuanEstimation, :QubitDephasing)
    @test isdefined(QuanEstimation, :Hamiltonian)

    # Metrological bounds
    @test isdefined(QuanEstimation, :QFIM)
    @test isdefined(QuanEstimation, :CFIM)
    @test isdefined(QuanEstimation, :HCRB)
    @test isdefined(QuanEstimation, :NHB)
    @test isdefined(QuanEstimation, :SLD)
    @test isdefined(QuanEstimation, :RLD)
    @test isdefined(QuanEstimation, :LLD)
    @test isdefined(QuanEstimation, :FIM)
    @test isdefined(QuanEstimation, :FI_Expt)
    @test isdefined(QuanEstimation, :QFIM_Kraus)
    @test isdefined(QuanEstimation, :QFIM_Bloch)
    @test isdefined(QuanEstimation, :QFIM_Gauss)

    # Bayesian bounds (BCRB, BQCRB exported; BCFIM, BQFIM, OBB are internal)
    @test isdefined(QuanEstimation, :BCRB)
    @test isdefined(QuanEstimation, :BQCRB)
    @test isdefined(QuanEstimation, :QZZB)
    @test isdefined(QuanEstimation, :VTB)
    @test isdefined(QuanEstimation, :QVTB)

    # Bayesian estimation
    @test isdefined(QuanEstimation, :Bayes)
    @test isdefined(QuanEstimation, :MLE)

    # Optimization scenarios (exported: ControlOpt, StateOpt, MeasurementOpt, Copt, Mopt, Sopt)
    @test isdefined(QuanEstimation, Symbol("optimize!"))
    @test isdefined(QuanEstimation, :ControlOpt)
    @test isdefined(QuanEstimation, :StateOpt)
    @test isdefined(QuanEstimation, :MeasurementOpt)
    @test isdefined(QuanEstimation, :Copt)
    @test isdefined(QuanEstimation, :Mopt)
    @test isdefined(QuanEstimation, :Sopt)
    @test isdefined(QuanEstimation, :CMopt)
    @test isdefined(QuanEstimation, :SCopt)
    @test isdefined(QuanEstimation, :SMopt)
    @test isdefined(QuanEstimation, :SCMopt)

    # Adaptive
    @test isdefined(QuanEstimation, :AdaptiveStrategy)
    @test isdefined(QuanEstimation, Symbol("adapt!"))
    @test isdefined(QuanEstimation, :Adapt_MZI)
    @test isdefined(QuanEstimation, :online)
    @test isdefined(QuanEstimation, :offline)
    @test isdefined(QuanEstimation, :DE_deltaphiOpt)
    @test isdefined(QuanEstimation, :PSO_deltaphiOpt)

    # Error evaluation
    @test isdefined(QuanEstimation, :error_evaluation)
    @test isdefined(QuanEstimation, :error_control)

    # Resources
    @test isdefined(QuanEstimation, :SpinSqueezing)
    @test isdefined(QuanEstimation, :TargetTime)

    # States
    @test isdefined(QuanEstimation, :PlusState)
    @test isdefined(QuanEstimation, :MinusState)
    @test isdefined(QuanEstimation, :BellState)
    @test isdefined(QuanEstimation, :SigmaX)
    @test isdefined(QuanEstimation, :SigmaY)
    @test isdefined(QuanEstimation, :SigmaZ)

    # Control waveforms
    @test isdefined(QuanEstimation, :ZeroCTRL)
    @test isdefined(QuanEstimation, :LinearCTRL)
    @test isdefined(QuanEstimation, :SineCTRL)
    @test isdefined(QuanEstimation, :SawCTRL)
    @test isdefined(QuanEstimation, :TriangleCTRL)
    @test isdefined(QuanEstimation, :GaussianCTRL)
    @test isdefined(QuanEstimation, :GaussianEdgeCTRL)

    # SIC / basis
    @test isdefined(QuanEstimation, :SIC)
    @test isdefined(QuanEstimation, :basis)
end

@testset "api_compat: keyword argument acceptance" begin
    sp = ComplexF64[0 1; 0 0]
    sm = ComplexF64[0 0; 1 0]
    sz = ComplexF64[1 0; 0 -1]
    H0 = sz / 2
    dH = [sz / 2]
    tspan = 0:0.1:1
    decay = [[sp, 0.0], [sm, 0.1]]
    M = [[1 0; 0 0], [0 0; 0 1]]
    rho0 = 0.5 * ones(ComplexF64, 2, 2)

    dynamics = Lindblad(H0, dH, tspan, decay)
    scheme = GeneralScheme(; probe=rho0, param=dynamics, measurement=M)
    rho, drho = expm(tspan, rho0, H0, dH; decay=decay)
    rho_T = rho[end]
    drho_T = drho[end]

    # error_evaluation with keyword args 
    @test_nowarn error_evaluation(scheme;
        objective=:QFIM, input_error_scaling=1e-8, SLD_eps=1e-6)

    # error_control with keyword args
    @test_nowarn error_control(scheme;
        objective="QFIM", output_error_scaling=1e-6,
        input_error_scaling=1e-8, SLD_eps=1e-6, max_episode=1)

    # QFIM with keyword args
    @test_nowarn QFIM(rho_T, drho_T; LDtype=:SLD)
    @test_nowarn QFIM(rho_T, drho_T; LDtype=:RLD)
    @test_nowarn QFIM(rho_T, drho_T; LDtype=:LLD)
    @test_nowarn QFIM(rho_T, drho_T; LDtype=:SLD, exportLD=true)
    @test_nowarn QFIM(rho_T, drho_T; eps=1e-8)

    # CFIM with keyword args
    @test_nowarn CFIM(rho_T, drho_T, M; eps=1e-8)
    @test_nowarn CFIM(rho_T, drho_T, M)

    # HCRB with keyword args (W must be a matrix)
    @test_nowarn HCRB(rho_T, drho_T, [1.0;;]; eps=1e-8)

    # NHB with keyword args (W must be a matrix)
    @test_nowarn NHB(rho_T, drho_T, [1.0;;])

    # BQCRB with keyword args (rho/drho must match x grid size)
    x_vals = [0.5, 1.0, 1.5]
    p_vals = [1/3, 1/3, 1/3]
    rho_list = [rho_T, rho_T, rho_T]
    drho_list = [drho_T, drho_T, drho_T]
    @test_nowarn BQCRB(x_vals, p_vals, nothing, rho_list, drho_list;
        b=nothing, db=nothing, btype=1, LDtype=:SLD, eps=1e-8)

    # BCRB with keyword args
    @test_nowarn BCRB(x_vals, p_vals, nothing, rho_list, drho_list;
        M=nothing, b=nothing, db=nothing, btype=1, eps=1e-8)

    # Bayes & MLE: exist and accept keyword args syntactically
    # (full runtime execution has complex integration/setup requirements;
    #  isdefined checks above already verify function existence)
    @test hasmethod(Bayes, Tuple{Vector{Float64}, Vector{Float64},
        Vector{Matrix{ComplexF64}}, Vector{Int64}})
    @test hasmethod(MLE, Tuple{Vector{Float64}, Vector{Matrix{ComplexF64}},
        Vector{Int64}})
end
