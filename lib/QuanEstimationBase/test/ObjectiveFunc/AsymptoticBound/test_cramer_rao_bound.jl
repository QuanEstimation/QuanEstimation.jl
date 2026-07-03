using LinearAlgebra
using Test
using QuanEstimationBase: 
    QFIM,       
    CFIM,
    QFIM_RLD,
    QFIM_LLD,
    QFIM_pure,
    FIM,       
    HCRB,
    NHB,        
    SLD,         
    SLD_liouville,  
    SLD_qr,  
    RLD,
    LLD,   
    expm,       
    basis,       
    SIC,
    Kraus,
    Lindblad,
    GeneralScheme,
    evolve,
    Hamiltonian,
    SigmaX, SigmaY, SigmaZ,
    PlusState,
    QFIM_Bloch,
    QFIM_Gauss,
    FI_Expt


function test_cramer_rao_bound_single_param()
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, M) = generate_qubit_dynamics()

    rho, drho = expm(tspan, rho0, H0, dH; decay=decay, Hc=Hc, ctrl=ctrl)
    # calculation of the CFI and QFI
    Im, F, H = Float64[], Float64[], Float64[]
    for ti = 2:length(tspan)
        I_tp = CFIM(rho[ti], drho[ti], M)
        append!(Im, I_tp)
        F_tp = QFIM(rho[ti], drho[ti])
        push!(F, F_tp[1, 1])
    end
    @test all(Im .>= 0)
    @test all(F .>= 0)
end

function test_cramer_rao_bound_multi_param()
    (; tspan, psi, H0, dH, decay) = generate_LMG2_dynamics()

    rho0 = psi * psi'
    rho, drho = expm(tspan, rho0, H0, dH; decay=decay)
    Im, F, H = Matrix{Float64}[], Matrix{Float64}[], Float64[]
    for ti = 2:length(tspan)
        I_tp = CFIM(rho[ti], drho[ti], SIC(3))
        push!(Im, I_tp)
        F_tp = QFIM(rho[ti], drho[ti])
        push!(F, F_tp)
        H_tp = HCRB(rho[ti], drho[ti], I(2))
        push!(H, H_tp)
    end
    SLD(rho[end], drho[end]; rep="eigen")
    SLD_liouville(rho[end], drho[end])
    SLD_qr(rho[end], drho[end][1])
    RLD(rho[end], drho[end])
    LLD(rho[end], drho[end])
    CFIM(rho[end], drho[end])
    QFIM_RLD(rho[end], drho[end])
    QFIM_RLD(rho[end], drho[end][1])
    QFIM_LLD(rho[end], drho[end])
    QFIM_LLD(rho[end], drho[end][1])
    QFIM_pure(rho0, [zero(rho0) for _ = 1:2])
    QFIM_pure(rho0, zero(rho0))
    NHB(rho[end], drho[end], one(zeros(2, 2)))

    @test all([tr(pinv(i)) >= 0 for i in Im])
    @test all([tr(pinv(f)) >= 0 for f in F])
    @test all(H .>= 0)
end

function test_bounds_with_scheme()
    H0(u) = (SigmaX() * cos(u) + SigmaZ() * sin(u))/2
    dH(u) = [(-SigmaX() * sin(u) + SigmaZ() * cos(u))/2] 
    ham = Hamiltonian(H0, dH, pi/4)
    dynamics = Lindblad(ham,0:0.01:1,[SigmaY()],[[SigmaZ(), 0.01]]) 
    scheme = GeneralScheme(; probe=PlusState(), param=dynamics, measurement = SIC(2)) 

    @test all(eigen(QFIM(scheme; LDtype=:SLD)).values .>= 0)
    @test all(eigen(CFIM(scheme)).values .>= 0) 
    @test HCRB(scheme) >= 0
    @test NHB(scheme) >= 0
end  # function test_bounds_with_scheme

function test_cramer_rao_bound_kraus()
    scheme = generate_scheme_kraus()
    rho, drho = evolve(scheme)

    @test isposdef(rho)

    @test tr(pinv(QFIM(rho, drho))) >= 0

end  # function test_cramer_rao_bound_kraus

function test_qfim_bloch()
    r1 = ones(3)/sqrt(3)
    dr1 = [[0.0,1.0,0.0]]
    @test QFIM_Bloch(r1, dr1) ≈ 1
    r2  = ones(8)/sqrt(8)
    dr2 = [[0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]]
    @test QFIM_Bloch(r2, dr2) > 0
end  # function test_qfim_bloch

function test_qfim_gauss()
    R =  [1.0,0.0,1.0,0.0]
    dR = [zero(R)]
    D = [2.0 0 1 0;0 1 0 0;1 0 2 0;0 0 0 1]
    dD=[zeros(4,4)]
    @test QFIM_Gauss(R, dR, D, dD) ≈ 0
end  # function test_qfim_gauss

function test_fim()
    p = 1 / pi * ones(10)
    @test FIM(p, zero(p)) ≈ 0
    @test FIM(p, [[0.0, 0.0] for _ = 1:10]) ≈ zeros(2, 2)
end  # function test_fim

# === error/branch path: error/branch path coverage ===

function test_sld_bad_rep()
    rho = ComplexF64[0.6 0.0; 0.0 0.4]
    drho = ComplexF64[0.1 0.05; 0.05 -0.1]
    @test_throws ArgumentError SLD(rho, [drho]; rep="bad")
end

function test_qfim_scheme_rld_lld()
    scheme = generate_qubit_scheme()
    # QFIM with RLD/LLD via Scheme
    F_rld = QFIM(scheme; LDtype=:RLD)
    F_lld = QFIM(scheme; LDtype=:LLD)
    @test isapprox(F_rld, F_lld, rtol=1e-10)
    @test all(isfinite.(F_rld))
    @test all(isfinite.(F_lld))
end

function test_qfim_bloch_mixed_state()
    # Mixed state: r_norm < 1 for dim=2
    r_mixed = [0.0, 0.0, 0.5]  # norm = 0.5 < 1 (mixed)
    dr_mixed = [[0.0, 1.0, 0.0]]
    F_mixed = QFIM_Bloch(r_mixed, dr_mixed)
    @test isfinite(F_mixed)
    @test F_mixed[1, 1] >= 0
end

function test_qfim_bloch_multipara()
    # Multi-parameter Bloch QFIM for qubit (Bloch dim=3)
    r = [0.0, 0.0, 0.5]  # mixed state, 3 components for qubit
    dr = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]  # 2 parameters
    F_mp = QFIM_Bloch(r, dr)
    @test size(F_mp) == (2, 2)
    @test isfinite(F_mp[1, 1])
    @test isfinite(F_mp[2, 2])
end

function test_qfim_full_trajectory()
    scheme = generate_qubit_scheme()
    (; tspan) = generate_qubit_dynamics()
    F_traj = QFIM(scheme; full_trajectory=true)
    @test F_traj isa Vector
    @test length(F_traj) == length(tspan)
    @test all(F -> all(isfinite, F), F_traj)
end

function test_cfim_full_trajectory()
    (; tspan, rho0, H0, dH, decay, M) = generate_qubit_dynamics()
    dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dynamics, measurement=M)
    C_traj = CFIM(scheme; full_trajectory=true)
    @test C_traj isa Vector
    @test length(C_traj) == length(tspan)
end

function test_fi_expt()
    y1 = randn(100) .+ 0.0
    y2 = randn(100) .+ 0.1
    dx = abs(sum(y1) / length(y1) - sum(y2) / length(y2))
    F_norm = FI_Expt(y1, y2, dx; ftype=:norm)
    @test isfinite(F_norm)
    @test F_norm >= 0
end

function test_cramer_rao()
    @testset "Single-parameter CR bounds" begin test_cramer_rao_bound_single_param() end
    @testset "Multi-parameter CR bounds" begin test_cramer_rao_bound_multi_param() end
    @testset "Kraus CR bounds" begin test_cramer_rao_bound_kraus() end
    @testset "Bounds with scheme" begin test_bounds_with_scheme() end
    @testset "QFIM Bloch" begin test_qfim_bloch() end
    @testset "QFIM Gauss" begin test_qfim_gauss() end
    @testset "FIM" begin test_fim() end
    @testset "SLD bad rep" begin test_sld_bad_rep() end
    @testset "QFIM scheme RLD/LLD" begin test_qfim_scheme_rld_lld() end
    @testset "QFIM Bloch mixed state" begin test_qfim_bloch_mixed_state() end
    @testset "QFIM Bloch multipara" begin test_qfim_bloch_multipara() end
    @testset "QFIM full trajectory" begin test_qfim_full_trajectory() end
    @testset "CFIM full trajectory" begin test_cfim_full_trajectory() end
    @testset "FI Expt" begin test_fi_expt() end

    # Bug #9: RLD/LLD existence check
    @testset "#9 RLD/LLD existence check" begin
        rho_full = ComplexF64[0.6 0.0; 0.0 0.4]
        drho_full = ComplexF64[0.1 0.05; 0.05 -0.1]
        R = RLD(rho_full, drho_full; rep="original")
        L = LLD(rho_full, drho_full; rep="original")
        @test norm(rho_full * R - drho_full) < 1e-10
        @test norm(L * rho_full - drho_full) < 1e-10
        rho_sing = ComplexF64[1.0 0.0; 0.0 0.0]
        drho_cross = ComplexF64[0.0 0.5; 0.5 0.0]
        @test_throws ErrorException RLD(rho_sing, drho_cross)
        @test_throws ErrorException LLD(rho_sing, drho_cross)
    end

    # pinv bypass: QFIM_RLD/LLD calls operator functions
    @testset "pinv bypass: QFIM_RLD/LLD calls operator functions" begin
        rho_full = ComplexF64[0.6 0.0; 0.0 0.4]
        drho_full = ComplexF64[0.1 0.05; 0.05 -0.1]
        F_rld = QFIM(rho_full, [drho_full]; LDtype=:RLD)
        F_lld = QFIM(rho_full, [drho_full]; LDtype=:LLD)
        @test isapprox(F_rld, F_lld, rtol=1e-10)
    end
end

@testset "Cramer-Rao bounds" begin test_cramer_rao() end
