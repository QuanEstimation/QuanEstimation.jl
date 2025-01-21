using QuanEstimationBase: QFIM_RLD, QFIM_LLD, QFIM_pure
function test_cramer_rao_bound_single_param()
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, M) = generate_qubit_dynamics()

    rho, drho = expm(tspan, rho0, H0, dH; decay = decay, Hc = Hc, ctrl = ctrl)
    # calculation of the CFI and QFI
    Im, F, H = Float64[], Float64[], Float64[]
    for ti = 2:length(tspan)
        I_tp = CFIM(rho[ti], drho[ti], M)
        append!(Im, I_tp)
        F_tp = QFIM(rho[ti], drho[ti])
    end
    @test all(Im .>= 0)
    @test all(F .>= 0)
end

function test_cramer_rao_bound_multi_param()
    (; tspan, psi, H0, dH, decay) = generate_LMG2_dynamics()

    rho0 = psi * psi'
    rho, drho = expm(tspan, rho0, H0, dH; decay = decay)
    Im, F, H = Matrix{Float64}[], Matrix{Float64}[], Float64[]
    for ti = 2:length(tspan)
        I_tp = CFIM(rho[ti], drho[ti], SIC(3))
        push!(Im, I_tp)
        F_tp = QFIM(rho[ti], drho[ti])
        push!(F, F_tp)
        H_tp = HCRB(rho[ti], drho[ti], I(2))
        push!(H, H_tp)
    end
    SLD(rho[end], drho[end]; rep = "eigen")
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

function test_cramer_rao_bound_kraus()
    scheme = generate_scheme_kraus()
    rho, drho = evolve(scheme)

    @test isposdef(rho)

    @test tr(pinv(QFIM(rho, drho))) >= 0

end  # function test_cramer_rao_bound_kraus

function test_qfim_bloch()
    r1 = ones(3) / sqrt(3)
    dr1 = [[0.0, 1.0, 0.0]]
    @test QFIM_Bloch(r1, dr1) ≈ 1
    r2 = ones(8) / sqrt(8)
    dr2 = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    @test QFIM_Bloch(r2, dr2) > 0
end  # function test_qfim_bloch

function test_qfim_gauss()
    R = [1.0, 0.0, 1.0, 0.0]
    dR = [zero(R)]
    D = [2.0 0 1 0; 0 1 0 0; 1 0 2 0; 0 0 0 1]
    dD = [zeros(4, 4)]
    @test QFIM_Gauss(R, dR, D, dD) ≈ 0
end  # function test_qfim_gauss

function test_fim()
    p = 1 / pi * ones(10)
    @test FIM(p, zero(p)) ≈ 0
    @test FIM(p, [[0.0, 0.0] for _ = 1:10]) ≈ zeros(2, 2)
end  # function test_fim

function test_cramer_rao()
    test_cramer_rao_bound_single_param()
    test_cramer_rao_bound_multi_param()
    test_cramer_rao_bound_kraus()
    test_qfim_bloch()
    test_qfim_gauss()
    test_fim()
end

test_cramer_rao()
