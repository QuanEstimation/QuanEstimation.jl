function test_cramer_rao_bound_single_param()
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, M) = generate_qubit_dynamics()

    rho, drho = expm(tspan, rho0, H0, dH, decay=decay, Hc=Hc, ctrl=ctrl)
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

    rho, drho = expm(tspan, psi*psi', H0, dH; decay=decay)
    Im, F, H = Matrix{Float64}[], Matrix{Float64}[], Float64[]
    for ti = 2:length(tspan)
        I_tp = CFIM(rho[ti], drho[ti], SIC(3))
        push!(Im, I_tp)
        F_tp = QFIM(rho[ti], drho[ti])
        push!(F, F_tp)
        H_tp = HCRB(rho[ti], drho[ti], I(2))
        push!(H, H_tp)
    end
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

test_cramer_rao_bound_single_param()
test_cramer_rao_bound_multi_param()
test_cramer_rao_bound_kraus()