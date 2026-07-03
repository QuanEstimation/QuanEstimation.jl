using Test
using LinearAlgebra
using QuanEstimationBase
using QuanEstimationBase: ControlOpt, CFIM_obj, GRAPE, autoGRAPE, DE, PSO, NM, RI,
    optimize!, init_opt, Objective, Output
using Suppressor
using Random

@testset "Sys-1: Magnetic Field Estimation (3-param)" begin

    I2 = Matrix{ComplexF64}(I, 2, 2)
    sx = SigmaX()
    sy = SigmaY()
    sz = SigmaZ()
    sx_1 = kron(sx, I2)
    sy_1 = kron(sy, I2)
    sz_1 = kron(sz, I2)
    sx_2 = kron(I2, sx)
    sy_2 = kron(I2, sy)
    sz_2 = kron(I2, sz)

    B = 1.0
    theta = pi / 4
    phi = pi / 4
    cth = cos(theta)
    sth = sin(theta)
    cph = cos(phi)
    sph = sin(phi)

    H0 = B * (sth * cph * sx_1 + sth * sph * sy_1 + cth * sz_1)
    dH_B = sth * cph * sx_1 + sth * sph * sy_1 + cth * sz_1
    dH_theta = B * (cth * cph * sx_1 + cth * sph * sy_1 - sth * sz_1)
    dH_phi = B * (-sth * sph * sx_1 + sth * cph * sy_1)
    dH = [dH_B, dH_theta, dH_phi]

    probe_ket = BellState(1)
    probe_rho = probe_ket * probe_ket'

    Hc = [sx_1, sy_1, sz_1, sx_2, sy_2, sz_2]

    @testset "State Evolution — Noiseless" begin
        for tval in [0.1, 0.5, 1.0, 2.0]
            tspan = range(0.0, tval; length=200)
            psi_an, _ = analytic_magnetic_state(tval, B, theta, phi)
            rho_an = psi_an * psi_an'

            param = Lindblad(H0, dH, tspan; dyn_method=:Expm)
            scheme = GeneralScheme(probe=probe_ket, param=param)
            rho_ev, _ = evolve(scheme)

            @test norm(rho_ev - rho_an) < 1e-10
        end
    end

    @testset "QFIM — Noiseless (Analytic rho/drho)" begin
        for tval in [0.1, 0.5, 1.0, 2.0]
            psi_an, dpsis_an = analytic_magnetic_state(tval, B, theta, phi)
            rho_an = psi_an * psi_an'
            drhos_an = [(dpsis_an[i] * psi_an' + psi_an * dpsis_an[i]') for i in 1:3]
            F_exact = analytic_magnetic_qfim_pure(tval, B, theta, phi)

            F_pure = QuanEstimationBase.QFIM_pure(rho_an, drhos_an)
            @test isapprox(F_pure, F_exact, rtol=1e-10)

            F_sld = QFIM(rho_an, drhos_an; LDtype=:SLD)
            @test isapprox(F_sld, F_exact, rtol=1e-10)
        end
    end

    @testset "QFIM — Noiseless (Evolved rho/drho)" begin
        for tval in [0.1, 0.5, 1.0, 2.0]
            tspan = range(0.0, tval; length=200)
            F_exact = analytic_magnetic_qfim_pure(tval, B, theta, phi)

            param = Lindblad(H0, dH, tspan; dyn_method=:Expm)
            scheme = GeneralScheme(probe=probe_ket, param=param)
            rho_ev, drhos_ev = evolve(scheme)

            F_sld = QFIM(rho_ev, drhos_ev; LDtype=:SLD)
            @test isapprox(F_sld, F_exact, rtol=2e-4)

            F_pure = QuanEstimationBase.QFIM_pure(rho_ev, drhos_ev)
            @test isapprox(F_pure, F_exact, rtol=2e-4)
        end
    end

    @testset "SLD Hermiticity" begin
        tval = 1.0
        tspan = range(0.0, tval; length=200)
        param = Lindblad(H0, dH, tspan; dyn_method=:Expm)
        scheme = GeneralScheme(probe=probe_ket, param=param)
        rho_ev, drhos_ev = evolve(scheme)

        Ls = [SLD(rho_ev, d) for d in drhos_ev]
        @test all(L -> norm(L - L') < 1e-13, Ls)
    end

    @testset "Scheme B finite-N J_N^max (sequential feedback analytic oracle)" begin
        # Yuan2016 sequential-feedback scheme: at N steps with unit time t=T/N,
        # J_N^max = 4 N^2 diag(t², sin²(Bt), sin²(Bt) sin²θ)  (exact, non-asymptotic).
        # N=1 reproduces the single-step scheme-A QFIM;
        # N→∞ converges to Cartesian 3/(4T²).
        for T in [0.5, 1.0, 2.0]
            for N in [1, 2, 5, 100]
                Fn = analytic_magnetic_qfim_limit_matrix_N(T, B, theta, N)
                if N == 1
                    F1 = Float64[4*T^2 0 0; 0 4*sin(B*T)^2 0; 0 0 4*sin(B*T)^2*sin(theta)^2]
                    @test isapprox(Fn, F1; rtol=1e-10)
                end
                cp = analytic_magnetic_cartesian_precision_N(T, B, N)
                if N >= 100
                    @test isapprox(cp, 3 / (4 * T^2); rtol=1e-2)
                end
            end
        end
    end

    @testset "Cartesian total precision — scheme A + scheme B N→∞" begin
        # Scheme A (no control): J_1^max(T).  Scheme B N→∞: 3/(4T²).
        # Both expressed as the Cartesian weighted trace tr(G F^{-1}) with
        # G = diag(1, B², B² sin²θ).  The bare (B,θ,φ) trace is a different number.
        for T in [0.5, 1.0, 2.0]
            # scheme A: J_1^max = 4 diag(T², sin²(BT), sin²(BT)sin²θ)
            Fa = analytic_magnetic_qfim_limit_matrix_N(T, B, theta, 1)
            cart_a = cartesian_trinv(Fa, B, theta)
            @test isapprox(cart_a, analytic_magnetic_cartesian_precision_N(T, B, 1); rtol=1e-10)
            # scheme B N→∞: 4 T^2 diag(1, B², B² sin²θ)
            Fo = analytic_magnetic_qfim_limit_matrix(T, B, theta)
            cart_b = cartesian_trinv(Fo, B, theta)
            @test isapprox(cart_b, 3 / (4 * T^2); rtol=1e-10)
        end
    end

    @testset "Noisy → Noiseless γ→0 regression (Appendix-B consistency)" begin
        # Paper Appendix-B noise formulas must reduce to scheme A when γ=0.
        # Scheme A QFIM = J_1^max = 4 diag(T², sin²(BT), sin²(BT) sin²θ).
        for T in [0.5, 1.0, 2.0]
            gam = 0.0
            Fq0 = analytic_magnetic_qfim_noisy(T, B, theta, phi, gam)
            Fa  = analytic_magnetic_qfim_limit_matrix_N(T, B, theta, 1)
            @test isapprox(Fq0, Fa; rtol=1e-10)
            p0 = analytic_magnetic_bell_probs(T, B, theta, phi, gam)
            pa = [cos(B*T)^2, sin(B*T)^2*cos(theta)^2,
                  sin(B*T)^2*sin(theta)^2*cos(phi)^2, sin(B*T)^2*sin(theta)^2*sin(phi)^2]
            @test isapprox(p0, pa; rtol=1e-10)
        end
    end

    @testset "Noisy Evolution — Properties (γ=0.1)" begin
        gamma_l = 0.1
        decay = [[sz_1, gamma_l]]

        for tval in [0.1, 0.5, 1.0, 2.0]
            tspan = range(0.0, tval; length=200)

            param = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
            scheme = GeneralScheme(probe=probe_ket, param=param)
            rho_ev, drhos_ev = evolve(scheme)

            @test isapprox(tr(rho_ev), 1.0, rtol=1e-12)
            @test norm(rho_ev - rho_ev') < 1e-12
            @test minimum(real(eigvals(rho_ev))) >= -1e-10

            F_sld = QFIM(rho_ev, drhos_ev; LDtype=:SLD)
            @test isposdef(F_sld)
            F_nn = QFIM(rho_ev, drhos_ev; LDtype=:SLD)
            @test isposdef(F_nn)
        end
    end

    @testset "Noisy Dephasing — Commuting Field (paper closed form exact)" begin
        # Pure-z field H0 = B σ3^(1) commutes with the σ3^(1) dephasing, so the
        # Appendix-B closed form (ρ rank-2, off-diagonal block × e^{-γ_p T}) is
        # lab-frame EXACT. QuanEstimation rate maps as γ_p = 2 γ_q.
        gamma_q = 0.1
        gamma_p = 2 * gamma_q
        H0z = B * sz_1
        dHz = [sz_1]                     # single parameter (θ,φ degenerate for pure-z)
        bell_projs = bell_basis().projs

        for tval in [0.3, 0.7, 1.2]
            tspan = range(0.0, tval; length=400)
            param = Lindblad(H0z, dHz, tspan, [[sz_1, gamma_q]]; dyn_method=:Expm)
            scheme = GeneralScheme(probe=probe_ket, param=param, measurement=bell_projs)
            rho_ev, _ = evolve(scheme)

            # (1) lab-frame ρ matches the Appendix-B closed form (θ=0, γ_p=2γ_q)
            rho_an = analytic_magnetic_rho_noisy(tval, B, 0.0, phi, gamma_p)
            @test norm(rho_ev - rho_an) < 1e-9

            # (2) strictly rank-2 with eigenvalues (1 ± e^{-γ_p t})/2 (paper §B)
            ev = sort(real(eigvals(rho_ev)); rev=true)
            @test abs(ev[3]) + abs(ev[4]) < 1e-9
            @test isapprox(ev[1], (1 + exp(-gamma_p * tval)) / 2; rtol=1e-6)
            @test isapprox(ev[2], (1 - exp(-gamma_p * tval)) / 2; rtol=1e-6)

            # (3) Bell-basis probabilities match the closed form
            p_num = [real(tr(bell_projs[i] * rho_ev)) for i in 1:4]
            p_an  = analytic_magnetic_bell_probs(tval, B, 0.0, phi, gamma_p)
            @test isapprox(p_num, p_an; rtol=1e-6)
        end
    end

    @testset "Noiseless vs Noisy QFIM Comparison" begin
        gamma_l = 0.1
        decay = [[sz_1, gamma_l]]

        for tval in [0.1, 0.5, 1.0]
            tspan_nn = range(0.0, tval; length=200)

            param_nn = Lindblad(H0, dH, tspan_nn; dyn_method=:Expm)
            scheme_nn = GeneralScheme(probe=probe_ket, param=param_nn)
            rho_nn, drhos_nn = evolve(scheme_nn)
            F_nn = QFIM(rho_nn, drhos_nn; LDtype=:SLD)

            param_noisy = Lindblad(H0, dH, tspan_nn, decay; dyn_method=:Expm)
            scheme_noisy = GeneralScheme(probe=probe_ket, param=param_noisy)
            rho_noisy, drhos_noisy = evolve(scheme_noisy)
            F_noisy = QFIM(rho_noisy, drhos_noisy; LDtype=:SLD)

            M_diff = F_nn - F_noisy
            @test minimum(real(eigvals(M_diff))) >= -1e-10
            @test tr(inv(F_nn)) <= tr(inv(F_noisy))
        end
    end

    @testset "Bell Measurement Probabilities" begin
        bell = bell_basis()
        bell_projs = bell.projs

        function measure_bell(rho)
            [real(tr(bell_projs[i] * rho)) for i in 1:4]
        end

        @testset "Noiseless (γ=0)" begin
            for tval in [0.1, 0.5, 1.0]
                tspan = range(0.0, tval; length=200)
                ps_an = analytic_magnetic_bell_probs(tval, B, theta, phi, 0.0)

                param = Lindblad(H0, dH, tspan; dyn_method=:Expm)
                scheme = GeneralScheme(probe=probe_ket, param=param)
                rho_ev, _ = evolve(scheme)
                ps_num = measure_bell(rho_ev)

                @test isapprox(ps_num, ps_an, rtol=1e-12)
                @test sum(ps_num) ≈ 1.0 rtol=1e-12
            end
        end

        @testset "Noisy (γ=0.1) — Properties" begin
            gamma_l = 0.1
            decay = [[sz_1, gamma_l]]
            for tval in [0.1, 0.5, 1.0]
                tspan = range(0.0, tval; length=200)
                param = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
                scheme = GeneralScheme(probe=probe_ket, param=param)
                rho_ev, _ = evolve(scheme)
                ps_num = measure_bell(rho_ev)

                @test sum(ps_num) ≈ 1.0 rtol=1e-12
                @test all(p -> p >= -1e-12, ps_num)
            end
        end
    end

    @testset "CFIM — Bell Measurement (Noiseless)" begin
        bell = bell_basis()
        bell_projs = bell.projs

        for tval in [0.1, 0.5, 1.0]
            tspan = range(0.0, tval; length=200)
            F_cfim_an = analytic_magnetic_cfim_bell(tval, B, theta, phi, 0.0; eps_val=1e-6)
            F_qfim_exact = analytic_magnetic_qfim_pure(tval, B, theta, phi)

            param = Lindblad(H0, dH, tspan; dyn_method=:Expm)
            M_meas = bell_projs
            scheme_meas = GeneralScheme(probe=probe_ket, param=param, measurement=M_meas)
            F_cfim = CFIM(scheme_meas)

            @test isapprox(F_cfim, F_cfim_an, rtol=1e-4)
            @test real(det(F_cfim)) >= 0

            pos_diff = real(eigvals(2 * F_qfim_exact - F_cfim))
            @test minimum(pos_diff) >= -1e-8
        end
    end

    @testset "CFIM — Noisy Bell (γ=0.1)" begin
        gamma_l = 0.1
        decay = [[sz_1, gamma_l]]
        bell = bell_basis()
        bell_projs = bell.projs

        for tval in [0.1, 0.5, 1.0]
            tspan = range(0.0, tval; length=200)

            param_cfim = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
            M_meas = bell_projs
            scheme_cfim = GeneralScheme(probe=probe_ket, param=param_cfim, measurement=M_meas)
            F_cfim = CFIM(scheme_cfim)

            @test real(det(F_cfim)) >= 0
            @test isposdef(F_cfim)

            param_qfim = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
            scheme_qfim = GeneralScheme(probe=probe_ket, param=param_qfim)
            rho_q, drhos_q = evolve(scheme_qfim)
            F_qfim = QFIM(rho_q, drhos_q; LDtype=:SLD)

            M_pos = real(eigvals(F_qfim - F_cfim))
            @test minimum(M_pos) >= -1e-8
        end
    end

    @testset "Analytic Optimal Control — QFIM reaches Yuan2016 limit" begin
        # Yuan2016 (PRL 117, 160801; arXiv:1601.04466): the reverse-free-evolution
        # control H_c^(1)(t) = -H0 drives the QFIM to its (B,θ,φ)-parametrization
        # optimum J^max = 4 T^2 diag(1, B^2, B^2 sin^2θ). Purely analytic oracle,
        # no iterative optimization — deterministic and exact.
        for T0 in [0.5, 1.0, 2.0]
            tspan = range(0.0, T0; length=100)
            cnum = length(tspan) - 1
            ctrl_opt = analytic_magnetic_optimal_ctrl(B, theta, phi, cnum)
            param = Lindblad(H0, dH, tspan, Hc; ctrl=ctrl_opt, dyn_method=:Expm)
            scheme = GeneralScheme(probe=probe_rho, param=param)
            Fq = QFIM(scheme)
            @test isapprox(Fq, analytic_magnetic_qfim_limit_matrix(T0, B, theta); rtol=1e-2)
            @test isapprox(tr(inv(Fq)), analytic_magnetic_qfim_limit_trinv(T0, B, theta); rtol=1e-2)
        end
    end

    @testset "GRAPE Convergence — Noiseless (CFIM+Bell → optimum)" begin
        # End-to-end: GRAPE optimizing CFIM under Bell measurement must converge to
        # the spherical QFIM optimum (1.0 at T=1). CFIM ≥ QFIM gives a hard lower
        # bound the optimizer can never breach. Deterministic (zero-init + fixed seed).
        bell_projs = bell_basis().projs
        T0 = 1.0
        tspan = range(0.0, T0; length=30)
        cnum = length(tspan) - 1
        limit = analytic_magnetic_qfim_limit_trinv(T0, B, theta)  # = 1.0

        ctrl = [zeros(cnum) for _ in eachindex(Hc)]
        param = Lindblad(H0, dH, tspan, Hc; ctrl=ctrl, dyn_method=:Expm)
        scheme = GeneralScheme(probe=probe_rho, param=param, measurement=bell_projs)
        tr_inv_before = tr(pinv(CFIM(scheme)))

        Random.seed!(1234)
        opt = ControlOpt(ctrl=ctrl, ctrl_bound=[-10.0, 10.0], seed=1234)
        obj = CFIM_obj(M=bell_projs, para_type=:multi_para)
        @suppress optimize!(scheme, opt; algorithm=GRAPE(Adam=false, max_episode=200, epsilon=0.5),
                            objective=obj)
        tr_inv_after = tr(pinv(CFIM(scheme)))
        rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

        @test tr_inv_after >= limit - 1e-6              # lower bound: CFIM ≥ QFIM optimum
        @test tr_inv_after < tr_inv_before               # monotone improvement (gradient sign)
        @test isapprox(tr_inv_after, limit; rtol=0.05)   # converges to the optimum
    end

    @testset "Control Optimization — Algorithm Coverage" begin
        # Cross-validate control-optimization algorithms against the analytic
        # spherical optimum. The lower bound (CFIM ≥ QFIM ⇒ tr(inv) ≥ limit) is a
        # hard oracle every correct algorithm must satisfy. Not-applicable
        # algorithms are recorded explicitly (regression guard).
        bell_projs = bell_basis().projs
        T0 = 1.0
        tspan = range(0.0, T0; length=30)
        cnum = length(tspan) - 1
        limit = analytic_magnetic_qfim_limit_trinv(T0, B, theta)
        obj = CFIM_obj(M=bell_projs, para_type=:multi_para)

        mkscheme() = begin
            ctrl = [zeros(cnum) for _ in eachindex(Hc)]
            param = Lindblad(H0, dH, tspan, Hc; ctrl=ctrl, dyn_method=:Expm)
            GeneralScheme(probe=probe_rho, param=param, measurement=bell_projs), ctrl
        end
        clean() = (rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true))

        @testset "GRAPE (gradient)" begin
            scheme, ctrl = mkscheme()
            before = tr(pinv(CFIM(scheme)))
            Random.seed!(1234)
            opt = ControlOpt(ctrl=ctrl, ctrl_bound=[-10.0, 10.0], seed=1234)
            @suppress optimize!(scheme, opt; algorithm=GRAPE(Adam=false, max_episode=100, epsilon=0.5), objective=obj)
            after = tr(pinv(CFIM(scheme))); clean()
            @test after >= limit - 1e-6          # lower bound never breached
            @test after < before                  # improves over no-control
        end

        @testset "PSO (swarm)" begin
            scheme, ctrl = mkscheme()
            before = tr(pinv(CFIM(scheme)))
            Random.seed!(1234)
            opt = init_opt(ControlOpt(ctrl=ctrl, ctrl_bound=[-10.0, 10.0], seed=1234), scheme)
            ow = Objective(scheme, obj); out = Output(opt; save=false)
            @suppress optimize!(opt, PSO(p_num=6, max_episode=[30, 30]), ow, scheme, out)
            after = out.f_list[end]; clean()
            @test after >= limit - 1e-6
            @test after < before
        end

        @testset "DE (evolutionary)" begin
            scheme, ctrl = mkscheme()
            before = tr(pinv(CFIM(scheme)))
            Random.seed!(1234)
            opt = init_opt(ControlOpt(ctrl=ctrl, ctrl_bound=[-10.0, 10.0], seed=1234), scheme)
            ow = Objective(scheme, obj); out = Output(opt; save=false)
            @suppress optimize!(opt, DE(p_num=6, max_episode=30), ow, scheme, out)
            after = out.f_list[end]; clean()
            @test after >= limit - 1e-6           # DE converges slowly; bound still holds
            @test after <= before + 1e-9          # non-worsening
        end

        @testset "Not applicable to ControlOpt (recorded)" begin
            scheme, ctrl = mkscheme()
            # autoGRAPE / AD: multi-parameter CFIM is broken — the autodiff path
            # mutates control buffers, triggering Zygote "Mutating arrays not
            # supported". Known limitation; this guards against silent regressions.
            opt = ControlOpt(ctrl=ctrl, ctrl_bound=[-10.0, 10.0], seed=1234)
            @test_throws Exception (@suppress optimize!(scheme, opt;
                algorithm=autoGRAPE(Adam=true, max_episode=2, epsilon=0.1), objective=obj))
            clean()
            # NM (Nelder-Mead) and RI (reverse iterative) have no ControlOpt
            # method — they target StateOpt only.
            opt2 = init_opt(ControlOpt(ctrl=ctrl, ctrl_bound=[-10.0, 10.0], seed=1234), scheme)
            ow = Objective(scheme, obj); out = Output(opt2; save=false)
            @test_throws MethodError optimize!(opt2, NM(p_num=5, max_episode=2), ow, scheme, out)
            @test_throws MethodError optimize!(opt2, RI(max_episode=2), ow, scheme, out)
        end
    end

    @testset "GRAPE Advantage — Noisy (control improves precision)" begin
        # Paper Fig.1(a): under dephasing, controls still improve the precision.
        # γ_q = 0.1 here corresponds to paper γ_p = 0.2 (QuanEstimation Lindblad
        # convention γ_q(σ3 ρ σ3 − ρ) vs paper (γ_p/2)(σ3 ρ σ3 − ρ) ⇒ γ_q = γ_p/2).
        bell_projs = bell_basis().projs
        gamma_l = 0.1
        decay = [[sz_1, gamma_l]]
        T0 = 1.0
        tspan = range(0.0, T0; length=30)
        cnum = length(tspan) - 1

        param_nc = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
        scheme_nc = GeneralScheme(probe=probe_ket, param=param_nc, measurement=bell_projs)
        tr_nc = tr(pinv(CFIM(scheme_nc)))

        ctrl = [zeros(cnum) for _ in eachindex(Hc)]
        param_ctrl = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
        scheme_ctrl = GeneralScheme(probe=probe_rho, param=param_ctrl, measurement=bell_projs)

        Random.seed!(1234)
        opt = ControlOpt(ctrl=ctrl, ctrl_bound=[-10.0, 10.0], seed=1234)
        obj = CFIM_obj(M=bell_projs, para_type=:multi_para)
        @suppress optimize!(scheme_ctrl, opt; algorithm=GRAPE(Adam=false, max_episode=200, epsilon=0.5),
                            objective=obj)
        tr_ctrl = tr(pinv(CFIM(scheme_ctrl)))
        rm("f.csv", force=true); rm("controls.dat", force=true); rm("controls.csv", force=true)

        @test tr_ctrl < tr_nc
    end

    @testset "Time Stability — optimal control tracks the optimal curve" begin
        # Paper Fig.1(a): controlled scheme has high time stability — it stays on
        # the optimal precision curve across T, whereas the uncontrolled QFIM
        # deviates (sin^2(BT) modulation). Uses the analytic optimal control.
        times = [0.8, 0.9, 1.0, 1.1, 1.2]
        dev_oc = Float64[]   # |controlled − optimum| / optimum
        dev_nc = Float64[]   # |no-control − optimum| / optimum
        for T0 in times
            tspan = range(0.0, T0; length=100)
            cnum = length(tspan) - 1
            limit = analytic_magnetic_qfim_limit_trinv(T0, B, theta)

            ctrl_opt = analytic_magnetic_optimal_ctrl(B, theta, phi, cnum)
            p_oc = Lindblad(H0, dH, tspan, Hc; ctrl=ctrl_opt, dyn_method=:Expm)
            s_oc = GeneralScheme(probe=probe_rho, param=p_oc)
            push!(dev_oc, abs(tr(inv(QFIM(s_oc))) - limit) / limit)

            p_nc = Lindblad(H0, dH, tspan; dyn_method=:Expm)
            s_nc = GeneralScheme(probe=probe_ket, param=p_nc)
            push!(dev_nc, abs(tr(inv(QFIM(s_nc))) - limit) / limit)
        end
        @test maximum(dev_oc) < 1e-2                  # controlled stays on optimal curve
        @test maximum(dev_nc) > maximum(dev_oc)       # uncontrolled deviates more
    end

end
