function analytic_sysA(t, omega)
    rho = ComplexF64[0.5 0.5*exp(-im*omega*t); 0.5*exp(im*omega*t) 0.5]
    drho = ComplexF64[0.0 -0.5*im*t*exp(-im*omega*t); 0.5*im*t*exp(im*omega*t) 0.0]
    F_exact = t^2
    return rho, [drho], F_exact
end

function analytic_sysB(t, omega, gamma_minus)
    r1 = exp(-gamma_minus * t / 2) * cos(omega * t)
    r2 = -exp(-gamma_minus * t / 2) * sin(omega * t)
    r3 = 1 - exp(-gamma_minus * t)
    rho = ComplexF64[0.5*(1+r3) 0.5*(r1-im*r2); 0.5*(r1+im*r2) 0.5*(1-r3)]
    dr1 = -t * exp(-gamma_minus * t / 2) * sin(omega * t)
    dr2 = -t * exp(-gamma_minus * t / 2) * cos(omega * t)
    dr3 = 0.0
    drho = ComplexF64[0.5*dr3 0.5*(dr1-im*dr2); 0.5*(dr1+im*dr2) -0.5*dr3]
    R = sqrt(r1^2 + r2^2 + r3^2)
    eigvals = [(1.0 + R) / 2.0, (1.0 - R) / 2.0]
    eigvecs = Vector{ComplexF64}[]
    if R > 1e-15
        n1 = ComplexF64[r1 - im * r2, R - r3]
        n1 ./= norm(n1)
        n2 = ComplexF64[r1 - im * r2, -R - r3]
        n2 ./= norm(n2)
        eigvecs = [n1, n2]
    else
        eigvecs = [ComplexF64[1.0, 0.0], ComplexF64[0.0, 1.0]]
    end
    F_exact = 0.0
    for i in 1:2, j in 1:2
        denom = eigvals[i] + eigvals[j]
        denom > 1e-15 || continue
        mat_elem = eigvecs[i]' * drho * eigvecs[j]
        F_exact += 2 * abs(mat_elem)^2 / denom
    end
    return rho, [drho], real(F_exact)
end

function analytic_sysC(t, omega, gamma)
    rho = ComplexF64[0.5 0.5*exp(-im*omega*t - 2*gamma*t); 0.5*exp(im*omega*t - 2*gamma*t) 0.5]
    d01 = -0.5 * im * t * exp(-im * omega * t - 2 * gamma * t)
    drho = ComplexF64[0.0 d01; conj(d01) 0.0]
    F_exact = t^2 * exp(-4 * gamma * t)
    return rho, [drho], F_exact
end

function bell_basis()
    Phi_plus = ComplexF64[1, 0, 0, 1] / sqrt(2)
    Phi_minus = ComplexF64[1, 0, 0, -1] / sqrt(2)
    Psi_plus = ComplexF64[0, 1, 1, 0] / sqrt(2)
    Psi_minus = ComplexF64[0, 1, -1, 0] / sqrt(2)
    kets = [Phi_plus, Phi_minus, Psi_plus, Psi_minus]
    projs = [ket * ket' for ket in kets]
    return (kets=kets, projs=projs)
end

function analytic_magnetic_state(t, B, theta, phi)
    ct = cos(B * t)
    st = sin(B * t)
    cth = cos(theta)
    sth = sin(theta)
    im = complex(0.0, 1.0)
    e_im = exp(-im * phi)
    e_ip = exp(im * phi)

    a = ct - im * st * cth
    b = -im * st * sth * e_im
    c = -im * st * sth * e_ip
    d = ct + im * st * cth

    psi = ComplexF64[a, b, c, d] / sqrt(2)

    da = -t * st - im * t * ct * cth
    db = -im * t * ct * sth * e_im
    dc = -im * t * ct * sth * e_ip
    dd = -t * st + im * t * ct * cth
    dpsi_B = ComplexF64[da, db, dc, dd] / sqrt(2)

    da = im * st * sth
    db = -im * st * cth * e_im
    dc = -im * st * cth * e_ip
    dd = -im * st * sth
    dpsi_theta = ComplexF64[da, db, dc, dd] / sqrt(2)

    da = 0.0
    db = -st * sth * e_im
    dc = st * sth * e_ip
    dd = 0.0
    dpsi_phi = ComplexF64[da, db, dc, dd] / sqrt(2)

    return psi, [dpsi_B, dpsi_theta, dpsi_phi]
end

function analytic_magnetic_qfim_pure(t, B, theta, phi)
    psi, dpsis = analytic_magnetic_state(t, B, theta, phi)
    F = zeros(Float64, 3, 3)
    for i in 1:3, j in 1:3
        overlap = dpsis[i]' * dpsis[j]
        F[i,j] = 4 * real(overlap - (dpsis[i]' * psi) * (psi' * dpsis[j]))
    end
    return F
end

# --- Noisy dephasing closed forms (paper arXiv:1710.06741, Appendix B) ---
# IMPORTANT: these closed forms (rho_noisy, qfim_noisy, bell_probs with γ≠0)
# are the paper's analytic results. They are LAB-FRAME EXACT only when the free
# Hamiltonian commutes with the dephasing operator, i.e. [H0, σ3^(1)] = 0
# (a pure-z field, θ=0). For a tilted field (H0 ∝ n̂·σ with σ1,σ2 components)
# they are an interaction-picture / commuting approximation: the true lab-frame
# ρ(T) leaks out of rank-2 and the effective decay drifts, with error
# ∝ γ t ‖[H0, σ3^(1)]‖. Numerically, QuanEstimation's decay rate maps as
# γ_paper = 2 γ_quanestimation (the off-diagonal coherence decays as e^{-2 γ_q t}).
function analytic_magnetic_rho_noisy(t, B, theta, phi, gamma)
    psi, _ = analytic_magnetic_state(t, B, theta, phi)
    rho_pure = psi * psi'
    decay = exp(-gamma * t)
    rho = copy(rho_pure)
    rho[1,3] *= decay; rho[1,4] *= decay
    rho[2,3] *= decay; rho[2,4] *= decay
    rho[3,1] *= decay; rho[3,2] *= decay
    rho[4,1] *= decay; rho[4,2] *= decay
    return rho
end

function analytic_magnetic_qfim_noisy(t, B, theta, phi, gamma)
    ct = cos(B * t)
    st = sin(B * t)
    cth = cos(theta)
    sth = sin(theta)
    c2th = cos(2 * theta)
    s2th = sin(2 * theta)
    ed = exp(-2 * gamma * t)

    F = zeros(Float64, 3, 3)

    F[1,1] = 4 * t^2 * (cth^2 * ed + sth^2)
    F[2,2] = 4 * st^2 * (cth^2 + sth^2 * (ed * ct^2 + st^2))
    F[3,3] = 4 * sth^2 * st^2 * (1 - (1 - ed) * sth^2 * st^2)

    F[1,2] = (1 - ed) * t * sin(2 * B * t) * s2th
    F[2,1] = F[1,2]

    F[1,3] = -2 * (1 - ed) * t * s2th * sth * st^2
    F[3,1] = F[1,3]

    F[2,3] = 2 * (1 - ed) * sth^3 * sin(2 * B * t) * st^2
    F[3,2] = F[2,3]

    return F
end

function analytic_magnetic_bell_probs(t, B, theta, phi, gamma)
    ct2 = cos(B * t)^2
    st2 = sin(B * t)^2
    cth2 = cos(theta)^2
    sth2 = sin(theta)^2
    ed = exp(-gamma * t)

    p1 = (1 + ed) * ct2 / 2 + (1 - ed) * st2 * cth2 / 2
    p2 = (1 - ed) * ct2 / 2 + (1 + ed) * st2 * cth2 / 2
    p3 = st2 * sth2 * (1 + ed * cos(2 * phi)) / 2
    p4 = st2 * sth2 * (1 - ed * cos(2 * phi)) / 2

    return [p1, p2, p3, p4]
end

function analytic_magnetic_cfim_bell(t, B, theta, phi, gamma; eps_val=1e-8)
    p = analytic_magnetic_bell_probs(t, B, theta, phi, gamma)
    nparam = 3
    dp = [zeros(Float64, 4) for _ in 1:nparam]
    for i in 1:nparam
        for s in 1:4
            p_plus = analytic_magnetic_bell_probs(t, B + (i==1)*eps_val, theta + (i==2)*eps_val, phi + (i==3)*eps_val, gamma)
            p_minus = analytic_magnetic_bell_probs(t, B - (i==1)*eps_val, theta - (i==2)*eps_val, phi - (i==3)*eps_val, gamma)
            dp[i][s] = (p_plus[s] - p_minus[s]) / (2 * eps_val)
        end
    end
    F = zeros(Float64, nparam, nparam)
    for i in 1:nparam, j in 1:nparam
        for s in 1:4
            p[s] > 1e-14 || continue
            F[i,j] += dp[i][s] * dp[j][s] / p[s]
        end
    end
    return F
end

# Yuan2016 (Phys. Rev. Lett. 117, 160801; arXiv:1601.04466), Eq. for J^max_N (N→∞):
# In the (B, theta, phi) parametrization, the maximal QFIM under the optimal
# sequential-feedback (reverse-free-evolution) control is
#   J^max = 4 T^2 diag(1, B^2, B^2 sin^2(theta)).
function analytic_magnetic_qfim_limit_matrix(T, B, theta)
    s2 = sin(theta)^2
    return 4 * T^2 * Float64[1.0 0.0 0.0; 0.0 B^2 0.0; 0.0 0.0 B^2*s2]
end

# tr( (J^max)^{-1} ) — the optimal total-variance limit in the (B,theta,phi) basis.
# NOTE: this differs from the Cartesian-parametrization value 3/(4T^2); for the
# (B,theta,phi) generators it equals (1/(4T^2)) (1 + 1/B^2 + 1/(B^2 sin^2 theta)).
function analytic_magnetic_qfim_limit_trinv(T, B, theta)
    return (1 / (4 * T^2)) * (1 + 1 / B^2 + 1 / (B^2 * sin(theta)^2))
end

# Yuan2016 optimal control in the continuous limit: the feedback U_k = e^{+iH0 t}
# reverses the free evolution, i.e. H_c^(1)(t) = -H0 = -B (n·σ) on the sensing
# qubit (and zero on the ancilla). Returns control amplitudes for the six fields
# [σx1, σy1, σz1, σx2, σy2, σz2], each a length-`cnum` constant vector.
function analytic_magnetic_optimal_ctrl(B, theta, phi, cnum)
    Vx1 = -B * sin(theta) * cos(phi)
    Vy1 = -B * sin(theta) * sin(phi)
    Vz1 = -B * cos(theta)
    return [fill(Vx1, cnum), fill(Vy1, cnum), fill(Vz1, cnum),
            zeros(cnum), zeros(cnum), zeros(cnum)]
end

function analytic_sysD(t, omega1, omega2, g)
    Omega = sqrt((omega1 + omega2)^2 + g^2)
    ct = cos(Omega * t)
    st = sin(Omega * t)
    alpha = (ct - im * (omega1 + omega2 + g) / Omega * st) / sqrt(2)
    beta = (ct + im * (omega1 + omega2 - g) / Omega * st) / sqrt(2)
    rho = zeros(ComplexF64, 4, 4)
    rho[1,1] = abs2(alpha)
    rho[1,4] = alpha * conj(beta)
    rho[4,1] = conj(rho[1,4])
    rho[4,4] = abs2(beta)
    dOmega_dw1 = (omega1 + omega2) / Omega
    dct = -st * Omega * t * dOmega_dw1
    dst = ct * Omega * t * dOmega_dw1
    dalpha = (dct - im * ((1/Omega - (omega1+omega2+g)/Omega^2 * dOmega_dw1) * st + (omega1+omega2+g)/Omega * dst)) / sqrt(2)
    dbeta = (dct + im * ((1/Omega - (omega1+omega2-g)/Omega^2 * dOmega_dw1) * st + (omega1+omega2-g)/Omega * dst)) / sqrt(2)
    drho1 = zeros(ComplexF64, 4, 4)
    drho1[1,1] = dalpha * conj(alpha) + alpha * conj(dalpha)
    drho1[1,4] = dalpha * conj(beta) + alpha * conj(dbeta)
    drho1[4,1] = conj(drho1[1,4])
    drho1[4,4] = dbeta * conj(beta) + beta * conj(dbeta)
    psi = [alpha; 0.0; 0.0; beta]
    dpsi_w1 = [dalpha; 0.0; 0.0; dbeta]
    F11 = 4 * real(dpsi_w1' * dpsi_w1 - abs2(dpsi_w1' * psi))
    dOmega_dw2 = (omega1 + omega2) / Omega
    dalpha2 = (dct - im * ((1/Omega - (omega1+omega2+g)/Omega^2 * dOmega_dw2) * st + (omega1+omega2+g)/Omega * dst)) / sqrt(2)
    dbeta2 = (dct + im * ((1/Omega - (omega1+omega2-g)/Omega^2 * dOmega_dw2) * st + (omega1+omega2-g)/Omega * dst)) / sqrt(2)
    drho2 = zeros(ComplexF64, 4, 4)
    drho2[1,1] = dalpha2 * conj(alpha) + alpha * conj(dalpha2)
    drho2[1,4] = dalpha2 * conj(beta) + alpha * conj(dbeta2)
    drho2[4,1] = conj(drho2[1,4])
    drho2[4,4] = dbeta2 * conj(beta) + beta * conj(dbeta2)
    dpsi_w2 = [dalpha2; 0.0; 0.0; dbeta2]
    F22 = 4 * real(dpsi_w2' * dpsi_w2 - abs2(dpsi_w2' * psi))
    F12 = 4 * real(dpsi_w1' * dpsi_w2 - (dpsi_w1' * psi) * conj(dpsi_w2' * psi))
    F_exact = ComplexF64[F11 F12; conj(F12) F22]
    return rho, [drho1, drho2], F_exact
end

# --- Scheme B (Yuan2016 sequential feedback) finite-N analytic oracles ---
# t = T/N; J_N^max = 4 N^2 diag(t², sin²(Bt), sin²(Bt) sin²θ)  (exact, non-asymptotic).
function analytic_magnetic_qfim_limit_matrix_N(T, B, theta, N)
    t = T / N
    Bt = B * t
    return 4 * N^2 * Float64[t^2 0 0; 0 sin(Bt)^2 0; 0 0 sin(Bt)^2 * sin(theta)^2]
end

# Cartesian total precision for scheme B with N steps:
#   δx1²+δx2²+δx3² = (1/(4N²)) [1/t² + 2 B² / sin²(Bt)]  (exact, non-asymptotic).
function analytic_magnetic_cartesian_precision_N(T, B, N)
    t = T / N
    return (1 / (4 * N^2)) * (1 / t^2 + 2 * B^2 / sin(B * t)^2)
end

# Cartesian weighted tr( G F^{-1} ) from a QFIM matrix F in the (B,θ,φ) basis,
# where G = diag(1, B², B² sin²θ).  The famous 3/(4T²) is this value for the
# optimal N→∞ QFIM 4 T^2 diag(1, B², B² sin²θ).
function cartesian_trinv(F, B, theta)
    Fi = inv(F)
    return real(Fi[1,1] + B^2 * Fi[2,2] + B^2 * sin(theta)^2 * Fi[3,3])
end
