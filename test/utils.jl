function generate_qubit_dynamics()
    tspan = range(0.0, 2.0, length = 100)
    rho0 = 0.5 * ones(2, 2)
    omega = 1.0
    sx = [0.0 1.0; 1.0 0.0im]
    sy = [0.0 -im; im 0.0]
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]
    Hc = [sx, sy, sz]
    sp = [0.0 1.0; 0.0 0.0im]
    sm = [0.0 0.0; 1.0 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    M1 = 0.5 * [1.0+0.0im 1.0; 1.0 1.0]
    M2 = 0.5 * [1.0+0.0im -1.0; -1.0 1.0]
    M = [M1, M2]
    cnum = length(tspan) - 1
    ctrl = [zeros(cnum) for _ in eachindex(Hc)]
    ctrl_bound = [-2.0, 2.0]

    return (;
        tspan = tspan,
        rho0 = rho0,
        H0 = H0,
        dH = dH,
        Hc = Hc,
        decay = decay,
        M = M,
        ctrl = ctrl,
        ctrl_bound = ctrl_bound,
    )
end

function generate_NV_dynamics()
    rho0 = zeros(ComplexF64, 6, 6)
    rho0[1:4:5, 1:4:5] .= 0.5
    dim = size(rho0, 1)
    sx = [0.0 1.0; 1.0 0.0]
    sy = [0.0 -im; im 0.0]
    sz = [1.0 0.0; 0.0 -1.0]
    s1 = [0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0] / sqrt(2)
    s2 = [0.0 -im 0.0; im 0.0 -im; 0.0 im 0.0] / sqrt(2)
    s3 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0]
    Is = I1, I2, I3 = [kron(I(3), sx), kron(I(3), sy), kron(I(3), sz)]
    S = S1, S2, S3 = [kron(s1, I(2)), kron(s2, I(2)), kron(s3, I(2))]
    B = B1, B2, B3 = [5.0e-4, 5.0e-4, 5.0e-4]
    cons = 100
    D = (2pi * 2.87 * 1000) / cons
    gS = (2pi * 28.03 * 1000) / cons
    gI = (2pi * 4.32) / cons
    A1 = (2pi * 3.65) / cons
    A2 = (2pi * 3.03) / cons
    H0 = sum([
        D * kron(s3^2, I(2)),
        sum(gS * B .* S),
        sum(gI * B .* Is),
        A1 * (kron(s1, sx) + kron(s2, sy)),
        A2 * kron(s3, sz),
    ])
    dH = gS * S + gI * Is
    Hc = [S1, S2, S3]
    decay = [[S3, 2 * pi / cons]]
    M = [QuanEstimation.basis(dim, i) * QuanEstimation.basis(dim, i)' for i = 1:dim]
    tspan = range(0.0, 2.0, length = 100)
    cnum = length(tspan) - 1
    Random.seed!(1234)
    ctrl = [-0.2 * ones(cnum) + 0.05 * rand(cnum) for _ in eachindex(Hc)]
    ctrl_bound = [-0.2, 0.2]

    return (;
        rho0 = rho0,
        H0 = H0,
        dH = dH,
        Hc = Hc,
        decay = decay,
        M = M,
        tspan = tspan,
        ctrl = ctrl,
        ctrl_bound = ctrl_bound,
    )
end

function generate_LMG1_dynamics()
    N = 3
    j, theta, phi = N ÷ 2, 0.5pi, 0.5pi
    Jp = Matrix(spdiagm(1 => [sqrt(j * (j + 1) - m * (m + 1)) for m = j:-1:-j][2:end]))
    Jm = Jp'
    psi =
        exp(0.5 * theta * exp(im * phi) * Jm - 0.5 * theta * exp(-im * phi) * Jp) *
        basis(Int(2 * j + 1), 1)
    lambda, g, h = 1.0, 0.5, 0.1
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5im * (Jp - Jm)
    Jz = spdiagm(j:-1:-j)
    H0 = -lambda * (Jx * Jx + g * Jy * Jy) / N - h * Jz
    dH = [-lambda * Jy * Jy / N]
    decay = [[Jz, 0.1]]
    tspan = range(0.0, 10.0, length = 100)

    return (; tspan = tspan, psi = psi, H0 = H0, dH = dH, decay = decay)
end

function generate_LMG2_dynamics()
    N = 3
    j, theta, phi = N ÷ 2, 0.5pi, 0.5pi
    Jp = Matrix(spdiagm(1 => [sqrt(j * (j + 1) - m * (m + 1)) for m = j:-1:-j][2:end]))
    Jm = Jp'
    psi =
        exp(0.5 * theta * exp(im * phi) * Jm - 0.5 * theta * exp(-im * phi) * Jp) *
        basis(Int(2 * j + 1), 1)
    lambda, g, h = 1.0, 0.5, 0.1
    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5im * (Jp - Jm)
    Jz = spdiagm(j:-1:-j)
    H0 = -lambda * (Jx * Jx + g * Jy * Jy) / N + g * Jy^2 / N - h * Jz
    dH = [-lambda * Jy * Jy / N, -Jz]
    decay = [[Jz, 0.1]]
    tspan = range(0.0, 10.0, length = 100)
    W = [1/3 0.0; 0.0 2/3]


    return (; tspan = tspan, psi = psi, H0 = H0, dH = dH, decay = decay, W = W)
end
