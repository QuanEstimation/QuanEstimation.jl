function generate_qubit_dynamics()
    tspan = range(0.0, 10.0, length=100)
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

    return (; tspan=tspan, rho0=rho0, H0=H0, dH=dH, Hc=Hc, decay=decay, M=M, ctrl=ctrl, ctrl_bound=ctrl_bound)
end

