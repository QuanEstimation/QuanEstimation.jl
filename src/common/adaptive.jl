function adaptive(
    x::AbstractVector,
    p,
    rho0::AbstractMatrix,
    tspan,
    H,
    dH;
    save_file = false,
    max_episode::Int = 1000,
    eps::Float64 = 1e-8,
    Hc::Union{Vector,Nothing} = nothing,
    ctrl::Union{Vector,Nothing} = nothing,
    decay::Union{Vector,Nothing} = nothing,
    M::Union{AbstractVector,Nothing} = nothing,
    W::Union{Matrix,Nothing} = nothing,
)
    dim = size(rho0)[1]
    para_num = length(x)
    if M == nothing
        M = SIC(size(rho0)[1])
    end
    if decay == nothing
        decay_opt = [zeros(ComplexF64, dim, dim)]
        gamma = [0.0]
    else
        decay_opt = [decay[i][1] for i = 1:len(decay)]
        gamma = [decay[i][2] for i = 1:len(decay)]
    end
    if Hc == nothing
        Hc = [zeros(ComplexF64, dim, dim)]
        ctrl = [zeros(length(tspan) - 1)]
    elseif ctrl == nothing
        ctrl = [zeros(length(tspan) - 1) for j in range(length(Hc))]
    else
        ctrl_length = length(ctrl)
        ctrlnum = length(Hc)
        if ctrlnum < ctrl_length
            throw(
                "There are $ctrlnum control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences",
            )
        elseif ctrlnum > ctrl_length
            println(
                "Not enough coefficients sequences: there are $ctrlnum control Hamiltonians 
                 but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.",
            )
            number = ceil((length(tspan) - 1) / length(ctrl[1]))
            if mod(length(tspan) - 1, length(ctrl[1])) != 0
                tnum = number * length(ctrl[1])
                tspan = arange(tspan[1], stop = tspan[end], length = tnum + 1) |> collect
            end
        end
    end
    if W == nothing
        W = zeros(para_num, para_num)
    end

    if para_num == 1
        #### singleparameter senario ####
        p_num = length(p)

        F = zeros(p_num)
        for hi = 1:p_num
            dynamics = Lindblad(H[hi], dH[hi], Hc, ctrl, rho0, tspan, decay_opt, gamma)
            rho_tp, drho_tp = evolve(dynamics)
            F[hi] = CFIM(rho_tp, drho_tp[1], M; eps = eps)
        end
        idx = findmax(F)[2]
        x_opt = x[1][idx]
        println("The optimal parameter is $x_opt")

        u = 0.0
        y, xout, pout = [], [], []
        for ei = 1:max_episode
            rho = [zeros(ComplexF64, dim, dim) for i = 1:p_num]
            for hj = 1:p_num
                x_idx = findmin(abs.(x[1] .- (x[1][hj] + u)))[2]
                H_tp = H[x_idx]
                dH_tp = dH[x_idx]
                dynamics = Lindblad(H_tp, dH_tp, Hc, ctrl, rho0, tspan, decay_opt, gamma)
                rho_tp, drho_tp = evolve(dynamics)
                rho[hj] = rho_tp
            end
            println("The tunable parameter is $u")
            print("Please enter the experimental result: ")
            enter = readline()
            res_exp = parse(Int64, enter)
            res_exp = Int(res_exp + 1)

            pyx = real.(tr.(rho .* [M[res_exp]]))

            py = trapz(x[1], pyx .* p)
            p_update = pyx .* p / py
            p = p_update
            p_idx = findmax(p)[2]
            x_out = x[1][p_idx]
            println("The estimator is $x_out ($ei episodes)")
            u = x_opt - x_out

            if mod(ei, 50) == 0
                if (x_out + u) > x[1][end] || (x_out + u) < x[1][1]
                    throw("please increase the regime of the parameters.")
                end
            end
            append!(xout, x_out)
            append!(y, Int(res_exp - 1))
            append!(pout, [p])
        end
        if save_file == false
            open("pout.csv", "w") do f
                writedlm(f, [p])
            end
            open("xout.csv", "w") do m
                writedlm(m, xout)
            end
            open("y.csv", "w") do n
                writedlm(n, y)
            end
        else
            open("pout.csv", "w") do f
                writedlm(f, pout)
            end
            open("xout.csv", "w") do m
                writedlm(m, xout)
            end
            open("y.csv", "w") do n
                writedlm(n, y)
            end
        end
    else
        #### multiparameter senario ####
        p_num = length(p |> vec)
        x_list = [(Iterators.product(x...))...]

        dynamics_res = [
            evolve(Lindblad(H_tp, dH_tp, Hc, ctrl, rho0, tspan, decay_opt, gamma)) for
            (H_tp, dH_tp) in zip(H, dH)
        ]
        F_all = zeros(p_num)
        for hi = 1:p_num
            F_tp = CFIM(dynamics_res[hi][1], dynamics_res[hi][2], M; eps = eps)
            F_all[hi] = abs(det(F_tp)) < eps ? eps : 1.0 / real(tr(W * inv(F_tp)))
        end
        F = reshape(F_all, size(p))
        idx = findmax(F)[2]
        x_opt = [x[i][idx[i]] for i = 1:para_num]
        println("The optimal parameter are $x_opt")

        u = [0.0 for i = 1:para_num]
        y, xout, pout = [], [], []
        for ei = 1:max_episode
            rho = Array{Matrix{ComplexF64}}(undef, p_num)
            for hj = 1:p_num
                x_idx =
                    [findmin(abs.(x[k] .- (x_list[hj][k] + u[k])))[2] for k = 1:para_num]
                H_tp = H[x_idx...]
                dH_tp = dH[x_idx...]
                dynamics = Lindblad(H_tp, dH_tp, Hc, ctrl, rho0, tspan, decay_opt, gamma)
                rho_tp, drho_tp = evolve(dynamics)
                rho[hj] = rho_tp
            end

            println("The tunable parameter are $u")
            print("Please enter the experimental result: ")
            enter = readline()
            res_exp = parse(Int64, enter)
            res_exp = Int(res_exp + 1)

            pyx_list = real.(tr.(rho .* [M[res_exp]]))
            pyx = reshape(pyx_list, size(p))

            arr = p .* pyx
            py = trapz(tuple(x...), arr)
            p_update = p .* pyx / py
            p = p_update

            p_idx = findmax(p)[2]
            x_out = [x[i][p_idx[i]] for i = 1:para_num]
            println("The estimator are $x_out ($ei episodes)")
            u = x_opt .- x_out

            if mod(ei, 50) == 0
                for un = 1:para_num
                    if (x_out[un] + u[un]) > x[un][end] || (x_out[un] + u[un]) < x[un][1]
                        throw("Please increase the regime of the parameters.")
                    end
                end
            end
            append!(xout, [x_out])
            append!(y, Int(res_exp - 1))
            append!(pout, [p])
        end
        if save_file == false
            open("pout.csv", "w") do f
                writedlm(f, [p])
            end
            open("xout.csv", "w") do m
                writedlm(m, xout)
            end
            open("y.csv", "w") do n
                writedlm(n, y)
            end
        else
            open("pout.csv", "w") do f
                writedlm(f, pout)
            end
            open("xout.csv", "w") do m
                writedlm(m, xout)
            end
            open("y.csv", "w") do n
                writedlm(n, y)
            end
        end
    end
end


function adaptive(
    x::AbstractVector,
    p,
    rho0::AbstractMatrix,
    K,
    dK;
    save_file = false,
    max_episode::Int = 1000,
    eps::Float64 = 1e-8,
    M::Union{AbstractVector,Nothing} = nothing,
    W::Union{Matrix,Nothing} = nothing,
)
    dim = size(rho0)[1]
    para_num = length(x)

    if W == nothing
        W = zeros(para_num, para_num)
    end

    if M == nothing
        M = SIC(size(rho0)[1])
    end

    if para_num == 1
        #### singleparameter senario ####
        p_num = length(p)

        F = zeros(p_num)
        for hi = 1:p_num
            rho_tp = sum([Ki * rho0 * Ki' for Ki in K[hi]])
            drho_tp = [
                sum([dKi * rho0 * Ki' + Ki * rho0 * dKi' for (Ki, dKi) in zip(K[hi], dKj)]) for dKj in dK[hi]
            ]
            F[hi] = CFIM(rho_tp, drho_tp[1], M; eps = eps)
        end
        indx = findmax(F)[2]
        x_opt = x[1][indx]
        println("The optimal parameter is $x_opt")

        u = 0.0
        y, xout, pout = [], [], []
        for ei = 1:max_episode
            rho = [zeros(ComplexF64, dim, dim) for i = 1:p_num]
            for hj = 1:p_num
                x_idx = findmin(abs.(x[1] .- (x[1][hj] + u)))[2]
                rho[hj] = sum([Ki * rho0 * Ki' for Ki in K[x_idx]])
            end
            println("The tunable parameter is $u")
            print("Please enter the experimental result: ")
            enter = readline()
            res_exp = parse(Int64, enter)
            res_exp = Int(res_exp + 1)

            pyx = real.(tr.(rho .* [M[res_exp]]))

            py = trapz(x[1], pyx .* p)
            p_update = pyx .* p / py
            p = p_update
            p_idx = findmax(p)[2]
            x_out = x[1][p_idx]
            println("The estimator is $x_out ($ei episodes)")
            u = x_opt - x_out

            if mod(ei, 50) == 0
                if (x_out + u) > x[1][end] || (x_out + u) < x[1][1]
                    throw("please increase the regime of the parameters.")
                end
            end
            append!(xout, x_out)
            append!(y, Int(res_exp - 1))
            append!(pout, [p])
        end
        if save_file == false
            open("pout.csv", "w") do f
                writedlm(f, [p])
            end
            open("xout.csv", "w") do m
                writedlm(m, xout)
            end
            open("y.csv", "w") do n
                writedlm(n, y)
            end
        else
            open("pout.csv", "w") do f
                writedlm(f, pout)
            end
            open("xout.csv", "w") do m
                writedlm(m, xout)
            end
            open("y.csv", "w") do n
                writedlm(n, y)
            end
        end
    else
        #### multiparameter senario ####
        k_num = length(vec(K)[1])
        para_num = length(x)
        x_list = [(Iterators.product(x...))...]

        rho_tp = [sum([Ki * rho0 * Ki' for Ki in K_tp]) for K_tp in K]

        drho_tp = [
            [
                sum([dKi * rho0 * Ki' + Ki * rho0 * dKi' for (Ki, dKi) in zip(K_tp, dKj)]) for dKj in [[dK_tp[i][j] for i = 1:k_num] for j = 1:para_num]
            ] for (K_tp, dK_tp) in zip(K, dK)
        ]

        F_all = zeros(length(p |> vec))
        for hi = 1:length(p |> vec)
            F_tp = CFIM(rho_tp[hi], drho_tp[hi], M; eps = eps)
            F_all[hi] = abs(det(F_tp)) < eps ? eps : 1.0 / real(tr(W * inv(F_tp)))
        end

        F = reshape(F_all, size(p))
        idx = findmax(F)[2]
        x_opt = [x[i][idx[i]] for i = 1:para_num]
        println("The optimal parameter are $x_opt")

        u = [0.0 for i = 1:para_num]
        y, xout, pout = [], [], []
        for ei = 1:max_episode
            rho = Array{Matrix{ComplexF64}}(undef, length(p |> vec))
            for hj = 1:length(p |> vec)
                x_indx =
                    [findmin(abs.(x[k] .- (x_list[hj][k] + u[k])))[2] for k = 1:para_num]
                rho_tp = sum([Ki * rho0 * Ki' for Ki in K[x_indx...]])
                rho[hj] = rho_tp
            end

            println("The tunable parameter are $u")
            print("Please enter the experimental result: ")
            enter = readline()
            res_exp = parse(Int64, enter)
            res_exp = Int(res_exp + 1)

            pyx_list = real.(tr.(rho .* [M[res_exp]]))
            pyx = reshape(pyx_list, size(p))
            arr = p .* pyx
            py = trapz(tuple(x...), arr)
            p_update = p .* pyx / py
            p = p_update

            p_idx = findmax(p)[2]
            x_out = [x[i][p_idx[i]] for i = 1:para_num]
            println("The estimator are $x_out ($ei episodes)")
            u = x_opt .- x_out

            if mod(ei, 50) == 0
                for un = 1:para_num
                    if (x_out[un] + u[un]) > x[un][end] || (x_out[un] + u[un]) < x[un][1]
                        throw("Please increase the regime of the parameters.")
                    end
                end
            end
            append!(xout, [x_out])
            append!(y, Int(res_exp + 1))
            append!(pout, [p])
        end
        if save_file == false
            open("pout.csv", "w") do f
                writedlm(f, [p])
            end
            open("xout.csv", "w") do m
                writedlm(m, xout)
            end
            open("y.csv", "w") do n
                writedlm(n, y)
            end
        else
            open("pout.csv", "w") do f
                writedlm(f, pout)
            end
            open("xout.csv", "w") do m
                writedlm(m, xout)
            end
            open("y.csv", "w") do n
                writedlm(n, y)
            end
        end
    end
end

mutable struct adaptMZI
    x::Any
    p::Any
    rho0::Any
end

function online(apt::adaptMZI; output::String = "phi")
    (; x, p, rho0) = apt
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1)' |> sparse
    adaptMZI_online(x, p, rho0, a, output)
end

function brgd(n)
    if n == 1
        return ["0", "1"]
    end
    L0 = brgd(n - 1)
    L1 = deepcopy(L0)
    reverse!(L1)
    L0 = ["0" * l for l in L0]
    L1 = ["1" * l for l in L1]
    return deepcopy(vcat(L0, L1))
end

function offline(apt::adaptMZI, alg; eps = GLOBAL_EPS)
    (; x, p, rho0) = apt
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1)' |> sparse
    comb = brgd(N) |> x -> [[parse(Int, s) for s in ss] for ss in x]
    if alg isa DE
        (; p_num, ini_population, c, cr, rng, max_episode) = alg
        if ismissing(ini_population)
            ini_population = ([apt.rho0],)
        end
        DE_DeltaPhiOpt(
            x,
            p,
            rho0,
            a,
            comb,
            p_num,
            ini_population[1],
            c,
            cr,
            rng.seed,
            max_episode,
            eps,
        )
    elseif alg isa PSO
        (; p_num, ini_particle, c0, c1, c2, rng, max_episode) = alg
        if ismissing(ini_particle)
            ini_particle = ([apt.rho0],)
        end
        PSO_DeltaPhiOpt(
            x,
            p,
            rho0,
            a,
            comb,
            p_num,
            ini_particle[1],
            c0,
            c1,
            c2,
            rng.seed,
            max_episode,
            eps,
        )
    end
end

function adaptMZI_online(x, p, rho0, a, output)

    N = size(a)[1] - 1
    exp_ix = [exp(1.0im * xi) for xi in x]
    phi_span = range(-pi, stop = pi, length = length(x)) |> collect

    phi = 0.0
    a_res = [Matrix{ComplexF64}(I, (N + 1)^2, (N + 1)^2) for i = 1:length(x)]
    xout, y = [], []
    if output == "phi"
        for ei = 1:N-1
            println("The tunable phase is $phi ($ei episodes)")
            print("Please enter the experimental result: ")
            enter = readline()
            u = parse(Int64, enter)

            pyx = zeros(length(x)) |> sparse
            for xi = 1:length(x)
                a_res_tp = a_res[xi] * a_u(a, x[xi], phi, u)
                pyx[xi] =
                    real(tr(rho0 * a_res_tp' * a_res_tp)) *
                    (factorial(N - ei) / factorial(N))
                a_res[xi] = a_res_tp
            end

            M_res = zeros(length(phi_span))
            for mj = 1:length(phi_span)
                M1_res = trapz(x, pyx .* p)
                pyx0, pyx1 = zeros(length(x)), zeros(length(x))
                M2_res = 0.0
                for xj = 1:length(x)
                    a_res0 = a_res[xj] * a_u(a, x[xj], phi_span[mj], 0)
                    a_res1 = a_res[xj] * a_u(a, x[xj], phi_span[mj], 1)
                    pyx0[xj] =
                        real(tr(rho0 * a_res0' * a_res0)) *
                        (factorial(N - (ei + 1)) / factorial(N))
                    pyx1[xj] =
                        real(tr(rho0 * a_res1' * a_res1)) *
                        (factorial(N - (ei + 1)) / factorial(N))
                    M2_res =
                        abs(trapz(x, pyx0 .* p .* exp_ix)) +
                        abs(trapz(x, pyx1 .* p .* exp_ix))
                end
                M_res[mj] = M2_res / M1_res
            end
            indx_m = findmax(M_res)[2]
            phi_update = phi_span[indx_m]

            append!(xout, phi)
            append!(y, u)
            phi = phi_update
        end
        println("The estimator of the unknown phase is $phi ")
        append!(xout, phi)
        open("xout.csv", "w") do m
            writedlm(m, xout)
        end
        open("y.csv", "w") do n
            writedlm(n, y)
        end
    else
        println("The initial tunable phase is $phi")
        for ei = 1:N-1
            print("Please enter the experimental result: ")
            enter = readline()
            u = parse(Int64, enter)

            pyx = zeros(length(x))
            for xi = 1:length(x)
                a_res_tp = a_res[xi] * a_u(a, x[xi], phi, u)
                pyx[xi] =
                    real(tr(rho0 * a_res_tp' * a_res_tp)) *
                    (factorial(N - ei) / factorial(N))
                a_res[xi] = a_res_tp
            end

            M_res = zeros(length(phi_span))
            for mj = 1:length(phi_span)
                M1_res = trapz(x, pyx .* p)
                pyx0, pyx1 = zeros(length(x)), zeros(length(x))
                M2_res = 0.0
                for xj = 1:length(x)
                    a_res0 = a_res[xj] * a_u(a, x[xj], phi_span[mj], 0)
                    a_res1 = a_res[xj] * a_u(a, x[xj], phi_span[mj], 1)
                    pyx0[xj] =
                        real(tr(rho0 * a_res0' * a_res0)) *
                        (factorial(N - (ei + 1)) / factorial(N))
                    pyx1[xj] =
                        real(tr(rho0 * a_res1' * a_res1)) *
                        (factorial(N - (ei + 1)) / factorial(N))
                    M2_res =
                        abs(trapz(x, pyx0 .* p .* exp_ix)) +
                        abs(trapz(x, pyx1 .* p .* exp_ix))
                end
                M_res[mj] = M2_res / M1_res
            end
            indx_m = findmax(M_res)[2]
            phi_update = phi_span[indx_m]
            println(
                "The adjustments of the feedback phase is $(abs(phi_update-phi)) ($ei episodes)",
            )
            append!(xout, abs(phi_update - phi))
            append!(y, u)

            phi = phi_update
        end
        open("xout.csv", "w") do m
            writedlm(m, xout)
        end
        open("y.csv", "w") do n
            writedlm(n, y)
        end
    end
end

function adaptMZI_offline(delta_phi, x, p, rho0, a, comb, eps)
    N = size(a)[1] - 1
    exp_ix = [exp(1.0im * xi) for xi in x]

    M_res = zeros(length(comb))
    for ui = 1:length(comb)
        u = comb[ui]
        phi = 0.0

        a_res = [Matrix{ComplexF64}(I, (N + 1)^2, (N + 1)^2) for i = 1:length(x)]
        for ei = 1:N-1
            phi = phi - (-1)^u[ei] * delta_phi[ei]
            for xi = 1:length(x)
                a_res[xi] = a_res[xi] * a_u(a, x[xi], phi, u[ei])
            end
        end

        pyx = zeros(length(x))
        for xj = 1:length(x)
            pyx[xj] = real(tr(rho0 * a_res[xj]' * a_res[xj])) * (1 / factorial(N))
        end
        M_res[ui] = abs(trapz(x, pyx .* p .* exp_ix))
    end
    return sum(M_res)
end

function a_u(a, x, phi, u)
    N = size(a)[1] - 1
    a_in = kron(a, Matrix(I, N + 1, N + 1))
    b_in = kron(Matrix(I, N + 1, N + 1), a)

    value = 0.5 * (x - phi) + 0.5 * pi * u
    return a_in * sin(value) + b_in * cos(value)
end

function logarithmic(number, N)
    res = zeros(N)
    res_tp = number
    for i = 1:N
        res_tp = res_tp / 2
        res[i] = res_tp
    end
    return res
end

function DE_DeltaPhiOpt(
    x,
    p,
    rho0,
    a,
    comb,
    popsize,
    ini_population,
    c,
    cr,
    seed,
    max_episode,
    eps,
)
    Random.seed!(seed)
    N = size(a)[1] - 1
    DeltaPhi = [zeros(N) for i = 1:popsize]
    # initialize
    res = logarithmic(2.0 * pi, N)
    if length(ini_population) > popsize
        ini_population = [ini_population[i] for i = 1:popsize]
    end
    for pj = 1:length(ini_population)
        DeltaPhi[pj] = [ini_population[pj][i] for i = 1:N]
    end
    for pk = (length(ini_population)+1):popsize
        DeltaPhi[pk] = [res[i] + rand() for i = 1:N]
    end

    p_fit = [0.0 for i = 1:popsize]
    for pl = 1:N
        p_fit[pl] = adaptMZI_offline(DeltaPhi[pl], x, p, rho0, a, comb, eps)
    end

    f_ini = maximum(p_fit)
    f_list = [f_ini]
    for ei = 1:(max_episode-1)
        for pm = 1:popsize
            #mutations
            mut_num = sample(1:popsize, 3, replace = false)
            DeltaPhi_mut = [0.0 for i = 1:N]
            for ci = 1:N
                DeltaPhi_mut[ci] =
                    DeltaPhi[mut_num[1]][ci] +
                    c * (DeltaPhi[mut_num[2]][ci] - DeltaPhi[mut_num[3]][ci])
            end
            #crossover
            DeltaPhi_cross = [0.0 for i = 1:N]
            cross_int = sample(1:N, 1, replace = false)[1]
            for cj = 1:N
                rand_num = rand()
                if rand_num <= cr
                    DeltaPhi_cross[cj] = DeltaPhi_mut[cj]
                else
                    DeltaPhi_cross[cj] = DeltaPhi[pm][cj]
                end
                DeltaPhi_cross[cross_int] = DeltaPhi_mut[cross_int]
            end
            #selection
            for cm = 1:N
                DeltaPhi_cross[cm] =
                    (x -> x < 0.0 ? 0.0 : x > pi ? pi : x)(DeltaPhi_cross[cm])
            end
            f_cross = adaptMZI_offline(DeltaPhi_cross, x, p, rho0, a, comb, eps)
            if f_cross > p_fit[pm]
                p_fit[pm] = f_cross
                for ck = 1:N
                    DeltaPhi[pm][ck] = DeltaPhi_cross[ck]
                end
            end
        end
        println(maximum(p_fit))
        println(DeltaPhi[findmax(p_fit)[2]])
        append!(f_list, maximum(p_fit))
    end
    return DeltaPhi[findmax(p_fit)[2]]
end

function PSO_DeltaPhiOpt(
    x,
    p,
    rho0,
    a,
    comb,
    particle_num,
    ini_particle,
    c0,
    c1,
    c2,
    seed,
    max_episode,
    eps,
)
    Random.seed!(seed)
    N = size(a)[1] - 1
    n = size(a)[1]

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    DeltaPhi = [zeros(N) for i = 1:particle_num]
    velocity = [zeros(N) for i = 1:particle_num]
    # initialize
    res = logarithmic(2.0 * pi, N)
    if length(ini_particle) > particle_num
        ini_particle = [ini_particle[i] for i = 1:particle_num]
    end
    for pj = 1:length(ini_particle)
        DeltaPhi[pj] = [ini_particle[pj][i] for i = 1:N]
    end
    for pk = (length(ini_particle)+1):particle_num
        DeltaPhi[pk] = [res[i] + rand() for i = 1:N]
    end
    for pl = 1:particle_num
        velocity[pl] = [0.1 * rand() for i = 1:N]
    end

    pbest = [zeros(N) for i = 1:particle_num]
    gbest = zeros(N)
    fit = 0.0
    p_fit = [0.0 for i = 1:particle_num]
    f_list = []
    for ei = 1:(max_episode[1]-1)
        for pm = 1:particle_num
            f_now = adaptMZI_offline(DeltaPhi[pm], x, p, rho0, a, comb, eps)
            if f_now > p_fit[pm]
                p_fit[pm] = f_now
                for ci = 1:N
                    pbest[pm][ci] = DeltaPhi[pm][ci]
                end
            end
        end

        for pn = 1:particle_num
            if p_fit[pn] > fit
                fit = p_fit[pn]
                for cj = 1:N
                    gbest[cj] = pbest[pn][cj]
                end
            end
        end

        for pa = 1:particle_num
            DeltaPhi_pre = [0.0 for i = 1:N]
            for ck = 1:N
                DeltaPhi_pre[ck] = DeltaPhi[pa][ck]
                velocity[pa][ck] =
                    c0 * velocity[pa][ck] + c1 * rand() * (pbest[pa][ck] - DeltaPhi[pa][ck])
                +c2 * rand() * (gbest[ck] - DeltaPhi[pa][ck])
                DeltaPhi[pa][ck] += velocity[pa][ck]
            end

            for cn = 1:N
                DeltaPhi[pa][cn] = (x -> x < 0.0 ? 0.0 : x > pi ? pi : x)(DeltaPhi[pa][cn])
                velocity[pa][cn] = DeltaPhi[pa][cn] - DeltaPhi_pre[cn]
            end
        end
        println(fit)
        println(gbest)
        append!(f_list, fit)
        if ei % max_episode[2] == 0
            for pb = 1:particle_num
                DeltaPhi[pb] = [gbest[i] for i = 1:N]
            end
        end
    end
    return gbest
end
