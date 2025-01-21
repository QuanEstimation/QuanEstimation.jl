mutable struct Adapt_MZI <: AbstractScheme
    x::AbstractVector
    p::AbstractVector
    rho0::AbstractMatrix
end

abstract type MIZtargetType end
abstract type sharpness <: MIZtargetType end
abstract type MI <: MIZtargetType end

struct calculate_online{P} end
struct calculate_offline{P} end

##========== online ==========##
@doc raw"""

    online(apt::Adapt_MZI; target::Symbol=:sharpness, output::String="phi")

Online adaptive phase estimation in the MZI.
- `apt`: Adaptive MZI struct which contains x, p, and rho0.
- `target`: Setting the target function for calculating the tunable phase. Options are: "sharpness" and "MI".
- `output`: Choose the output variables. Options are: "phi" and "dphi".
"""
function online(apt::Adapt_MZI; target::Symbol = :sharpness, output::String = "phi", res=nothing)
    (; x, p, rho0) = apt
    adaptMZI_online(x, p, rho0, Symbol(output), target; res=res)
end

function adaptMZI_online(x, p, rho0, output, target::Symbol; res=nothing)
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1) |> sparse
    exp_ix = [exp(1.0im * xi) for xi in x]
    phi_span = range(-pi, stop = pi, length = length(x)) |> collect

    phi = 0.0
    a_res = [Matrix{ComplexF64}(I, (N + 1)^2, (N + 1)^2) for i in eachindex(x)]

    xout, y = [], []

    if output == :phi
        for ei = 1:N-1
            if isnothing(res)
                println("The tunable phase is $phi ($ei episodes)")
                print("Please enter the experimental result: ")
                enter = readline()
                u = parse(Int64, enter)
            else
                u = Int64(res[ei])
            end
            pyx = zeros(length(x)) |> sparse
            for xi in eachindex(x)
                a_res_tp = a_res[xi] * a_u(a, x[xi], phi, u)
                pyx[xi] =
                    real(tr(rho0 * a_res_tp' * a_res_tp)) *
                    (factorial(N - ei) / factorial(N))
                a_res[xi] = a_res_tp
            end
            phi_update = calculate_online{eval(target)}(
                x,
                p,
                pyx,
                a_res,
                a,
                rho0,
                N,
                ei,
                phi_span,
                exp_ix,
            )

            append!(xout, phi)
            append!(y, u)
            phi = phi_update
        end
        println("The estimator of the unknown phase is $phi ")
        append!(xout, phi)
        savefile_online(xout, y)
    else
        println("The initial tunable phase is $phi")
        for ei = 1:N-1
            if isnothing(res)
                println("The tunable phase is $phi ($ei episodes)")
                print("Please enter the experimental result: ")
                enter = readline()
                u = parse(Int64, enter)
            else
                u = Int64(res[ei])
            end

            pyx = zeros(length(x)) |> sparse
            for xi in eachindex(x)
                a_res_tp = a_res[xi] * a_u(a, x[xi], phi, u)
                pyx[xi] =
                    real(tr(rho0 * a_res_tp' * a_res_tp)) *
                    (factorial(N - ei) / factorial(N))
                a_res[xi] = a_res_tp
            end

            phi_update = calculate_online{eval(target)}(
                x,
                p,
                pyx,
                a_res,
                a,
                rho0,
                N,
                ei,
                phi_span,
                exp_ix,
            )

            println(
                "The adjustments of the feedback phase is $(abs(phi_update-phi)) ($ei episodes)",
            )
            append!(xout, abs(phi_update - phi))
            append!(y, u)
            phi = phi_update
        end
        savefile_online(xout, y)
    end
end

adaptMZI_online(x, p, rho0, output::String, target::String) =
    adaptMZI_online(x, p, rho0, Symbol(output), Symbol(target))

function calculate_online{sharpness}(x, p, pyx, a_res, a, rho0, N, ei, phi_span, exp_ix)

    M_res = zeros(length(phi_span))
    for mj in eachindex(phi_span)
        M1_res = trapz(x, pyx .* p)
        pyx0, pyx1 = zeros(length(x)), zeros(length(x))
        M2_res = 0.0
        for xj in eachindex(x)
            a_res0 = a_res[xj] * a_u(a, x[xj], phi_span[mj], 0)
            a_res1 = a_res[xj] * a_u(a, x[xj], phi_span[mj], 1)
            pyx0[xj] =
                real(tr(rho0 * a_res0' * a_res0)) * (factorial(N - (ei + 1)) / factorial(N))
            pyx1[xj] =
                real(tr(rho0 * a_res1' * a_res1)) * (factorial(N - (ei + 1)) / factorial(N))
            M2_res = abs(trapz(x, pyx0 .* p .* exp_ix)) + abs(trapz(x, pyx1 .* p .* exp_ix))
        end
        M_res[mj] = M2_res / M1_res
    end
    indx_m = findmax(M_res)[2]
    phi_span[indx_m]
end

function calculate_online{MI}(x, p, pyx, a_res, a, rho0, N, ei, phi_span, exp_ix)

    M_res = zeros(length(phi_span))
    for mj in eachindex(phi_span)
        M1_res = trapz(x, pyx .* p)
        pyx0, pyx1 = zeros(length(x)), zeros(length(x))
        M2_res = 0.0
        for xj in eachindex(x)
            a_res0 = a_res[xj] * a_u(a, x[xj], phi_span[mj], 0)
            a_res1 = a_res[xj] * a_u(a, x[xj], phi_span[mj], 1)
            pyx0[xj] =
                real(tr(rho0 * a_res0' * a_res0)) * (factorial(N - (ei + 1)) / factorial(N))
            pyx1[xj] =
                real(tr(rho0 * a_res1' * a_res1)) * (factorial(N - (ei + 1)) / factorial(N))
            M2_res =
                trapz(x, pyx0 .* p .* log.(2, pyx0 ./ trapz(x, pyx0 .* p))) +
                trapz(x, pyx1 .* p .* log.(2, pyx1 ./ trapz(x, pyx1 .* p)))
        end
        M_res[mj] = M2_res / M1_res
    end
    indx_m = findmax(M_res)[2]
    phi_span[indx_m]
end

function savefile_online(xout, y)
    jldopen("adaptive.dat", "w") do f
        f["x"] = xout
        f["y"] = y
    end
end

##========== offline ==========##
@doc raw"""

    offline(apt::Adapt_MZI, alg; target::Symbol=:sharpness, eps = GLOBAL_EPS, seed=1234)

Offline adaptive phase estimation in the MZI.
- `apt`: Adaptive MZI struct which contains `x`, `p`, and `rho0`.
- `alg`: The algorithms for searching the optimal tunable phase. Here, DE and PSO are available. 
- `target`: Setting the target function for calculating the tunable phase. Options are: "sharpness" and "MI".
- `eps`: Machine epsilon.
- `seed`: Random seed.
"""
function offline(
    apt::Adapt_MZI,
    alg;
    target::Symbol = :sharpness,
    eps = GLOBAL_EPS,
    seed = 1234,
)
    rng = MersenneTwister(seed)
    (; x, p, rho0) = apt
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1) |> sparse
    comb = brgd(N) |> x -> [[parse(Int, s) for s in ss] for ss in x]
    if alg isa DE
        (; p_num, ini_population, c, cr, max_episode) = alg
        if isnothing(ini_population)
            ini_population = ([apt.rho0],)
        end
        DE_deltaphiOpt(
            x,
            p,
            rho0,
            comb,
            p_num,
            ini_population[1],
            c,
            cr,
            rng,
            max_episode,
            target,
            eps,
        )
    elseif alg isa PSO
        (; p_num, ini_particle, c0, c1, c2, max_episode) = alg
        if isnothing(ini_particle)
            ini_particle = ([apt.rho0],)
        end
        PSO_deltaphiOpt(
            x,
            p,
            rho0,
            comb,
            p_num,
            ini_particle[1],
            c0,
            c1,
            c2,
            rng,
            max_episode,
            target,
            eps,
        )
    end
end

function DE_deltaphiOpt(
    x,
    p,
    rho0,
    comb,
    p_num,
    ini_population,
    c,
    cr,
    rng::AbstractRNG,
    max_episode,
    target::Symbol,
    eps,
)
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1) |> sparse
    deltaphi = [zeros(N) for i = 1:p_num]
    # initialize
    res = logarithmic(2.0 * pi, N)
    if length(ini_population) > p_num
        ini_population = [ini_population[i] for i = 1:p_num]
    end
    for pj in eachindex(ini_population)
        deltaphi[pj] = [ini_population[pj][i] for i = 1:N]
    end
    for pk = (length(ini_population)+1):p_num
        deltaphi[pk] = [res[i] + rand(rng) for i = 1:N]
    end

    p_fit = [0.0 for i = 1:p_num]
    for pl = 1:N
        p_fit[pl] = calculate_offline{eval(target)}(deltaphi[pl], x, p, rho0, a, comb, eps)
    end

    f_ini = maximum(p_fit)
    f_list = [f_ini]
    for ei = 1:(max_episode-1)
        for pm = 1:p_num
            #mutations
            mut_num = sample(rng, 1:p_num, 3, replace = false)
            deltaphi_mut = [0.0 for i = 1:N]
            for ci = 1:N
                deltaphi_mut[ci] =
                    deltaphi[mut_num[1]][ci] +
                    c * (deltaphi[mut_num[2]][ci] - deltaphi[mut_num[3]][ci])
            end
            #crossover
            deltaphi_cross = [0.0 for i = 1:N]
            cross_int = sample(rng, 1:N, 1, replace = false)[1]
            for cj = 1:N
                rand_num = rand(rng)
                if rand_num <= cr
                    deltaphi_cross[cj] = deltaphi_mut[cj]
                else
                    deltaphi_cross[cj] = deltaphi[pm][cj]
                end
                deltaphi_cross[cross_int] = deltaphi_mut[cross_int]
            end
            #selection
            for cm = 1:N
                deltaphi_cross[cm] =
                    (x -> x < 0.0 ? 0.0 : x > pi ? pi : x)(deltaphi_cross[cm])
            end
            f_cross =
                calculate_offline{eval(target)}(deltaphi_cross, x, p, rho0, a, comb, eps)
            if f_cross > p_fit[pm]
                p_fit[pm] = f_cross
                for ck = 1:N
                    deltaphi[pm][ck] = deltaphi_cross[ck]
                end
            end
        end
        append!(f_list, maximum(p_fit))
    end
    savefile_offline(deltaphi[findmax(p_fit)[2]], f_list)
    return deltaphi[findmax(p_fit)[2]]
end

DE_deltaphiOpt(
    x,
    p,
    rho0,
    comb,
    p_num,
    ini_population,
    c,
    cr,
    seed::Number,
    max_episode,
    target::String,
    eps,
) = DE_deltaphiOpt(
    x,
    p,
    rho0,
    comb,
    p_num,
    ini_population,
    c,
    cr,
    MersenneTwister(seed),
    max_episode,
    Symbol(target),
    eps,
)

function PSO_deltaphiOpt(
    x,
    p,
    rho0,
    comb,
    p_num,
    ini_particle,
    c0,
    c1,
    c2,
    rng::AbstractRNG,
    max_episode,
    target::Symbol,
    eps,
)
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1) |> sparse
    n = size(a)[1]

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    deltaphi = [zeros(N) for i = 1:p_num]
    velocity = [zeros(N) for i = 1:p_num]
    # initialize
    res = logarithmic(2.0 * pi, N)
    if length(ini_particle) > p_num
        ini_particle = [ini_particle[i] for i = 1:p_num]
    end
    for pj in eachindex(ini_particle)
        deltaphi[pj] = [ini_particle[pj][i] for i = 1:N]
    end
    for pk = (length(ini_particle)+1):p_num
        deltaphi[pk] = [res[i] + rand(rng) for i = 1:N]
    end
    for pl = 1:p_num
        velocity[pl] = [0.1 * rand(rng) for i = 1:N]
    end

    pbest = [zeros(N) for i = 1:p_num]
    gbest = zeros(N)
    fit = 0.0
    p_fit = [0.0 for i = 1:p_num]
    f_list = []
    for ei = 1:(max_episode[1]-1)
        for pm = 1:p_num
            f_now = calculate_offline{eval(target)}(deltaphi[pm], x, p, rho0, a, comb, eps)
            if f_now > p_fit[pm]
                p_fit[pm] = f_now
                for ci = 1:N
                    pbest[pm][ci] = deltaphi[pm][ci]
                end
            end
        end

        for pn = 1:p_num
            if p_fit[pn] > fit
                fit = p_fit[pn]
                for cj = 1:N
                    gbest[cj] = pbest[pn][cj]
                end
            end
        end

        for pa = 1:p_num
            deltaphi_pre = [0.0 for i = 1:N]
            for ck = 1:N
                deltaphi_pre[ck] = deltaphi[pa][ck]
                velocity[pa][ck] =
                    c0 * velocity[pa][ck] +
                    c1 * rand(rng) * (pbest[pa][ck] - deltaphi[pa][ck])
                +c2 * rand(rng) * (gbest[ck] - deltaphi[pa][ck])
                deltaphi[pa][ck] += velocity[pa][ck]
            end

            for cn = 1:N
                deltaphi[pa][cn] = (x -> x < 0.0 ? 0.0 : x > pi ? pi : x)(deltaphi[pa][cn])
                velocity[pa][cn] = deltaphi[pa][cn] - deltaphi_pre[cn]
            end
        end
        append!(f_list, fit)
        if ei % max_episode[2] == 0
            for pb = 1:p_num
                deltaphi[pb] = [gbest[i] for i = 1:N]
            end
        end
    end
    savefile_offline(gbest, f_list)
    return gbest
end

PSO_deltaphiOpt(
    x,
    p,
    rho0,
    comb,
    p_num,
    ini_particle,
    c0,
    c1,
    c2,
    seed::Number,
    max_episode,
    target::String,
    eps,
) = PSO_deltaphiOpt(
    x,
    p,
    rho0,
    comb,
    p_num,
    ini_particle,
    c0,
    c1,
    c2,
    MersenneTwister(seed),
    max_episode,
    Symbol(target),
    eps,
)

function calculate_offline{sharpness}(delta_phi, x, p, rho0, a, comb, eps)
    N = size(a)[1] - 1
    exp_ix = [exp(1.0im * xi) for xi in x]

    M_res = zeros(length(comb))
    for ui in eachindex(comb)
        u = comb[ui]
        phi = 0.0

        a_res = [Matrix{ComplexF64}(I, (N + 1)^2, (N + 1)^2) for i in eachindex(x)]
        for ei = 1:N-1
            phi = phi - (-1)^u[ei] * delta_phi[ei]
            for xi in eachindex(x)
                a_res[xi] = a_res[xi] * a_u(a, x[xi], phi, u[ei])
            end
        end

        pyx = zeros(length(x))
        for xj in eachindex(x)
            pyx[xj] = real(tr(rho0 * a_res[xj]' * a_res[xj])) * (1 / factorial(N))
        end
        M_res[ui] = abs(trapz(x, pyx .* p .* exp_ix))
    end
    return sum(M_res)
end

function calculate_offline{MI}(delta_phi, x, p, rho0, a, comb, eps)
    N = size(a)[1] - 1
    exp_ix = [exp(1.0im * xi) for xi in x]

    M_res = zeros(length(comb))
    for ui in eachindex(comb)
        u = comb[ui]
        phi = 0.0

        a_res = [Matrix{ComplexF64}(I, (N + 1)^2, (N + 1)^2) for i in eachindex(x)]
        for ei = 1:N-1
            phi = phi - (-1)^u[ei] * delta_phi[ei]
            for xi in eachindex(x)
                a_res[xi] = a_res[xi] * a_u(a, x[xi], phi, u[ei])
            end
        end

        pyx = zeros(length(x))
        for xj in eachindex(x)
            pyx[xj] = real(tr(rho0 * a_res[xj]' * a_res[xj])) * (1 / factorial(N))
        end
        M_res[ui] = trapz(x, pyx .* p .* log.(2, pyx ./ trapz(x, pyx .* p)))
    end
    return sum(M_res)
end

function savefile_offline(deltaphi, flist)
    open("deltaphi.csv", "w") do m
        writedlm(m, deltaphi)
    end
    open("f.csv", "w") do n
        writedlm(n, flist)
    end
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


function adapt!(scheme::Adapt_MZI; target::Symbol = :sharpness, output::String = "phi")
    return online(scheme, target = target, output = output)
end

function adapt!(
    scheme::Adapt_MZI,
    alg;
    target::Symbol = :sharpness,
    eps = GLOBAL_EPS,
    seed = 1234,
)
    return offline(scheme, alg, target = target, eps = eps, seed = seed)
end
