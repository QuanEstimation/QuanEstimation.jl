struct AdaptiveStrategy <: EstimationStrategy
    x::Any #ParameterRegion
    p::Any #PriorDistribution
    online::Bool
end

AdaptiveStrategy(x::AbstractVector, p::AbstractArray; online = true) =
    AdaptiveStrategy(x, p, online)

function adapt_param!(scheme::Scheme{S,P,M,AdaptiveStrategy}, x) where {S,P,M}
    scheme.Parameterization.params = [x...]
end

function adapt_scheme!(
    scheme::Scheme{ST,PT,MT,AdaptiveStrategy},
    x_list,
    M,
    W,
    eps,
) where {ST,PT,MT}
    ρ_all, F_all = Matrix{ComplexF64}[], Float64[]
    for x in x_list
        adapt_param!(scheme, x)
        ρ, dρ = evolve(scheme)
        F_tp = CFIM(ρ, dρ, M; eps = eps)
        append!(F_all, abs(det(F_tp)) < eps ? eps : 1.0 / real(tr(W * inv(F_tp))))
        append!(ρ_all, [ρ])
    end
    return ρ_all, F_all
end

function adapt!(
    scheme::Scheme{ST,PT,MT,AdaptiveStrategy};
    method = "FOP",
    savefile = false,
    max_episode::Int = 1000,
    W = nothing,
    eps = GLOBAL_EPS,
) where {ST,PT,MT}
    (; x, p) = strat_data(scheme)
    M = meas_data(scheme)
    para_num = length(x)
    if isnothing(W)
        W = I(para_num)
    end

    p_num = length(p |> vec)

    x_list = [(Iterators.product(x...))...]
    rho_all, F_all = adapt_scheme!(scheme, x_list, M, W, eps)
    rho_all = reshape(rho_all, size(p))
    u = zeros(para_num)

    if method == "FOP"
        F = reshape(F_all, size(p))
        idx = findmax(F)[2]
        x_opt = [x[i][idx[i]] for i = 1:para_num]
        println("The optimal parameter are $x_opt")
        if savefile == false
            y, xout = Int64[], Float64[]
            for ei = 1:max_episode
                p, x_out, res_exp, u = iter_FOP_multipara(
                    p,
                    p_num,
                    para_num,
                    x,
                    x_list,
                    u,
                    rho_all,
                    M,
                    x_opt,
                    ei,
                )
                append!(xout, [x_out])
                append!(y, Int(res_exp + 1))
            end
            savefile_false(p, xout, y)
        else
            for ei = 1:max_episode
                p, x_out, res_exp, u =
                    iter_FOP(p, p_num, para_num, x, x_list, u, rho_all, M, x_opt, ei)
                savefile_true(p, x_out, Int(res_exp + 1))
            end
        end
    elseif method == "MI"
        if savefile == false
            y, xout = Int64[], Float64[]
            for ei = 1:max_episode
                p, x_out, res_exp, u =
                    iter_MI(p, p_num, para_num, x, x_list, u, rho_all, M, ei)
                append!(xout, [x_out])
                append!(y, Int(res_exp + 1))
            end
            savefile_false(p, xout, y)
        else
            for ei = 1:max_episode
                p, x_out, res_exp, u =
                    iter_MI(p, p_num, para_num, x, x_list, u, rho_all, M, ei)
                savefile_true(p, x_out, Int(res_exp + 1))
            end
        end
    end
end

function iter_FOP(p, p_num, para_num, x, x_list, u, rho_all, M, x_opt, ei)
    rho = Array{Matrix{ComplexF64}}(undef, p_num)
    for hj = 1:p_num
        x_idx = [findmin(abs.(x[k] .- (x_list[hj][k] + u[k])))[2] for k = 1:para_num]
        rho[hj] = rho_all[x_idx...]
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
    p_update = (p .* pyx / py) |> vec

    p_list = p |> vec
    arr = zeros(p_num)
    for i = 1:p_num
        res =
            [x_list[1][ri] < (x_list[i][ri] + u[ri]) < x_list[end][ri] for ri = 1:para_num]
        if all(res)
            p_list[i] = p_update[i]
        else
            p_list[i] = 0.0
        end
    end
    p = reshape(p_list, size(p))

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
    return p, x_out, res_exp, u
end


function iter_MI(p, p_num, para_num, x, x_list, u, rho_all, M, ei)
    rho = Array{Matrix{ComplexF64}}(undef, p_num)
    for hj = 1:p_num
        x_idx = [findmin(abs.(x[k] .- (x_list[hj][k] + u[k])))[2] for k = 1:para_num]
        rho[hj] = rho_all[x_idx...]
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

    p_list = p |> vec
    arr = zeros(p_num)
    for i = 1:p_num
        res =
            [x_list[1][ri] < (x_list[i][ri] + u[ri]) < x_list[end][ri] for ri = 1:para_num]
        if all(res)
            p_list[i] = p_update[i]
        else
            p_list[i] = 0.0
        end
    end
    p = reshape(p_list, size(p))

    p_idx = findmax(p)[2]
    x_out = [x[i][p_idx[i]] for i = 1:para_num]
    println("The estimator are $x_out ($ei episodes)")

    MI = zeros(p_num)
    for ui = 1:p_num
        rho_u = Array{Matrix{ComplexF64}}(undef, p_num)
        for hj = 1:p_num
            x_idx = [
                findmin(abs.(x[k] .- (x_list[hj][k] + x_list[ui][k])))[2] for
                k = 1:para_num
            ]
            rho_u[hj] = rho_all[x_idx...]
        end

        value_tp = zeros(size(p))
        for mi in eachindex(M)
            pyx_list_tp = real.(tr.(rho_u .* [M[mi]]))
            pyx_tp = reshape(pyx_list, size(p))
            mean_tp = trapz(tuple(x...), p .* pyx_tp)
            value_tp += pyx_tp .* log.(2, pyx_tp / mean_tp)
        end

        # value_int = trapz(tuple(x...), p.*value_tp)

        arr = zeros(p_num)
        for hj = 1:p_num
            res = [
                x_list[1][ri] < (x_list[hj][ri] + x_list[ui][ri]) < x_list[end][ri] for
                ri = 1:para_num
            ]
            if all(res)
                arr[hj] = vec(p)[hj] * vec(value_tp)[hj]
            end
        end
        value_int = trapz(tuple(x...), reshape(arr, size(p)))

        MI[ui] = value_int
    end
    p_idx = findmax(reshape(MI, size(p)))[2]
    u = [x[i][p_idx[i]] for i = 1:para_num]

    if mod(ei, 50) == 0
        for un = 1:para_num
            if (x_out[un] + u[un]) > x[un][end] || (x_out[un] + u[un]) < x[un][1]
                throw("Please increase the regime of the parameters.")
            end
        end
    end
    return p, x_out, res_exp, u
end

function savefile_true(p, xout, y)
    fp = isfile("adaptive.dat") ? load("adaptive.dat")["p"] : []
    fx = isfile("adaptive.dat") ? load("adaptive.dat")["x"] : []
    fy = isfile("adaptive.dat") ? load("adaptive.dat")["y"] : []
    jldopen("adaptive.dat", "w") do f
        f["p"] = append!(fp, [p])
        f["x"] = append!(fx, [xout])
        f["y"] = append!(fy, [y])
    end
end

function savefile_false(p, xout, y)
    jldopen("adaptive.dat", "w") do f
        f["p"] = [p]
        f["x"] = xout
        f["y"] = y
    end
end

include("AdptiveMZI.jl")
