@doc raw"""

    Adapt(x::AbstractVector, p, rho0::AbstractMatrix, tspan, H, dH; dyn_method=:Expm, method="FOP", savefile=false, max_episode::Int=1000, eps::Float64=1e-8, Hc=missing, ctrl=missing, decay=missing, M=missing, W=missing)

In QuanEstimation, the Hamiltonian of the adaptive system should be written as
``H(\textbf{x}+\textbf{u})`` with ``\textbf{x}`` the unknown parameters and ``\textbf{u}``
the tunable parameters. The tunable parameters ``\textbf{u}`` are used to let the 
Hamiltonian work at the optimal point ``\textbf{x}_{\mathrm{opt}}``. 
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho0`: Density matrix.
- `tspan`: The experimental results obtained in practice.
- `H`: Free Hamiltonian with respect to the values in x.
- `dH`: Derivatives of the free Hamiltonian with respect to the unknown parameters to be estimated.
- `dyn_method`: Setting the method for solving the Lindblad dynamics. Options are: "expm" and "ode".
- `method`: Choose the method for updating the tunable parameters (u). Options are: "FOP" and "MI".
- `savefile`: Whether or not to save all the posterior distributions. 
- `max_episode`: The number of episodes.
- `eps`: Machine epsilon.
- `Hc`: Control Hamiltonians.
- `ctrl`: Control coefficients.
- `decay`: Decay operators and the corresponding decay rates.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `W`: Whether or not to save all the posterior distributions. 
"""
function Adapt(x::AbstractVector, p, rho0::AbstractMatrix, tspan, H, dH; dyn_method=:Expm, method="FOP", savefile=false, max_episode::Int=1000, eps::Float64=1e-8, 
                  Hc=missing, ctrl=missing, decay=missing, M=missing, W=missing)
    dim = size(rho0)[1]
    rho0 = complex.(rho0)
    para_num = length(x)
    if ismissing(M)
        M = SIC(size(rho0)[1])
    end
    if ismissing(decay)
        decay_opt = [zeros(ComplexF64, dim, dim)]
        gamma = [0.0]
    else
        decay_opt = [decay[i][1] for i in 1:len(decay)]
        gamma = [decay[i][2] for i in 1:len(decay)]
    end
    if ismissing(Hc)
        Hc = [zeros(ComplexF64, dim, dim)]
        ctrl = [zeros(length(tspan)-1)]
    elseif ismissing(ctrl)
        ctrl = [zeros(length(tspan)-1) for j in range(length(Hc))]
    else
        ctrl_length = length(ctrl)
        ctrlnum = length(Hc)
        if ctrlnum < ctrl_length
            throw("There are $ctrlnum control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences")
        elseif ctrlnum > ctrl_length 
            println("Not enough coefficients sequences: there are $ctrlnum control Hamiltonians 
                     but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.")
            number = ceil((length(tspan)-1)/length(ctrl[1]))
            if mod(length(tspan)-1, length(ctrl[1])) != 0
                tnum = number*length(ctrl[1])
                tspan = arange(tspan[1], stop=tspan[end], length=tnum+1) |> collect
            end
        end
    end
    if ismissing(W)
        W = zeros(para_num, para_num)
    end    

    if para_num == 1
        #### singleparameter senario ####
        p_num = length(p)

        F = zeros(p_num)
        rho_all = []
        for hi in 1:p_num
            # dynamics = Lindblad_noisy_controlled{dyn_method}(H[hi], dH[hi], rho0, tspan, decay_opt, gamma, Hc, ctrl)
            dynamics = Lindblad(H[hi], dH[hi], Hc, ctrl, rho0, tspan, decay_opt, gamma, dyn_method=dyn_method)
            rho_tp, drho_tp = evolve(dynamics)
            F[hi] = CFIM(rho_tp, drho_tp[1], M; eps=eps)
            append!(rho_all, [rho_tp])
        end
        
        u = 0.0
        if method == "FOP"
            idx = findmax(F)[2]
            x_opt = x[1][idx]
            println("The optimal parameter is $x_opt")
            if savefile == false
                y, xout = [], []
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
                    append!(xout, x_out)
                    append!(y, Int(res_exp-1))
                end
                savefile_false(p, xout, y)
            else
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
                    savefile_true(p, x_out, Int(res_exp-1))
                end
            end
        elseif method == "MI"
            if savefile == false
                y, xout = [], []
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
                    append!(xout, x_out)
                    append!(y, Int(res_exp-1))
                end
                savefile_false(p, xout, y)
            else
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
                    savefile_true(p, x_out, Int(res_exp-1))
                end
            end
        end
    else
        #### multiparameter senario ####
        p_num = length(p|>vec)
        x_list = [(Iterators.product(x...))...]
        # dynamics_res = [evolve(Lindblad_noisy_controlled{dyn_method}(H_tp, dH_tp, rho0, tspan, decay_opt, gamma, Hc, ctrl)) for (H_tp, dH_tp) in zip(H, dH)]
        dynamics_res = [evolve(Lindblad(H_tp, dH_tp, Hc, ctrl, rho0, tspan, decay_opt, gamma, dyn_method=dyn_method)) for (H_tp, dH_tp) in zip(H, dH)]
        F_all = zeros(p_num)
        rho_all_list = []
        for hi in 1:p_num
            F_tp = CFIM(dynamics_res[hi][1], dynamics_res[hi][2], M; eps=eps)
            F_all[hi] = abs(det(F_tp)) < eps ? eps : 1.0/real(tr(W*inv(F_tp)))
            append!(rho_all_list, [dynamics_res[hi][1]])
        end
        rho_all = reshape(rho_all_list, size(p))
        
        u = [0.0 for i in 1:para_num]

        if method == "FOP"
            F = reshape(F_all, size(p))
            idx = findmax(F)[2]
            x_opt = [x[i][idx[i]] for i in 1:para_num]
            println("The optimal parameter are $x_opt")
            if savefile == false
                y, xout = [], []
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, x_opt, ei)
                    append!(xout, [x_out])
                    append!(y, Int(res_exp+1))
                end
                savefile_false(p, xout, y)
            else
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, x_opt, ei)
                    savefile_true(p, x_out, Int(res_exp+1))
                end
            end
        elseif method == "MI"
            if savefile == false
                y, xout = [], []
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, ei)
                    append!(xout, [x_out])
                    append!(y, Int(res_exp+1))
                end
                savefile_false(p, xout, y)
            else
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, ei)
                    savefile_true(p, x_out, Int(res_exp+1))
                end
            end
        end
    end
end

@doc raw"""

    Adapt(x::AbstractVector, p, rho0::AbstractMatrix, K, dK; method="FOP", savefile=false, max_episode::Int=1000, eps::Float64=1e-8, M=missing, W=missing)

In QuanEstimation, the Hamiltonian of the adaptive system should be written as
``H(\textbf{x}+\textbf{u})`` with ``\textbf{x}`` the unknown parameters and ``\textbf{u}``
the tunable parameters. The tunable parameters ``\textbf{u}`` are used to let the 
Hamiltonian work at the optimal point ``\textbf{x}_{\mathrm{opt}}``. 
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho0`: Density matrix.
- `K`: Kraus operator(s) with respect to the values in x.
- `dK`: Derivatives of the Kraus operator(s) with respect to the unknown parameters to be estimated.
- `method`: Choose the method for updating the tunable parameters (u). Options are: "FOP" and "MI".
- `savefile`: Whether or not to save all the posterior distributions. 
- `max_episode`: The number of episodes.
- `eps`: Machine epsilon.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `W`: Whether or not to save all the posterior distributions. 
"""
function Adapt(x::AbstractVector, p, rho0::AbstractMatrix, K, dK; method="FOP", savefile=false, max_episode::Int=1000, 
    eps::Float64=1e-8, M=missing, W=missing)
    dim = size(rho0)[1]
    para_num = length(x)

    if ismissing(W)
        W = zeros(para_num, para_num)
    end
    if ismissing(M)
        M = SIC(size(rho0)[1])
    end
    
    if para_num == 1
        #### singleparameter senario ####
        p_num = length(p)
        F = zeros(p_num)
        rho_all = []
        for hi in 1:p_num
            rho_tp = sum([Ki*rho0*Ki' for Ki in K[hi]]) 
            drho_tp = [sum([dKi*rho0*Ki' + Ki*rho0*dKi' for (Ki,dKi) in zip(K[hi],dKj)]) for dKj in dK[hi]]
            F[hi] = CFIM(rho_tp, drho_tp[1], M; eps=eps)
            append!(rho_all, [rho_tp])
        end
        
        u = 0.0
        if method == "FOP"
            indx = findmax(F)[2]
            x_opt = x[1][indx]
            println("The optimal parameter is $x_opt")
            if savefile == false
                y, xout = [], []
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
                    append!(xout, x_out)
                    append!(y, Int(res_exp-1))
                end
                savefile_false(p, xout, y)
            else
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
                    savefile_true(p, x_out, Int(res_exp-1))
                end
            end
        elseif method == "MI"
            if savefile == false
                y, xout = [], []
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
                    append!(xout, x_out)
                    append!(y, Int(res_exp-1))
                end
                savefile_false(p, xout, y)
            else
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
                    savefile_true(p, x_out, Int(res_exp-1))
                end
            end
        end
    else
        #### multiparameter senario ####
        k_num = length(vec(K)[1])
        para_num = length(x)
        p_num = length(p |> vec)
        x_list = [(Iterators.product(x...))...]
        rho_tp = [sum([Ki*rho0*Ki' for Ki in K_tp]) for K_tp in K]
        drho_tp = [[sum([dKi*rho0*Ki' + Ki*rho0*dKi' for (Ki,dKi) in zip(K_tp,dKj)]) for dKj in 
                   [[dK_tp[i][j] for i in 1:k_num] for j in 1:para_num]] for (K_tp,dK_tp) in zip(K,dK)]

        F_all = zeros(p_num)
        for hi in 1:p_num
            F_tp = CFIM(rho_tp[hi], drho_tp[hi], M; eps=eps)
            F_all[hi] = abs(det(F_tp)) < eps ? eps : 1.0/real(tr(W*inv(F_tp)))
        end
        rho_all = reshape(rho_tp, size(p))
        
        u = [0.0 for i in 1:para_num]
        if method == "FOP"
            F = reshape(F_all, size(p))
            idx = findmax(F)[2]
            x_opt = [x[i][idx[i]] for i in 1:para_num]
            println("The optimal parameter are $x_opt")
            if savefile == false
                y, xout = [], []
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, x_opt, ei)
                    append!(xout, [x_out])
                    append!(y, Int(res_exp+1))
                end
                savefile_false(p, xout, y)
            else
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, x_opt, ei)
                    savefile_true(p, x_out, Int(res_exp+1))
                end
            end
        elseif method == "MI"
            if savefile == false
                y, xout = [], []
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, ei)
                    append!(xout, [x_out])
                    append!(y, Int(res_exp+1))
                end
                savefile_false(p, xout, y)
            else
                for ei in 1:max_episode
                    p, x_out, res_exp, u = iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, ei)
                    savefile_true(p, x_out, Int(res_exp+1))
                end
            end
        end
    end
end

function iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
    rho = [zeros(ComplexF64, dim, dim) for i in 1:p_num]
    for hj in 1:p_num
        x_idx = findmin(abs.(x[1] .- (x[1][hj]+u)))[2]
        rho[hj] = rho_all[x_idx]
    end
    println("The tunable parameter is $u")
    print("Please enter the experimental result: ")
    enter = readline()
    res_exp = parse(Int64, enter)
    res_exp = Int(res_exp+1)

    pyx = real.(tr.(rho .* [M[res_exp]]))
    py = trapz(x[1], pyx.*p)
    p_update = pyx.*p/py

    for i in 1:p_num
        if x[1][1] < x[1][i]+u < x[1][end]
            p[i] = p_update[i]
        else
            p[i] = 0.0
        end
    end

    p_idx = findmax(p)[2]
    x_out = x[1][p_idx]
    println("The estimator is $x_out ($ei episodes)")
    u = x_opt - x_out

    if mod(ei, 50) == 0
        if (x_out+u) > x[1][end] || (x_out+u) < x[1][1]
            throw("please increase the regime of the parameters.")
        end
    end
    return p, x_out, res_exp, u
end

function iter_FOP_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, x_opt, ei)
    rho = Array{Matrix{ComplexF64}}(undef, p_num)
    for hj in 1:p_num
        x_idx = [findmin(abs.(x[k] .- (x_list[hj][k]+u[k])))[2] for k in 1:para_num]
        rho[hj] = rho_all[x_idx...]
    end

    println("The tunable parameter are $u")
    print("Please enter the experimental result: ")
    enter = readline()
    res_exp = parse(Int64, enter)
    res_exp = Int(res_exp+1)

    pyx_list = real.(tr.(rho.*[M[res_exp]]))
    pyx = reshape(pyx_list, size(p))

    arr = p.*pyx
    py = trapz(tuple(x...), arr)
    p_update = (p.*pyx/py) |> vec
    
    p_list = p |> vec
    arr = zeros(p_num)
    for i in 1:p_num
        res = [x_list[1][ri] < (x_list[i][ri] + u[ri]) < x_list[end][ri] for ri in 1:para_num]
        if all(res)
            p_list[i] = p_update[i]
        else
            p_list[i] = 0.0
        end
    end
    p = reshape(p_list, size(p))

    p_idx = findmax(p)[2]
    x_out = [x[i][p_idx[i]] for i in 1:para_num]
    println("The estimator are $x_out ($ei episodes)")
    u = x_opt .- x_out

    if mod(ei, 50) == 0
        for un in 1:para_num
            if (x_out[un]+u[un]) > x[un][end] || (x_out[un]+u[un]) < x[un][1]
                throw("Please increase the regime of the parameters.")
            end
        end
    end
    return p, x_out, res_exp, u
end

function iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
    rho = [zeros(ComplexF64, dim, dim) for i in 1:p_num]
    for hj in 1:p_num
        x_idx = findmin(abs.(x[1] .- (x[1][hj]+u)))[2]
        rho[hj] = rho_all[x_idx]
    end

    println("The tunable parameter is $u")
    print("Please enter the experimental result: ")
    enter = readline()
    res_exp = parse(Int64, enter)
    res_exp = Int(res_exp+1)

    pyx = real.(tr.(rho .* [M[res_exp]]))

    py = trapz(x[1], pyx.*p)
    p_update = pyx.*p/py
    
    for i in 1:p_num
        if x[1][1] < x[1][i]+u < x[1][end]
            p[i] = p_update[i]
        else
            p[i] = 0.0
        end
    end

    p_idx = findmax(p)[2]
    x_out = x[1][p_idx]
    println("The estimator is $x_out ($ei episodes)")

    MI = zeros(p_num)
    for ui in 1:p_num
        rho_u = [zeros(ComplexF64, dim, dim) for i in 1:p_num]
        for hj in 1:p_num
            x_idx = findmin(abs.(x[1] .- (x[1][hj]+x[1][ui])))[2]
            rho_u[hj] = rho_all[x_idx]
        end
        value_tp = zeros(p_num)
        for mi in 1:length(M)
            pyx_tp = real.(tr.(rho_u .* [M[mi]]))
            mean_tp = trapz(x[1], pyx_tp.*p)
            value_tp += pyx_tp.*log.(2, pyx_tp/mean_tp)
        end

        # arr = [value_tp[i]*p[i] for i in range(p_num)]

        arr = zeros(p_num)
        for i in 1:p_num
            if x[1][1] < x[1][i]+x[1][ui] < x[1][end]
                arr[i] = value_tp[i]*p[i]
            end
        end
        MI[ui] = trapz(x[1], arr)
    end
    u = x[1][findmax(MI)[2]]

    if mod(ei, 50) == 0
        if (x_out+u) > x[1][end] || (x_out+u) < x[1][1]
            throw("please increase the regime of the parameters.")
        end
    end
    return p, x_out, res_exp, u
end

function iter_MI_multipara(p, p_num, para_num, x, x_list, u, rho_all, M, ei)
    rho = Array{Matrix{ComplexF64}}(undef, p_num)
    for hj in 1:p_num
        x_idx = [findmin(abs.(x[k] .- (x_list[hj][k]+u[k])))[2] for k in 1:para_num]
        rho[hj] = rho_all[x_idx...]
    end

    println("The tunable parameter are $u")
    print("Please enter the experimental result: ")
    enter = readline()
    res_exp = parse(Int64, enter)
    res_exp = Int(res_exp+1)

    pyx_list = real.(tr.(rho.*[M[res_exp]]))
    pyx = reshape(pyx_list, size(p))

    arr = p.*pyx
    py = trapz(tuple(x...), arr)
    p_update = p.*pyx/py
    
    p_list = p |> vec
    arr = zeros(p_num)
    for i in 1:p_num
        res = [x_list[1][ri] < (x_list[i][ri] + u[ri]) < x_list[end][ri] for ri in 1:para_num]
        if all(res)
            p_list[i] = p_update[i]
        else
            p_list[i] = 0.0
        end
    end
    p = reshape(p_list, size(p))

    p_idx = findmax(p)[2]
    x_out = [x[i][p_idx[i]] for i in 1:para_num]
    println("The estimator are $x_out ($ei episodes)")
    
    MI = zeros(p_num)
    for ui in 1:p_num
        rho_u = Array{Matrix{ComplexF64}}(undef, p_num)
        for hj in 1:p_num
            x_idx = [findmin(abs.(x[k] .- (x_list[hj][k]+x_list[ui][k])))[2] for k in 1:para_num]
            rho_u[hj] = rho_all[x_idx...]
        end

        value_tp = zeros(size(p))
        for mi in 1:length(M)
            pyx_list_tp = real.(tr.(rho_u.*[M[mi]]))
            pyx_tp = reshape(pyx_list, size(p))
            mean_tp = trapz(tuple(x...), p.*pyx_tp)
            value_tp += pyx_tp.*log.(2, pyx_tp/mean_tp)   
        end

        # value_int = trapz(tuple(x...), p.*value_tp)

        arr = zeros(p_num)
        for hj in 1:p_num
            res = [x_list[1][ri] < (x_list[hj][ri] + x_list[ui][ri]) < x_list[end][ri] for ri in 1:para_num]
            if all(res)
                arr[hj] = vec(p)[hj]*vec(value_tp)[hj]
            end
        end
        value_int = trapz(tuple(x...), reshape(arr, size(p)))

        MI[ui] = value_int
    end
    p_idx = findmax(reshape(MI, size(p)))[2]
    u = [x[i][p_idx[i]] for i in 1:para_num]

    if mod(ei, 50) == 0
        for un in 1:para_num
            if (x_out[un]+u[un]) > x[un][end] || (x_out[un]+u[un]) < x[un][1]
                throw("Please increase the regime of the parameters.")
            end
        end
    end
    return p, x_out, res_exp, u
end

function savefile_true(p, xout, y)
    open("pout.csv","w") do f
        writedlm(f, [p])
    end
    open("xout.csv","w") do m
        writedlm(m, [xout])
    end
    open("y.csv","w") do n
        writedlm(n, [y])
    end
end

function savefile_false(p, xout, y)
    open("pout.csv","w") do f
        writedlm(f, [p])
    end
    open("xout.csv","w") do m
        writedlm(m, xout)
    end
    open("y.csv","w") do n
        writedlm(n, y)
    end
end

mutable struct Adapt_MZI
    x
    p
    rho0
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
function online(apt::Adapt_MZI; target::Symbol=:sharpness, output::String="phi")
    (;x, p, rho0) = apt
    adaptMZI_online(x, p, rho0, Symbol(output), target)
end

function adaptMZI_online(x, p, rho0, output, target::Symbol)
    N = Int(sqrt(size(rho0,1))) - 1
    a = destroy(N+1) |> sparse
    exp_ix = [exp(1.0im*xi) for xi in x]
    phi_span = range(-pi, stop=pi, length=length(x)) |> collect

    phi = 0.0
    a_res = [Matrix{ComplexF64}(I, (N+1)^2, (N+1)^2) for i in 1:length(x)]

    xout, y = [], []

    if output == :phi
        for ei in 1:N-1
            println("The tunable phase is $phi ($ei episodes)")
            print("Please enter the experimental result: ")
            enter = readline()
            u = parse(Int64, enter)
            pyx = zeros(length(x)) |> sparse
            for xi in 1:length(x)
                a_res_tp = a_res[xi]*a_u(a, x[xi], phi, u)
                pyx[xi] = real(tr(rho0*a_res_tp'*a_res_tp))*(factorial(N-ei)/factorial(N))
                a_res[xi] = a_res_tp
            end
            phi_update = calculate_online{eval(target)}(x, p, pyx, a_res, a, rho0, N, ei, phi_span, exp_ix)
                
            append!(xout, phi)
            append!(y, u)
            phi = phi_update
        end
        println("The estimator of the unknown phase is $phi ")
        append!(xout, phi)
        savefile_online(xout, y)
    else
        println("The initial tunable phase is $phi")
        for ei in 1:N-1
            print("Please enter the experimental result: ")
            enter = readline()
            u = parse(Int64, enter)

            pyx = zeros(length(x)) |> sparse
            for xi in 1:length(x)
                a_res_tp = a_res[xi]*a_u(a, x[xi], phi, u)
                pyx[xi] = real(tr(rho0*a_res_tp'*a_res_tp))*(factorial(N-ei)/factorial(N))
                a_res[xi] = a_res_tp
            end

            phi_update = calculate_online{eval(target)}(x, p, pyx, a_res, a, rho0, N, ei, phi_span, exp_ix)
                
            println("The adjustments of the feedback phase is $(abs(phi_update-phi)) ($ei episodes)")
            append!(xout, abs(phi_update-phi))
            append!(y, u)
            phi = phi_update
        end
        savefile_online(xout, y)
    end
end

adaptMZI_online(x, p, rho0, output::String, target::String) = adaptMZI_online(x, p, rho0, Symbol(output), Symbol(target))

function calculate_online{sharpness}(x, p, pyx, a_res, a, rho0, N, ei, phi_span, exp_ix)
    
    M_res = zeros(length(phi_span))
    for mj in 1:length(phi_span)
        M1_res = trapz(x, pyx.*p)
        pyx0, pyx1 = zeros(length(x)), zeros(length(x))
        M2_res = 0.0
        for xj in 1:length(x)
            a_res0 = a_res[xj]*a_u(a, x[xj], phi_span[mj], 0)
            a_res1 = a_res[xj]*a_u(a, x[xj], phi_span[mj], 1)
            pyx0[xj] = real(tr(rho0*a_res0'*a_res0))*(factorial(N-(ei+1))/factorial(N))
            pyx1[xj] = real(tr(rho0*a_res1'*a_res1))*(factorial(N-(ei+1))/factorial(N))
            M2_res = abs(trapz(x, pyx0.*p.*exp_ix))+abs(trapz(x, pyx1.*p.*exp_ix))
        end
        M_res[mj] = M2_res/M1_res
    end
    indx_m = findmax(M_res)[2]
    phi_span[indx_m]
end

function calculate_online{MI}(x, p, pyx, a_res, a, rho0, N, ei, phi_span, exp_ix)
    
    M_res = zeros(length(phi_span))
    for mj in 1:length(phi_span)
        M1_res = trapz(x, pyx.*p)
        pyx0, pyx1 = zeros(length(x)), zeros(length(x))
        M2_res = 0.0
        for xj in 1:length(x)
            a_res0 = a_res[xj]*a_u(a, x[xj], phi_span[mj], 0)
            a_res1 = a_res[xj]*a_u(a, x[xj], phi_span[mj], 1)
            pyx0[xj] = real(tr(rho0*a_res0'*a_res0))*(factorial(N-(ei+1))/factorial(N))
            pyx1[xj] = real(tr(rho0*a_res1'*a_res1))*(factorial(N-(ei+1))/factorial(N))
            M2_res = trapz(x, pyx0.*p.*log.(2, pyx0./trapz(x, pyx0.*p)))+trapz(x, pyx1.*p.*log.(2, pyx1./trapz(x, pyx1.*p)))
        end
        M_res[mj] = M2_res/M1_res
    end
    indx_m = findmax(M_res)[2]
    phi_span[indx_m]
end

function savefile_online(xout, y)
    open("xout.csv","w") do m
        writedlm(m, xout)
    end
    open("y.csv","w") do n
        writedlm(n, y)
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
function offline(apt::Adapt_MZI, alg; target::Symbol=:sharpness, eps = GLOBAL_EPS, seed=1234)
    rng = MersenneTwister(seed)
    (;x,p,rho0) = apt
    N = Int(sqrt(size(rho0,1))) - 1
    a = destroy(N+1) |> sparse
    comb = brgd(N)|>x->[[parse(Int, s) for s in ss] for ss in x]
    if alg isa DE
        (;p_num,ini_population,c,cr,max_episode) = alg
        if ismissing(ini_population)
            ini_population = ([apt.rho0],)
        end
        DE_deltaphiOpt(x,p,rho0,comb,p_num,ini_population[1],c,cr,rng,max_episode,target,eps)
    elseif alg isa PSO
        (;p_num,ini_particle,c0,c1,c2,max_episode) = alg
        if ismissing(ini_particle)
            ini_particle = ([apt.rho0],)
        end
        PSO_deltaphiOpt(x,p,rho0,comb,p_num,ini_particle[1],c0,c1,c2,rng,max_episode,target,eps)
    end
end

function DE_deltaphiOpt(x, p, rho0, comb, p_num, ini_population, c, cr, rng::AbstractRNG, max_episode, target::Symbol, eps)
    N = Int(sqrt(size(rho0,1))) - 1
    a = destroy(N+1) |> sparse
    deltaphi = [zeros(N) for i in 1:p_num]
    # initialize
    res = logarithmic(2.0*pi, N)
    if length(ini_population) > p_num
        ini_population = [ini_population[i] for i in 1:p_num]
    end
    for pj in 1:length(ini_population)
        deltaphi[pj] = [ini_population[pj][i] for i in 1:N]
    end
    for pk in (length(ini_population)+1):p_num
        deltaphi[pk] = [res[i]+rand(rng) for i in 1:N]
    end

    p_fit = [0.0 for i in 1:p_num]
    for pl in 1:N
        p_fit[pl] = calculate_offline{eval(target)}(deltaphi[pl], x, p, rho0, a, comb, eps)
    end
    
    f_ini = maximum(p_fit)
    f_list = [f_ini]
    for ei in 1:(max_episode-1)
        for pm in 1:p_num
            #mutations
            mut_num = sample(rng, 1:p_num, 3, replace=false)
            deltaphi_mut = [0.0 for i in 1:N]
            for ci in 1:N
                deltaphi_mut[ci] = deltaphi[mut_num[1]][ci]+c*(deltaphi[mut_num[2]][ci]-deltaphi[mut_num[3]][ci])
            end
            #crossover
            deltaphi_cross = [0.0 for i in 1:N]
            cross_int = sample(rng, 1:N, 1, replace=false)[1]
            for cj in 1:N
                rand_num = rand(rng)
                if rand_num <= cr
                    deltaphi_cross[cj] = deltaphi_mut[cj]
                else
                    deltaphi_cross[cj] = deltaphi[pm][cj]
                end
                deltaphi_cross[cross_int] = deltaphi_mut[cross_int]
            end
            #selection
            for cm in 1:N
                deltaphi_cross[cm] = (x-> x < 0.0 ? 0.0 : x > pi ? pi : x)(deltaphi_cross[cm])
            end
            f_cross = calculate_offline{eval(target)}(deltaphi_cross, x, p, rho0, a, comb, eps)
            if f_cross > p_fit[pm]
                p_fit[pm] = f_cross
                for ck in 1:N
                    deltaphi[pm][ck] = deltaphi_cross[ck]
                end
            end
        end
        append!(f_list, maximum(p_fit))
    end
    savefile_offline(deltaphi[findmax(p_fit)[2]], f_list)
    return deltaphi[findmax(p_fit)[2]]
end

DE_deltaphiOpt(x, p, rho0, comb, p_num, ini_population, c, cr, seed::Number, max_episode, target::String, eps) = 
DE_deltaphiOpt(x, p, rho0, comb, p_num, ini_population, c, cr, MersenneTwister(seed), max_episode, Symbol(target), eps)

function PSO_deltaphiOpt(x, p, rho0, comb, p_num, ini_particle, c0, c1, c2, rng::AbstractRNG, max_episode, target::Symbol, eps)   
    N = Int(sqrt(size(rho0,1))) - 1
    a = destroy(N+1) |> sparse
    n = size(a)[1]

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    deltaphi = [zeros(N) for i in 1:p_num]
    velocity = [zeros(N) for i in 1:p_num]
    # initialize
    res = logarithmic(2.0*pi, N)
    if length(ini_particle) > p_num
        ini_particle = [ini_particle[i] for i in 1:p_num]
    end
    for pj in 1:length(ini_particle)
        deltaphi[pj] = [ini_particle[pj][i] for i in 1:N]
    end
    for pk in (length(ini_particle)+1):p_num
        deltaphi[pk] = [res[i]+rand(rng) for i in 1:N]
    end
    for pl in 1:p_num
        velocity[pl] = [0.1*rand(rng) for i in 1:N]
    end
    
    pbest = [zeros(N) for i in 1:p_num]
    gbest = zeros(N)
    fit = 0.0
    p_fit = [0.0 for i in 1:p_num]
    f_list = []
    for ei in 1:(max_episode[1]-1)
        for pm in 1:p_num
            f_now = calculate_offline{eval(target)}(deltaphi[pm], x, p, rho0, a, comb, eps)
            if f_now > p_fit[pm]
                p_fit[pm] = f_now
                for ci in 1:N
                    pbest[pm][ci] = deltaphi[pm][ci]
                end
            end
        end

        for pn in 1:p_num
            if p_fit[pn] > fit
                fit = p_fit[pn]
                for cj in 1:N
                    gbest[cj] = pbest[pn][cj]
                end
            end 
        end

        for pa in 1:p_num
            deltaphi_pre = [0.0 for i in 1:N]
            for ck in 1:N
                deltaphi_pre[ck] = deltaphi[pa][ck]
                velocity[pa][ck] = c0*velocity[pa][ck] + c1*rand(rng)*(pbest[pa][ck] - deltaphi[pa][ck]) 
                                    + c2*rand(rng)*(gbest[ck] - deltaphi[pa][ck])
                deltaphi[pa][ck] += velocity[pa][ck]
            end

            for cn in 1:N
                deltaphi[pa][cn] = (x-> x < 0.0 ? 0.0 : x > pi ? pi : x)(deltaphi[pa][cn])
                velocity[pa][cn] = deltaphi[pa][cn] - deltaphi_pre[cn]
            end
        end
        append!(f_list, fit)
        if ei%max_episode[2] == 0
            for pb in 1:p_num
                deltaphi[pb] = [gbest[i] for i in 1:N]
            end
        end
    end
    savefile_offline(gbest, f_list)
    return gbest
end

PSO_deltaphiOpt(x, p, rho0, comb, p_num, ini_particle, c0, c1, c2, seed::Number, max_episode, target::String, eps) = 
PSO_deltaphiOpt(x, p, rho0, comb, p_num, ini_particle, c0, c1, c2, MersenneTwister(seed), max_episode, Symbol(target), eps)   

function calculate_offline{sharpness}(delta_phi, x, p, rho0, a, comb, eps)
    N = size(a)[1] - 1
    exp_ix = [exp(1.0im*xi) for xi in x]
    
    M_res = zeros(length(comb))
    for ui in 1:length(comb)
        u = comb[ui]
        phi = 0.0

        a_res = [Matrix{ComplexF64}(I, (N+1)^2, (N+1)^2) for i in 1:length(x)]
        for ei in 1:N-1
            phi = phi - (-1)^u[ei]*delta_phi[ei]
            for xi in 1:length(x)
                a_res[xi] = a_res[xi]*a_u(a, x[xi], phi, u[ei])
            end
        end

        pyx = zeros(length(x))
        for xj in 1:length(x)
            pyx[xj] = real(tr(rho0*a_res[xj]'*a_res[xj]))*(1/factorial(N))
        end
        M_res[ui] = abs(trapz(x, pyx.*p.*exp_ix))
    end
    return sum(M_res)
end

function calculate_offline{MI}(delta_phi, x, p, rho0, a, comb, eps)
    N = size(a)[1] - 1
    exp_ix = [exp(1.0im*xi) for xi in x]
    
    M_res = zeros(length(comb))
    for ui in 1:length(comb)
        u = comb[ui]
        phi = 0.0

        a_res = [Matrix{ComplexF64}(I, (N+1)^2, (N+1)^2) for i in 1:length(x)]
        for ei in 1:N-1
            phi = phi - (-1)^u[ei]*delta_phi[ei]
            for xi in 1:length(x)
                a_res[xi] = a_res[xi]*a_u(a, x[xi], phi, u[ei])
            end
        end

        pyx = zeros(length(x))
        for xj in 1:length(x)
            pyx[xj] = real(tr(rho0*a_res[xj]'*a_res[xj]))*(1/factorial(N))
        end
        M_res[ui] = trapz(x, pyx.*p.*log.(2, pyx./trapz(x, pyx.*p)))
    end
    return sum(M_res)
end

function savefile_offline(deltaphi, flist)
    open("deltaphi.csv","w") do m
        writedlm(m, deltaphi)
    end
    open("f.csv","w") do n
        writedlm(n, flist)
    end
end

function a_u(a, x, phi, u)
    N = size(a)[1] - 1
    a_in = kron(a, Matrix(I, N+1, N+1))
    b_in = kron(Matrix(I, N+1, N+1), a)

    value = 0.5*(x-phi)+0.5*pi*u
    return a_in*sin(value) + b_in*cos(value)
end

function logarithmic(number, N)
    res = zeros(N)
    res_tp = number
    for i in 1:N
        res_tp = res_tp/2
        res[i] = res_tp
    end
    return res
end

function brgd(n)
    if n == 1
        return ["0", "1"]
    end
    L0 = brgd(n-1)
    L1 = deepcopy(L0)
    reverse!(L1)
    L0 = ["0"*l for l in L0]
    L1 = ["1"*l for l in L1]
    return deepcopy(vcat(L0,L1))
end
