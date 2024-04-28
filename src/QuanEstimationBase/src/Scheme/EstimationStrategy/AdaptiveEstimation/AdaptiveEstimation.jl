function adapt!(scheme; method = "FOP", savefile = false, max_episode::Int = 1000)
	init!(scheme)
	@unpack x, p, dp = scheme.Parameterization.data
	p_num = length(p)

	rho_all = adptive_evolve(scheme)
	F = CFIM(scheme)

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
				append!(y, Int(res_exp - 1))
			end
			savefile_false(p, xout, y)
		else
			for ei in 1:max_episode
				p, x_out, res_exp, u = iter_FOP_singlepara(p, p_num, x, u, rho_all, M, dim, x_opt, ei)
				savefile_true(p, x_out, Int(res_exp - 1))
			end
		end
	elseif method == "MI"
		if savefile == false
			y, xout = [], []
			for ei in 1:max_episode
				p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
				append!(xout, x_out)
				append!(y, Int(res_exp - 1))
			end
			savefile_false(p, xout, y)
		else
			for ei in 1:max_episode
				p, x_out, res_exp, u = iter_MI_singlepara(p, p_num, x, u, rho_all, M, dim, ei)
				savefile_true(p, x_out, Int(res_exp - 1))
			end
		end
	end

end

function adapt(scheme) end


function iter_FOP(p, p_num, para_num, x, x_list, u, rho_all, M, x_opt, ei)
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