function optimize!(opt::StateOpt, alg::NM, obj, scheme, output)
    (; max_episode, p_num, ini_state, ar, ae, ac, as0) = alg
    if isnothing(ini_state)
        ini_state = [opt.psi]
    end
    dim = get_dim(scheme)
    nelder_mead = repeat_copy(scheme, p_num)

    # initialize 
    if length(ini_state) > p_num
        ini_state = [ini_state[i] for i = 1:p_num]
    end
    for pj in eachindex(ini_state)
        nelder_mead[pj].StatePreparation.data = [ini_state[pj][i] for i = 1:dim]
    end
    for pj = (length(ini_state)+1):p_num
        r_ini = 2 * rand(opt.rng, dim) - ones(dim)
        r = r_ini / norm(r_ini)
        phi = 2 * pi * rand(opt.rng, dim)
        nelder_mead[pj].StatePreparation.data = [r[i] * exp(1.0im * phi[i]) for i = 1:dim]
    end

    p_fit, p_out = zeros(p_num), zeros(p_num)
    for pj = 1:p_num
        p_out[pj], p_fit[pj] = objective(obj, nelder_mead[pj])
    end
    sort_ind = sortperm(p_fit, rev = true)

    set_f!(output, p_out[1])
    set_buffer!(output, scheme.StatePreparation.data)
    set_io!(output, p_out[1])
    show(opt, output, obj, alg)

    f_list = [p_out[1]]
    idx = 0
    for ei = 1:(max_episode-1)
        # calculate the average vector
        vec_ave = zeros(ComplexF64, dim)
        for ni = 1:dim
            vec_ave[ni] =
                [nelder_mead[pk].StatePreparation.data[ni] for pk = 1:(p_num-1)] |> sum
            vec_ave[ni] = vec_ave[ni] / (p_num - 1)
        end
        vec_ave = vec_ave / norm(vec_ave)

        # reflection
        vec_ref = zeros(ComplexF64, dim)
        for nj = 1:dim
            vec_ref[nj] =
                vec_ave[nj] +
                ar * (vec_ave[nj] - nelder_mead[sort_ind[end]].StatePreparation.data[nj])
        end
        vec_ref = vec_ref / norm(vec_ref)
        scheme_copy = set_state!(scheme, vec_ref)
        fr_out, fr = objective(obj, scheme_copy)

        if fr > p_fit[sort_ind[1]]
            # expansion
            vec_exp = zeros(ComplexF64, dim)
            for nk = 1:dim
                vec_exp[nk] = vec_ave[nk] + ae * (vec_ref[nk] - vec_ave[nk])
            end
            vec_exp = vec_exp / norm(vec_exp)
            scheme_copy = set_state!(scheme, vec_exp)
            fe_out, fe = objective(obj, scheme_copy)
            if fe <= fr
                for np = 1:dim
                    nelder_mead[sort_ind[end]].StatePreparation.data[np] = vec_ref[np]
                end
                p_fit[sort_ind[end]] = fr
                p_out[sort_ind[end]] = fr_out
                sort_ind = sortperm(p_fit, rev = true)
            else
                for np = 1:dim
                    nelder_mead[sort_ind[end]].StatePreparation.data[np] = vec_exp[np]
                end
                p_fit[sort_ind[end]] = fe
                p_out[sort_ind[end]] = fe_out
                sort_ind = sortperm(p_fit, rev = true)
            end
        elseif fr < p_fit[sort_ind[end-1]]
            # constraction
            if fr <= p_fit[sort_ind[end]]
                # inside constraction
                vec_ic = zeros(ComplexF64, dim)
                for nl = 1:dim
                    vec_ic[nl] =
                        vec_ave[nl] -
                        ac *
                        (vec_ave[nl] - nelder_mead[sort_ind[end]].StatePreparation.data[nl])
                end
                vec_ic = vec_ic / norm(vec_ic)
                scheme_copy = set_state!(scheme, vec_ic)
                fic_out, fic = objective(obj, scheme_copy)
                if fic > p_fit[sort_ind[end]]
                    for np = 1:dim
                        nelder_mead[sort_ind[end]].StatePreparation.data[np] = vec_ic[np]
                    end
                    p_fit[sort_ind[end]] = fic
                    p_out[sort_ind[end]] = fic_out
                    sort_ind = sortperm(p_fit, rev = true)
                else
                    # shrink
                    vec_first =
                        [nelder_mead[sort_ind[1]].StatePreparation.data[i] for i = 1:dim]
                    for pk = 1:p_num
                        for nq = 1:dim
                            nelder_mead[pk].StatePreparation.data[nq] =
                                vec_first[nq] +
                                as0 *
                                (nelder_mead[pk].StatePreparation.data[nq] - vec_first[nq])
                        end
                        nelder_mead[pk].StatePreparation.data =
                            nelder_mead[pk].StatePreparation.data /
                            norm(nelder_mead[pk].StatePreparation.data)
                        p_out[pk], p_fit[pk] = objective(obj, nelder_mead[pk])
                    end
                    sort_ind = sortperm(p_fit, rev = true)
                end
            else
                # outside constraction
                vec_oc = zeros(ComplexF64, dim)
                for nn = 1:dim
                    vec_oc[nn] = vec_ave[nn] + ac * (vec_ref[nn] - vec_ave[nn])
                end
                vec_oc = vec_oc / norm(vec_oc)
                scheme_copy = set_state!(scheme, vec_oc)
                foc_out, foc = objective(obj, scheme_copy)
                if foc >= fr
                    for np = 1:dim
                        nelder_mead[sort_ind[end]].StatePreparation.data[np] = vec_oc[np]
                    end
                    p_fit[sort_ind[end]] = foc
                    p_out[sort_ind[end]] = foc_out
                    sort_ind = sortperm(p_fit, rev = true)
                else
                    # shrink
                    vec_first =
                        [nelder_mead[sort_ind[1]].StatePreparation.data[i] for i = 1:dim]
                    for pk = 1:p_num
                        for nq = 1:dim
                            nelder_mead[pk].StatePreparation.data[nq] =
                                vec_first[nq] +
                                as0 *
                                (nelder_mead[pk].StatePreparation.data[nq] - vec_first[nq])
                        end
                        nelder_mead[pk].StatePreparation.data =
                            nelder_mead[pk].StatePreparation.data /
                            norm(nelder_mead[pk].StatePreparation.data)
                        p_out[pk], p_fit[pk] = objective(obj, nelder_mead[pk])
                    end
                    sort_ind = sortperm(p_fit, rev = true)
                end
            end
        else
            for np = 1:dim
                nelder_mead[sort_ind[end]].StatePreparation.data[np] = vec_ref[np]
            end
            p_fit[sort_ind[end]] = fr
            p_out[sort_ind[end]] = fr_out
            sort_ind = sortperm(p_fit, rev = true)
        end
        idx = findmax(p_fit)[2]
        set_f!(output, p_out[idx])
        set_buffer!(output, nelder_mead[sort_ind[1]].StatePreparation.data)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end
