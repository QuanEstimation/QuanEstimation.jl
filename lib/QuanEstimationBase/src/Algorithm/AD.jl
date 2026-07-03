#### control optimization ####
"""
    optimize!(opt::ControlOpt, alg::AbstractautoGRAPE, obj, scheme, output)

Optimize control amplitudes using auto-GRAPE with automatic differentiation.
"""
function optimize!(opt::ControlOpt, alg::AbstractautoGRAPE, obj, scheme, output)
    (; max_episode) = alg
    ctrl_length = length(param_data(scheme).ctrl[1])
    ctrl_num = length(param_data(scheme).Hc)

    scheme_copy = set_ctrl(scheme, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, scheme_copy)
    f_ini, f_comp = objective(obj, scheme)

    set_f!(output, f_ini)
    set_buffer!(output, param_data(scheme).ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj, alg)

    for ei = 1:(max_episode-1)
        δ = Zygote.gradient(
            () -> objective(obj, scheme)[2],
            Zygote.Params([param_data(scheme).ctrl]),
        )
        update_ctrl!(alg, obj, scheme, δ[param_data(scheme).ctrl])
        bound!(param_data(scheme).ctrl, opt.ctrl_bound)
        f_out, f_now = objective(obj, scheme)

        set_f!(output, f_out)
        set_buffer!(output, param_data(scheme).ctrl)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

"""
    update_ctrl!(alg::autoGRAPE_Adam, obj, scheme, δ)

Update control coefficients using the Adam optimizer (auto-GRAPE variant).
"""
function update_ctrl!(alg::autoGRAPE_Adam, obj, scheme, δ)
    (; epsilon, beta1, beta2) = alg
    for ci in eachindex(δ)
        mt, vt = 0.0, 0.0
        for ti in eachindex(δ[1])
            param_data(scheme).ctrl[ci][ti], mt, vt = Adam(
                δ[ci][ti],
                ti,
                param_data(scheme).ctrl[ci][ti],
                mt,
                vt,
                epsilon,
                beta1,
                beta2,
                obj.eps,
            )
        end
    end
end

"""
    update_ctrl!(alg::autoGRAPE, obj, scheme, δ)

Update control coefficients using a fixed learning rate (auto-GRAPE variant).
"""
function update_ctrl!(alg::autoGRAPE, obj, scheme, δ)
    param_data(scheme).ctrl += alg.epsilon * δ
end

#### state optimization ####
"""
    optimize!(opt::StateOpt, alg::AbstractAD, obj, scheme, output)

Optimize the initial probe state using automatic differentiation.
"""
function optimize!(opt::StateOpt, alg::AbstractAD, obj, scheme, output)
    (; max_episode) = alg
    f_ini, _ = objective(obj, scheme)
    set_f!(output, f_ini)
    set_buffer!(output, state_data(scheme))
    set_io!(output, f_ini)
    show(opt, output, obj, alg)
    for ei = 1:(max_episode-1)
        δ = Zygote.gradient(
            () -> objective(obj, scheme)[2],
            Zygote.Params([state_data(scheme)]),
        )
        update_state!(alg, obj, scheme, δ[state_data(scheme)])
        scheme.StatePreparation.data = state_data(scheme) / norm(state_data(scheme))
        f_out, _ = objective(obj, scheme)
        set_f!(output, f_out)
        set_buffer!(output, state_data(scheme))
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

"""
    update_state!(alg::AD_Adam, obj, scheme, δ)

Update the probe state using the Adam optimizer.
"""
function update_state!(alg::AD_Adam, obj, scheme, δ)
    (; epsilon, beta1, beta2) = alg
    mt, vt = 0.0, 0.0
    for ti in eachindex(δ)
        scheme.StatePreparation.data[ti], mt, vt =
            Adam(δ[ti], ti, state_data(scheme)[ti], mt, vt, epsilon, beta1, beta2, obj.eps)
    end
end

"""
    update_state!(alg::AD, obj, scheme, δ)

Update the probe state using a fixed learning rate.
"""
function update_state!(alg::AD, obj, scheme, δ)
    scheme.StatePreparation.data += alg.epsilon * δ
end

#### find the optimal linear combination of a given set of POVM ####
"""
    optimize!(opt::Mopt_LinearComb, alg::AbstractAD, obj, scheme, output)

Optimize measurement by finding the best linear combination of a POVM basis.
"""
function optimize!(opt::Mopt_LinearComb, alg::AbstractAD, obj, scheme, output)
    (; max_episode) = alg
    (; POVM_basis, M_num) = opt
    rng = MersenneTwister(1234)
    basis_num = length(POVM_basis)

    bound_LC_coeff!(opt.B, rng)
    M = [sum([opt.B[i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num]
    obj_QFIM = QFIM_obj(obj)
    f_opt, f_comp = objective(obj_QFIM, scheme)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, scheme)
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, scheme)
    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_ini, f_povm, f_opt)
    show(opt, output, obj, alg)
    for ei = 1:(max_episode-1)
        δ = Zygote.gradient(() -> objective(opt, obj, scheme)[2], Zygote.Params([opt.B]))
        update_M!(opt, alg, obj, δ[opt.B])
        bound_LC_coeff!(opt.B, rng)
        M = [sum([opt.B[i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num]
        obj_copy = set_M(obj, M)
        f_out, f_now = objective(obj_copy, scheme)
        set_f!(output, f_out)
        set_buffer!(output, M)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

"""
    update_M!(opt::Mopt_LinearComb, alg::AD_Adam, obj, δ)

Update POVM linear-combination coefficients using Adam.
"""
function update_M!(opt::Mopt_LinearComb, alg::AD_Adam, obj, δ)
    (; epsilon, beta1, beta2) = alg
    for ci in eachindex(δ)
        mt, vt = 0.0, 0.0
        for ti in eachindex(δ[1])
            opt.B[ci][ti], mt, vt =
                Adam(δ[ci][ti], ti, opt.B[ci][ti], mt, vt, epsilon, beta1, beta2, obj.eps)
        end
    end
end

"""
    update_M!(opt::Mopt_LinearComb, alg::AD, obj, δ)

Update POVM linear-combination coefficients using a fixed learning rate.
"""
function update_M!(opt::Mopt_LinearComb, alg::AD, obj, δ)
    opt.B += alg.epsilon * δ
end

#### find the optimal rotated measurement of a given set of POVM ####
"""
    optimize!(opt::Mopt_Rotation, alg::AbstractAD, obj, scheme, output)

Optimize measurement by finding the best unitary rotation of a POVM basis.
"""
function optimize!(opt::Mopt_Rotation, alg::AbstractAD, obj, scheme, output)
    (; max_episode) = alg
    (; POVM_basis) = opt
    dim = get_dim(scheme)
    M_num = length(POVM_basis)
    suN = suN_generator(dim)
    opt.Lambda = Matrix{ComplexF64}[]
    append!(opt.Lambda, [Matrix{ComplexF64}(I, dim, dim)])
    append!(opt.Lambda, [suN[i] for i in eachindex(suN)])


    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U * POVM_basis[i] * U' for i = 1:M_num]
    obj_QFIM = QFIM_obj(obj)
    f_opt, f_comp = objective(obj_QFIM, scheme)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, scheme)
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, scheme)
    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_ini, f_povm, f_opt)
    show(opt, output, obj, alg)
    for ei = 1:(max_episode-1)
        δ = Zygote.gradient(() -> objective(opt, obj, scheme)[2], Zygote.Params([opt.s]))
        update_M!(opt, alg, obj, δ[opt.s])
        bound_rot_coeff!(opt.s)
        U = rotation_matrix(opt.s, opt.Lambda)
        M = [U * POVM_basis[i] * U' for i = 1:M_num]
        obj_copy = set_M(obj, M)
        f_out, f_now = objective(obj_copy, scheme)
        set_f!(output, f_out)
        set_buffer!(output, M)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

"""
    update_M!(opt::Mopt_Rotation, alg::AD_Adam, obj, δ)

Update POVM rotation parameters using Adam.
"""
function update_M!(opt::Mopt_Rotation, alg::AD_Adam, obj, δ)
    (; epsilon, beta1, beta2) = alg
    mt, vt = 0.0, 0.0
    for ti in eachindex(δ)
        opt.s[ti], mt, vt =
            Adam(δ[ti], ti, opt.s[ti], mt, vt, epsilon, beta1, beta2, obj.eps)
    end
end

"""
    update_M!(opt::Mopt_Rotation, alg::AD, obj, δ)

Update POVM rotation parameters using a fixed learning rate.
"""
function update_M!(opt::Mopt_Rotation, alg::AD, obj, δ)
    opt.s += alg.epsilon * δ
end

#### state abd control optimization ####
"""
    optimize!(opt::StateControlOpt, alg::AbstractAD, obj, scheme, output)

Jointly optimize the initial probe state and control amplitudes.
"""
function optimize!(opt::StateControlOpt, alg::AbstractAD, obj, scheme, output)
    (; max_episode) = alg
    ctrl_length = length(param_data(scheme).ctrl[1])
    ctrl_num = length(param_data(scheme).Hc)

    scheme_copy = set_ctrl(scheme, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, scheme_copy)
    f_ini, f_comp = objective(obj, scheme)

    set_f!(output, f_ini)
    set_buffer!(output, state_data(scheme), param_data(scheme).ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj, alg)

    for ei = 1:(max_episode-1)
        δ = Zygote.gradient(
            () -> objective(obj, scheme)[2],
            Zygote.Params([state_data(scheme), param_data(scheme).ctrl]),
        )
        update_state!(alg, obj, scheme, δ[state_data(scheme)])
        update_ctrl!(alg, obj, scheme, δ[param_data(scheme).ctrl])
        bound!(param_data(scheme).ctrl, opt.ctrl_bound)
        scheme.StatePreparation.data = state_data(scheme) / norm(state_data(scheme))
        f_out, f_now = objective(obj, scheme)

        set_f!(output, f_out)
        set_buffer!(output, state_data(scheme), param_data(scheme).ctrl)
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

"""
    update_ctrl!(alg::AD_Adam, obj, scheme, δ)

Update control coefficients using the Adam optimizer.
"""
function update_ctrl!(alg::AD_Adam, obj, scheme, δ)
    (; epsilon, beta1, beta2) = alg
    for ci in eachindex(δ)
        mt, vt = 0.0, 0.0
        for ti in eachindex(δ[1])
            param_data(scheme).ctrl[ci][ti], mt, vt = Adam(
                δ[ci][ti],
                ti,
                param_data(scheme).ctrl[ci][ti],
                mt,
                vt,
                epsilon,
                beta1,
                beta2,
                obj.eps,
            )
        end
    end
end

"""
    update_ctrl!(alg::AD, obj, scheme, δ)

Update control coefficients using a fixed learning rate.
"""
function update_ctrl!(alg::AD, obj, scheme, δ)
    param_data(scheme).ctrl += alg.epsilon * δ
end
