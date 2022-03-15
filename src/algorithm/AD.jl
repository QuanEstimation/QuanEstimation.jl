#### AD ####
function update!(opt::ControlOpt, alg::AD, obj, dynamics, output)
    (; max_episode, ϵ, beta1, beta2) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)

    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj)

    output.f_list = [f_ini]
    for ei = 1:(max_episode-1)
        # δ = grad(obj, dynamics)
        δ = gradient(() -> objective(obj, dynamics)[2], Flux.Params([dynamics.data.ctrl]))
        update_ctrl!(alg, obj, dynamics, δ[dynamics.data.ctrl])
        bound!(dynamics.data.ctrl, opt.ctrl_bound)
        f_out, f_now = objective(obj, dynamics)

        set_f!(output, f_out)
        set_buffer!(output, [dynamics.data.ctrl])
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function update_ctrl!(alg::AD{Adam}, obj, dynamics, δ)
    (; ϵ, beta1, beta2) = alg
    for ci = 1:length(δ)
        mt, vt = 0.0, 0.0
        for ti = 1:length(δ[1])
            dynamics.data.ctrl[ci][ti], mt, vt = Adam(
                δ[ci][ti],
                ti,
                dynamics.data.ctrl[ci][ti],
                mt,
                vt,
                ϵ,
                beta1,
                beta2,
                obj.eps,
            )
        end
    end
end

function update_ctrl!(alg::AD{GradDescent}, obj, dynamics, δ)
    (; ϵ) = alg
    dynamics.data.ctrl += ϵ * δ
end

#### state optimization ####
function update!(opt::StateOpt, alg::AD, obj, dynamics, output)
    (; max_episode) = alg
    f_ini, f_comp = objective(obj, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ψ0)
    set_io!(output, f_ini)
    show(opt, output, obj)
    for ei = 1:(max_episode-1)
        δ = grad(opt, obj, dynamics)
        update_state!(alg, obj, dynamics, δ)
        dynamics.data.ψ0 = dynamics.data.ψ0 / norm(dynamics.data.ψ0)
        f_out, f_now = objective(obj, dynamics)
        set_output!(output, f_out)
        set_buffer!(output, dynamics.data.ψ0)
        show(output, obj)
    end
    show(output, output)
end

function update_state!(alg::AD{Adam}, obj, dynamics, δ)
    (; ϵ, beta1, beta2) = alg
    mt, vt = 0.0, 0.0
    for ti = 1:length(δ)
        dynamics.data.ψ0[ti], mt, vt =
            Adam(δ[ti], ti, dynamics.data.ψ0[ti], mt, vt, ϵ, beta1, beta2, obj.eps)
    end
end

function update_state!(alg::AD{GradDescent}, obj, dynamics, δ)
    (; ϵ) = alg
    dynamics.data.ψ0 += ϵ * δ
end

function grad(opt, obj, dynamics::Lindblad{noiseless,free,ket})
    (; H0, dH, ψ0, tspan) = dynamics.data
    δ = gradient(x -> objective(obj, H0, dH, x, tspan), ψ0) .|> real |> sum
end

function grad(opt, obj, dynamics::Lindblad{noisy,free,ket})
    (; H0, dH, ψ0, tspan, decay_opt, γ) = dynamics.data
    δ = gradient(x -> objective(obj, H0, dH, x, tspan, decay_opt, γ), ψ0) .|> real |> sum
end

function grad(opt, obj, dynamics::Lindblad{noiseless,timedepend,ket})
    (; H0, dH, ψ0, tspan) = dynamics.data
    δ = gradient(x -> objective(obj, H0, dH, x, tspan), ψ0) .|> real |> sum
end

function grad(opt, obj, dynamics::Lindblad{noisy,timedepend,ket})
    (; H0, dH, ψ0, tspan, decay_opt, γ) = dynamics.data
    δ = gradient(x -> objective(obj, H0, dH, x, tspan, decay_opt, γ), ψ0) .|> real |> sum
end

function grad(opt, obj, dynamics::Kraus{noiseless,free,ket})
    (; K, dK, ψ0) = dynamics.data
    δ = gradient(x -> objective(obj, K, dK, x), ψ0) .|> real |> sum
end

#### measurement optimization (linear conbination) ####
function update!(opt::Mopt_LinearComb, alg::AD, obj, dynamics, output, POVM_basis, M_num)
    (; max_episode, rng) = alg
    basis_num = length(POVM_basis)
    # initialize 
    B = [rand(rng, basis_num) for i = 1:M_num]
    B = bound_LC_coeff!(B)

    M = [sum([B[i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num]
    f_opt, f_comp = objective(obj::QFIM{SLD}, dynamics)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, dynamics)
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_opt, f_povm, f_ini)
    show(opt, output, obj)
    for ei = 1:(max_episode-1)
        δ = grad(opt, obj, dynamics, B, POVM_basis, M_num)
        update_M!(alg, obj, dynamics, δ, B)
        B = bound_LC_coeff!(B)
        M = [sum([B[i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num]
        obj_copy = set_M(obj, M)
        f_out, f_now = objective(obj_copy, dynamics)
        set_output!(output, f_out)
        set_buffer!(output, M)
        show(output, obj)
    end
    show(output, output)
end

function update_M!(alg::AD{Adam}, obj, dynamics, δ, B::Vector{Vector{Float64}})
    (; ϵ, beta1, beta2) = alg
    for ci = 1:length(δ)
        mt, vt = 0.0, 0.0
        for ti = 1:length(δ[1])
            B[ci][ti], mt, vt =
                Adam(δ[ci][ti], ti, B[ci][ti], mt, vt, ϵ, beta1, beta2, obj.eps)
        end
    end
end

function update_M!(alg::AD{GradDescent}, obj, dynamics, δ, B::Vector{Vector{Float64}})
    (; ϵ) = alg
    B += ϵ * δ
end

function grad(opt, obj, dynamics, B, POVM_basis, M_num::Number)
    (; H0, dH, ρ0, tspan, decay_opt, γ) = dynamics.data
    basis_num = length(POVM_basis)
    δ =
        gradient(
            x -> objective(
                obj,
                [sum([x[i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num],
                H0,
                dH,
                ρ0,
                tspan,
                decay_opt,
                γ,
            ),
            B,
        ) .|>
        real |>
        sum
end

function grad(opt, obj, dynamics, B, POVM_basis, M_num::Number)
    (; K, dK, ρ0) = dynamics.data
    basis_num = length(POVM_basis)
    δ =
        gradient(
            x -> objective(
                obj,
                [sum([x[i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num],
                K,
                dK,
                ρ0,
            ),
            B,
        ) .|>
        real |>
        sum
end

#### measurement optimization (rotation) ####
function update!(opt::Mopt_Rotation, alg::AD, obj, dynamics, output, POVM_basis)
    (; max_episode, rng) = alg
    dim = size(dynamics.data.ρ0)[1]
    suN = suN_generator(dim)
    Lambda = [Matrix{ComplexF64}(I, dim, dim)]
    append!(Lambda, [suN[i] for i = 1:length(suN)])

    M_num = length(POVM_basis)

    s = rand(rng, dim * dim)
    U = rotation_matrix(s, Lambda)
    M = [U * POVM_basis[i] * U' for i = 1:M_num]
    f_opt, f_comp = objective(obj::QFIM{SLD}, dynamics)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, dynamics)
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_opt, f_povm, f_ini)
    show(opt, output, obj)
    for ei = 1:(max_episode-1)
        δ = grad(opt, obj, dynamics, s, POVM_basis, Lambda)
        update_M!(alg, obj, dynamics, δ, s)
        s = bound_rot_coeff!(s)
        U = rotation_matrix(s, Lambda)
        M = [U * POVM_basis[i] * U' for i = 1:M_num]
        obj_copy = set_M(obj, M)
        f_out, f_now = objective(obj_copy, dynamics)

        set_output!(output, f_out)
        set_buffer!(output, M)
        show(output, obj)
    end
    show(output, output)
end

function update_M!(alg::AD{Adam}, obj, dynamics, δ, s::Vector{Float64})
    (; ϵ, beta1, beta2) = alg
    mt, vt = 0.0, 0.0
    for ti = 1:length(δ)
        s[ti], mt, vt = Adam(δ[ti], ti, s[ti], mt, vt, ϵ, beta1, beta2, obj.eps)
    end
end

function update_M!(alg::AD{GradDescent}, obj, dynamics, δ, s::Vector{Float64})
    (; ϵ) = alg
    s += ϵ * δ
end

function grad(opt, obj, dynamics, s, POVM_basis, Lambda::AbstractArray)
    (; H0, dH, ρ0, tspan, decay_opt, γ) = dynamics.data
    δ =
        gradient(
            x -> objective(obj, x, POVM_basis, Lambda, H0, dH, ρ0, tspan, decay_opt, γ),
            s,
        ) .|>
        real |>
        sum
end

function grad(opt, obj, dynamics, s, POVM_basis, Lambda::AbstractArray)
    (; K, dK, ρ0) = dynamics.data
    δ = gradient(x -> objective(obj, x, POVM_basis, Lambda, K, dK, ρ0), s) .|> real |> sum
end
