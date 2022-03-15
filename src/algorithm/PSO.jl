#### ControlOpt ####
function update!(opt::ControlOpt, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    particles = repeat(dynamics, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num)
    pbest = zeros(ctrl_num, ctrl_length, p_num)
    gbest = zeros(ctrl_num, ctrl_length)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    # initialize 
    initial_ctrl!(opt, ini_particle, particles, p_num)

    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj)

    output.f_list = [f_ini]
    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            f_out[pj], f_now = objective(obj, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for di = 1:ctrl_num
                    for ni = 1:ctrl_length
                        pbest[di, ni, pj] = particles[pj].data.ctrl[di][ni]
                    end
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for dj = 1:ctrl_num
                    for nj = 1:ctrl_length
                        gbest[dj, nj] = pbest[dj, nj, pj]
                    end
                end
            end
        end

        for pk = 1:p_num
            control_coeff_pre = [zeros(ctrl_length) for i = 1:ctrl_num]
            for dk = 1:ctrl_num
                for ck = 1:ctrl_length
                    control_coeff_pre[dk][ck] = particles[pk].data.ctrl[dk][ck]
                    velocity[dk, ck, pk] =
                        c0 * velocity[dk, ck, pk] +
                        c1 *
                        rand(rng) *
                        (pbest[dk, ck, pk] - particles[pk].data.ctrl[dk][ck])
                    +c2 * rand(rng) * (gbest[dk, ck] - particles[pk].data.ctrl[dk][ck])
                    particles[pk].data.ctrl[dk][ck] += velocity[dk, ck, pk]
                end
            end

            for dm = 1:ctrl_num
                for cm = 1:ctrl_length
                    particles[pk].data.ctrl[dm][cm] = (
                        x ->
                            x < opt.ctrl_bound[1] ? opt.ctrl_bound[1] :
                            x > opt.ctrl_bound[2] ? opt.ctrl_bound[2] : x
                    )(
                        particles[pk].data.ctrl[dm][cm],
                    )
                    velocity[dm, cm, pk] =
                        particles[pk].data.ctrl[dm][cm] - control_coeff_pre[dm][cm]
                end
            end
        end
        if ei % max_episode[2] == 0
            dynamics.data.ctrl = [gbest[k, :] for k = 1:ctrl_num]
            particles = repeat(dynamics, p_num)
        end

        set_f!(output, f_out[idx])
        set_buffer!(output, gbest)
        set_io!(output, f_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function update!(opt::StateOpt, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    dim = length(dynamics.data.ψ0)
    particles = repeat(dynamics, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = v0 .* rand(rng, ComplexF64, dim, p_num)
    pbest = zeros(ComplexF64, dim, p_num)
    gbest = zeros(ComplexF64, dim)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    # initialization 
    initial_state!(ini_particle, particles, p_num)

    f_ini, f_comp = objective(obj, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ψ0)
    set_io!(output, f_ini)
    show(opt, output, obj)
    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            p_out[pj], f_now = objective(obj, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for di = 1:dim
                    pbest[di, pj] = particles[pj].data.ψ0[di]
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for dj = 1:dim
                    gbest[dj] = pbest[dj, pj]
                end
            end
        end

        for pk = 1:p_num
            psi_pre = zeros(ComplexF64, dim)
            for dk = 1:dim
                psi_pre[dk] = particles[pk].data.ψ0[dk]
                velocity[dk, pk] =
                    c0 * velocity[dk, pk] +
                    c1 * rand(rng) * (pbest[dk, pk] - particles[pk].data.ψ0[dk]) +
                    c2 * rand(rng) * (gbest[dk] - particles[pk].data.ψ0[dk])
                particles[pk].data.ψ0[dk] = particles[pk].data.ψ0[dk] + velocity[dk, pk]
            end
            particles[pk].data.ψ0 = particles[pk].data.ψ0 / norm(particles[pk].data.ψ0)

            for dm = 1:dim
                velocity[dm, pk] = particles[pk].data.ψ0[dm] - psi_pre[dm]
            end
        end
        if ei % max_episode[2] == 0
            dynamics.data.data.ψ0 = [gbest[i] for i = 1:dim]
            particles = repeat(dynamics, p_num)
        end
        set_f!(output, p_out[idx])
        set_buffer!(output, gbest)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end

function update!(opt::MOpt_Projection, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    dim = size(dynamics.data.ρ0)[1]
    C = ini_particle[1]
    M_num = length(C)
    particles = repeat(C, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = 0.1 * rand(rng, ComplexF64, M_num, dim, p_num)
    pbest = zeros(ComplexF64, M_num, dim, p_num)
    gbest = zeros(ComplexF64, M_num, dim)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    # initialization  
    particles = initial_M!(ini_particle, particles, dim, p_num)

    M = [particles[1][i] * (particles[1][i])' for i = 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_ini)
    show(opt, output, obj)
    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            M = [particles[pj][i] * (particles[pj][i])' for i = 1:M_num]
            obj_copy = set_M(obj, M)
            p_out[pj], f_now = objective(obj_copy, dynamics)
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for di = 1:M_num
                    for ni = 1:dim
                        pbest[di, ni, pj] = particles[pj][di][ni]
                    end
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for dj = 1:M_num
                    for nj = 1:dim
                        gbest[dj, nj] = pbest[dj, nj, pj]
                    end
                end
            end
        end

        for pk = 1:p_num
            meas_pre = [zeros(ComplexF64, dim) for i = 1:M_num]
            for dk = 1:M_num
                for ck = 1:dim
                    meas_pre[dk][ck] = particles[pk][dk][ck]

                    velocity[dk, ck, pk] =
                        c0 * velocity[dk, ck, pk] +
                        c1 * rand(rng) * (pbest[dk, ck, pk] - particles[pk][dk][ck])
                    +c2 * rand(rng) * (gbest[dk, ck] - particles[pk][dk][ck])
                    particles[pk][dk][ck] += velocity[dk, ck, pk]
                end
            end
            particles[pk] = gramschmidt(particles[pk])

            for dm = 1:M_num
                for cm = 1:dim
                    velocity[dm, cm, pk] = particles[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [gbest[i] * (gbest[i])' for i = 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end

function update!(opt::MOpt_LinearComb, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    (; POVM_basis, M_num) = opt
    basis_num = length(POVM_basis)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = 0.1 * rand(rng, Float64, M_num, basis_num, p_num)
    pbest = zeros(Float64, M_num, basis_num, p_num)
    gbest = zeros(Float64, M_num, basis_num)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    # initialization  
    B_all = [[zeros(basis_num) for i = 1:M_num] for j = 1:p_num]
    for pj = 1:p_num
        B_all[pj] = [rand(rng, basis_num) for i = 1:M_num]
        B_all[pj] = bound_LC_coeff!(B_all[pj])
    end

    f_opt, f_comp = objective(obj::QFIM{SLD}, dynamics)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, dynamics)

    M = [sum([B_all[1][i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_opt, f_povm, f_ini)
    show(opt, output, obj)

    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            M = [
                sum([B_all[pj][i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num
            ]
            obj_copy = set_M(obj, M)
            p_out[pj], f_now = objective(obj_copy, dynamics)
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for di = 1:M_num
                    for ni = 1:basis_num
                        pbest[di, ni, pj] = B_all[pj][di][ni]
                    end
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for dj = 1:M_num
                    for nj = 1:basis_num
                        gbest[dj, nj] = pbest[dj, nj, pj]
                    end
                end
            end
        end

        for pk = 1:p_num
            meas_pre = [zeros(Float64, basis_num) for i = 1:M_num]
            for dk = 1:M_num
                for ck = 1:basis_num
                    meas_pre[dk][ck] = B_all[pk][dk][ck]

                    velocity[dk, ck, pk] =
                        c0 * velocity[dk, ck, pk] +
                        c1 * rand(rng) * (pbest[dk, ck, pk] - B_all[pk][dk][ck])
                    +c2 * rand(rng) * (gbest[dk, ck] - B_all[pk][dk][ck])
                    B_all[pk][dk][ck] += velocity[dk, ck, pk]
                end
            end
            B_all[pk] = bound_LC_coeff!(B_all[pk])

            for dm = 1:M_num
                for cm = 1:basis_num
                    velocity[dm, cm, pk] = B_all[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [sum([gbest[i][j] * POVM_basis[j] for j = 1:basis_num]) for i = 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end

function update!(opt::MOpt_Rotation, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    POVM_basis = opt.POVM_basis
    M_num = length(POVM_basis)
    dim = size(dynamics.data.ρ0)[1]
    suN = suN_generator(dim)
    Lambda = [Matrix{ComplexF64}(I, dim, dim)]
    append!(Lambda, [suN[i] for i = 1:length(suN)])

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = 0.1 * rand(rng, Float64, dim^2, p_num)
    pbest = zeros(Float64, dim^2, p_num)
    gbest = zeros(Float64, dim^2)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    # initialization  
    s_all = [rand(rng, dim * dim) for i = 1:p_num]

    f_opt, f_comp = objective(obj::QFIM{SLD}, dynamics)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, dynamics)

    U = rotation_matrix(s_all[1], Lambda)
    M = [U * POVM_basis[i] * U' for i = 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)
    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_opt, f_povm, f_ini)
    show(opt, output, obj)

    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            U = rotation_matrix(s_all[pj], Lambda)
            M = [U * POVM_basis[i] * U' for i = 1:M_num]
            obj_copy = set_M(obj, M)
            p_out[pj], f_now = objective(obj_copy, dynamics)
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for ni = 1:dim^2
                    pbest[ni, pj] = s_all[pj][ni]
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for nj = 1:dim^2
                    gbest[nj] = pbest[nj, pj]
                end
            end
        end

        for pk = 1:p_num
            meas_pre = zeros(Float64, dim^2)

            for ck = 1:dim^2
                meas_pre[ck] = s_all[pk][ck]

                velocity[ck, pk] =
                    c0 * velocity[ck, pk] +
                    c1 * rand(rng) * (pbest[ck, pk] - s_all[pk][ck]) +
                    c2 * rand(rng) * (gbest[ck] - s_all[pk][ck])
                s_all[pk][ck] += velocity[ck, pk]
            end

            s_all[pk] = bound_rot_coeff!(s_all[pk])

            for cm = 1:dim^2
                velocity[cm, pk] = s_all[pk][cm] - meas_pre[cm]
            end
        end
        U = rotation_matrix(gbest, Lambda)
        M = [U * POVM_basis[i] * U' for i = 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end

#### state and control optimization ####
function update!(opt::StateControlOpt, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    psi0, ctrl0 = ini_particle
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    particles = repeat(dynamics, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity_state = 0.1 .* rand(rng, ComplexF64, dim, p_num)
    pbest_state = zeros(ComplexF64, dim, p_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_ctrl = initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num)
    pbest_ctrl = zeros(ctrl_num, ctrl_length, p_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    # initialization  
    initial_state!(psi0, particles, p_num)
    initial_ctrl!(opt, ctrl0, particles, p_num)

    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, particles[1].data.ψ0, particles[1].data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj)

    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            p_out[pj], f_now = objective(obj, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for di = 1:dim
                    pbest_state[di, pj] = particles[pj].data.ψ0[di]
                end

                for di = 1:ctrl_num
                    for ni = 1:ctrl_length
                        pbest_ctrl[di, ni, pj] = particles[pj].data.ctrl[di][ni]
                    end
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for dj = 1:dim
                    gbest_state[dj] = pbest_state[dj, pj]
                end

                for dj = 1:ctrl_num
                    for nj = 1:ctrl_length
                        gbest_ctrl[dj, nj] = pbest_ctrl[dj, nj, pj]
                    end
                end
            end
        end

        for pk = 1:p_num
            psi_pre = zeros(ComplexF64, dim)
            for dk = 1:dim
                psi_pre[dk] = particles[pk].data.ψ0[dk]
                velocity_state[dk, pk] =
                    c0 * velocity_state[dk, pk] +
                    c1 * rand(rng) * (pbest_state[dk, pk] - particles[pk].data.ψ0[dk]) +
                    c2 * rand(rng) * (gbest_state[dk] - particles[pk].data.ψ0[dk])
                particles[pk].data.ψ0[dk] =
                    particles[pk].data.ψ0[dk] + velocity_state[dk, pk]
            end
            particles[pk].data.ψ0 = particles[pk].data.ψ0 / norm(particles[pk].data.ψ0)
            for dm = 1:dim
                velocity_state[dm, pk] = particles[pk].data.ψ0[dm] - psi_pre[dm]
            end

            control_coeff_pre = [zeros(ctrl_length) for i = 1:ctrl_num]
            for dk = 1:ctrl_num
                for ck = 1:ctrl_length
                    control_coeff_pre[dk][ck] = particles[pk].data.ctrl[dk][ck]
                    velocity_ctrl[dk, ck, pk] =
                        c0 * velocity_ctrl[dk, ck, pk] +
                        c1 *
                        rand(rng) *
                        (pbest_ctrl[dk, ck, pk] - particles[pk].ctrl[dk][ck])
                    +c2 * rand(rng) * (gbest_ctrl[dk, ck] - particles[pk].data.ctrl[dk][ck])
                    particles[pk].data.ctrl[dk][ck] += velocity_ctrl[dk, ck, pk]
                end
            end

            for dm = 1:ctrl_num
                for cm = 1:ctrl_length
                    particles[pk].data.ctrl[dm][cm] = (
                        x ->
                            x < opt.ctrl_bound[1] ? opt.ctrl_bound[1] :
                            x > opt.ctrl_bound[2] ? opt.ctrl_bound[2] : x
                    )(
                        particles[pk].data.ctrl[dm][cm],
                    )
                    velocity_ctrl[dm, cm, pk] =
                        particles[pk].data.ctrl[dm][cm] - control_coeff_pre[dm][cm]
                end
            end
        end
        set_f!(output, p_out[idx])
        set_buffer!(output, gbest_state, gbest_ctrl)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end

#### state and measurement optimization ####
function update!(opt::StateMeasurementOpt, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    psi0, measurement0 = ini_particle
    dim = length(dynamics.data.ψ0)
    M_num = length(measurement0[1])
    particles = repeat(dynamics, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity_state = 0.1 .* rand(rng, ComplexF64, dim, p_num)
    pbest_state = zeros(ComplexF64, dim, p_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_meas = 0.1 * rand(rng, ComplexF64, M_num, dim, p_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, p_num)
    gbest_meas = zeros(ComplexF64, M_num, dim)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    # initialization  
    initial_state!(psi0, particles, p_num)
    C_all = [measurement0[1] for i = 1:p_num]
    C_all = initial_M!(measurement0, C_all, dim, p_num)

    M = [C_all[1][i] * (C_all[1][i])' for i = 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, particles[1].data.ψ0, M)
    set_io!(output, f_ini)
    show(opt, output, obj)

    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            M = [C_all[pj][i] * (C_all[pj][i])' for i = 1:M_num]
            f_now = objective(obj, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for di = 1:dim
                    pbest_state[di, pj] = particles[pj].data.ψ0[di]
                end

                for di = 1:M_num
                    for ni = 1:dim
                        pbest_meas[di, ni, pj] = C_all[pj][di][ni]
                    end
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for dj = 1:dim
                    gbest_state[dj] = pbest_state[dj, pj]
                end

                for dj = 1:M_num
                    for nj = 1:dim
                        gbest_meas[dj, nj] = pbest_meas[dj, nj, pj]
                    end
                end
            end
        end

        for pk = 1:p_num
            psi_pre = zeros(ComplexF64, dim)
            for dk = 1:dim
                psi_pre[dk] = particles[pk].data.ψ0[dk]
                velocity_state[dk, pk] =
                    c0 * velocity_state[dk, pk] +
                    c1 * rand(rng) * (pbest_state[dk, pk] - particles[pk].data.ψ0[dk]) +
                    c2 * rand(rng) * (gbest_state[dk] - particles[pk].data.ψ0[dk])
                particles[pk].data.ψ0[dk] =
                    particles[pk].data.ψ0[dk] + velocity_state[dk, pk]
            end
            particles[pk].data.ψ0 = particles[pk].data.ψ0 / norm(particles[pk].data.ψ0)
            for dm = 1:dim
                velocity_state[dm, pk] = particles[pk].data.ψ0[dm] - psi_pre[dm]
            end

            meas_pre = [zeros(ComplexF64, dim) for i = 1:M_num]
            for dk = 1:M_num
                for ck = 1:dim
                    meas_pre[dk][ck] = C_all[pk][dk][ck]

                    velocity_meas[dk, ck, pk] =
                        c0 * velocity_meas[dk, ck, pk] +
                        c1 * rand(rng) * (pbest_meas[dk, ck, pk] - C_all[pk][dk][ck])
                    +c2 * rand(rng) * (gbest_meas[dk, ck] - C_all[pk][dk][ck])
                    C_all[pk][dk][ck] += velocity_meas[dk, ck, pk]
                end
            end
            C_all[pk] = gramschmidt(C_all[pk])

            for dm = 1:M_num
                for cm = 1:dim
                    velocity_meas[dm, cm, pk] = C_all[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [gbest_meas[i] * (gbest_meas[i])' for i = 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, gbest_state, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end

#### control and measurement optimization ####
function update!(opt::ControlMeasurementOpt, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    ctrl0, measurement0 = ini_particle
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    dim = length(dynamics.data.ψ0)
    M_num = length(measurement0[1])
    particles = repeat(dynamics, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity_ctrl = initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num)
    pbest_ctrl = zeros(ctrl_num, ctrl_length, p_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)
    velocity_meas = 0.1 * rand(rng, ComplexF64, M_num, dim, p_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, p_num)
    gbest_meas = zeros(ComplexF64, M_num, dim)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    initial_ctrl!(opt, ctrl0, particles, p_num)
    C_all = [measurement0[1] for i = 1:p_num]
    C_all = initial_M!(measurement0, C_all, dim, p_num)

    M = [C_all[1][i] * (C_all[1][i])' for i = 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, particles[1].data.ctrl, M)
    set_io!(output, f_ini)
    show(opt, output, obj)

    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            M = [C_all[pj][i] * (C_all[pj][i])' for i = 1:M_num]
            p_out[pj], f_now = objective(obj, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for di = 1:ctrl_num
                    for ni = 1:ctrl_length
                        pbest_ctrl[di, ni, pj] = particles[pj].data.ctrl[di][ni]
                    end
                end
                for di = 1:M_num
                    for ni = 1:dim
                        pbest_meas[di, ni, pj] = C_all[pj][di][ni]
                    end
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for dj = 1:ctrl_num
                    for nj = 1:ctrl_length
                        gbest_ctrl[dj, nj] = pbest_ctrl[dj, nj, pj]
                    end
                end
                for dj = 1:M_num
                    for nj = 1:dim
                        gbest_meas[dj, nj] = pbest_meas[dj, nj, pj]
                    end
                end
            end
        end

        for pk = 1:p_num
            control_coeff_pre = [zeros(ctrl_length) for i = 1:ctrl_num]
            for dk = 1:ctrl_num
                for ck = 1:ctrl_length
                    control_coeff_pre[dk][ck] = particles[pk].data.ctrl[dk][ck]
                    velocity_ctrl[dk, ck, pk] =
                        c0 * velocity_ctrl[dk, ck, pk] +
                        c1 *
                        rand(rng) *
                        (pbest_ctrl[dk, ck, pk] - particles[pk].ctrl[dk][ck])
                    +c2 * rand(rng) * (gbest_ctrl[dk, ck] - particles[pk].data.ctrl[dk][ck])
                    particles[pk].data.ctrl[dk][ck] += velocity_ctrl[dk, ck, pk]
                end
            end

            for dm = 1:ctrl_num
                for cm = 1:ctrl_length
                    particles[pk].data.ctrl[dm][cm] = (
                        x ->
                            x < opt.ctrl_bound[1] ? opt.ctrl_bound[1] :
                            x > opt.ctrl_bound[2] ? opt.ctrl_bound[2] : x
                    )(
                        particles[pk].data.ctrl[dm][cm],
                    )
                    velocity_ctrl[dm, cm, pk] =
                        particles[pk].data.ctrl[dm][cm] - control_coeff_pre[dm][cm]
                end
            end

            meas_pre = [zeros(ComplexF64, dim) for i = 1:M_num]
            for dk = 1:M_num
                for ck = 1:dim
                    meas_pre[dk][ck] = C_all[pk][dk][ck]

                    velocity_meas[dk, ck, pk] =
                        c0 * velocity_meas[dk, ck, pk] +
                        c1 * rand(rng) * (pbest_meas[dk, ck, pk] - C_all[pk][dk][ck])
                    +c2 * rand(rng) * (gbest_meas[dk, ck] - C_all[pk][dk][ck])
                    C_all[pk][dk][ck] += velocity_meas[dk, ck, pk]
                end
            end
            C_all[pk] = gramschmidt(C_all[pk])

            for dm = 1:M_num
                for cm = 1:dim
                    velocity_meas[dm, cm, pk] = C_all[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [gbest_meas[i] * (gbest_meas[i])' for i = 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, gbest_ctrl, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end

#### state, control and measurement optimization ####
function update!(opt::StateControlMeasurementOpt, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    psi0, ctrl0, measurement0 = ini_particle
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    dim = length(dynamics.data.ψ0)
    M_num = length(measurement0[1])
    particles = repeat(dynamics, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity_state = 0.1 .* rand(rng, ComplexF64, dim, p_num)
    pbest_state = zeros(ComplexF64, dim, p_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_ctrl = initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num)
    pbest_ctrl = zeros(ctrl_num, ctrl_length, p_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)
    velocity_meas = 0.1 * rand(rng, ComplexF64, M_num, dim, p_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, p_num)
    gbest_meas = zeros(ComplexF64, M_num, dim)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit = 0.0

    # initialization 
    initial_state!(psi0, particles, p_num)
    initial_ctrl!(opt, ctrl0, particles, p_num)
    C_all = [measurement0[1] for i = 1:p_num]
    C_all = initial_M!(measurement0, C_all, dim, p_num)

    M = [C_all[1][i] * (C_all[1][i])' for i = 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, particles[1].data.ψ0, particles[1].data.ctrl, M)
    set_io!(output, f_ini)
    show(opt, output, obj)

    idx = 0
    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            M = [C_all[pj][i] * (C_all[pj][i])' for i = 1:M_num]
            obj_copy = set_M(obj, M)
            f_now = objective(obj_copy, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                for di = 1:dim
                    pbest_state[di, pj] = particles[pj].data.ψ0[di]
                end
                for di = 1:ctrl_num
                    for ni = 1:ctrl_length
                        pbest_ctrl[di, ni, pj] = particles[pj].data.ctrl[di][ni]
                    end
                end
                for di = 1:M_num
                    for ni = 1:dim
                        pbest_meas[di, ni, pj] = C_all[pj][di][ni]
                    end
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                idx = pj
                for dj = 1:dim
                    gbest_state[dj] = pbest_state[dj, pj]
                end
                for dj = 1:ctrl_num
                    for nj = 1:ctrl_length
                        gbest_ctrl[dj, nj] = pbest_ctrl[dj, nj, pj]
                    end
                end
                for dj = 1:M_num
                    for nj = 1:dim
                        gbest_meas[dj, nj] = pbest_meas[dj, nj, pj]
                    end
                end
            end
        end

        for pk = 1:p_num
            psi_pre = zeros(ComplexF64, dim)
            for dk = 1:dim
                psi_pre[dk] = particles[pk].data.ψ0[dk]
                velocity_state[dk, pk] =
                    c0 * velocity_state[dk, pk] +
                    c1 * rand(rng) * (pbest_state[dk, pk] - particles[pk].data.ψ0[dk]) +
                    c2 * rand(rng) * (gbest_state[dk] - particles[pk].data.ψ0[dk])
                particles[pk].data.ψ0[dk] =
                    particles[pk].data.ψ0[dk] + velocity_state[dk, pk]
            end
            particles[pk].data.ψ0 = particles[pk].data.ψ0 / norm(particles[pk].data.ψ0)
            for dm = 1:dim
                velocity_state[dm, pk] = particles[pk].data.ψ0[dm] - psi_pre[dm]
            end

            control_coeff_pre = [zeros(ctrl_length) for i = 1:ctrl_num]
            for dk = 1:ctrl_num
                for ck = 1:ctrl_length
                    control_coeff_pre[dk][ck] = particles[pk].data.ctrl[dk][ck]
                    velocity_ctrl[dk, ck, pk] =
                        c0 * velocity_ctrl[dk, ck, pk] +
                        c1 *
                        rand(rng) *
                        (pbest_ctrl[dk, ck, pk] - particles[pk].ctrl[dk][ck])
                    +c2 * rand(rng) * (gbest_ctrl[dk, ck] - particles[pk].data.ctrl[dk][ck])
                    particles[pk].data.ctrl[dk][ck] += velocity_ctrl[dk, ck, pk]
                end
            end

            for dm = 1:ctrl_num
                for cm = 1:ctrl_length
                    particles[pk].data.ctrl[dm][cm] = (
                        x ->
                            x < opt.ctrl_bound[1] ? opt.ctrl_bound[1] :
                            x > opt.ctrl_bound[2] ? opt.ctrl_bound[2] : x
                    )(
                        particles[pk].data.ctrl[dm][cm],
                    )
                    velocity_ctrl[dm, cm, pk] =
                        particles[pk].data.ctrl[dm][cm] - control_coeff_pre[dm][cm]
                end
            end

            meas_pre = [zeros(ComplexF64, dim) for i = 1:M_num]
            for dk = 1:M_num
                for ck = 1:dim
                    meas_pre[dk][ck] = C_all[pk][dk][ck]

                    velocity_meas[dk, ck, pk] =
                        c0 * velocity_meas[dk, ck, pk] +
                        c1 * rand(rng) * (pbest_meas[dk, ck, pk] - C_all[pk][dk][ck])
                    +c2 * rand(rng) * (gbest_meas[dk, ck] - C_all[pk][dk][ck])
                    C_all[pk][dk][ck] += velocity_meas[dk, ck, pk]
                end
            end
            C_all[pk] = gramschmidt(C_all[pk])

            for dm = 1:M_num
                for cm = 1:dim
                    velocity_meas[dm, cm, pk] = C_all[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [gbest_meas[i] * (gbest_meas[i])' for i = 1:M_num]
        set_f!(output, p_out[idx])
        set_buffer!(output, gbest_ctrl, gbest_meas, M)
        set_io!(output, p_out[idx], ei)
        show(output, obj)
    end
    set_io!(output, p_out[idx])
end
