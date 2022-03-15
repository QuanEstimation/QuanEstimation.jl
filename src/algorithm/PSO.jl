#### ControlOpt ####
function update!(opt::ControlOpt, alg::PSO, obj, dynamics, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2, rng) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    particles = repeat(dynamics, p_num)

    if opt.ctrl_bound[1] == -Inf || opt.ctrl_bound[2] == Inf
        velocity =
            0.1 * (
                2.0 * rand(rng, ctrl_num, ctrl_length, p_num) -
                ones(ctrl_num, ctrl_length, p_num)
            )
    else
        a = opt.ctrl_bound[1]
        b = opt.ctrl_bound[2]
        velocity =
            0.1 * (
                (b - a) * rand(rng, ctrl_num, ctrl_length, p_num) +
                a * ones(ctrl_num, ctrl_length, p_num)
            )
    end
    pbest = zeros(ctrl_num, ctrl_length, p_num)
    gbest = zeros(ctrl_num, ctrl_length)
    p_fit = zeros(p_num)
    f_out = zeros(p_num)
    fit = 0.0

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    # initialize 
    if length(ini_particle) > p_num
        ini_particle = [ini_particle[i] for i = 1:p_num]
    end
    for pj = 1:length(ini_particle)
        particles[pj].data.ctrl =
            [[ini_particle[pj][i, j] for j = 1:ctrl_length] for i = 1:ctrl_num]
    end
    if opt.ctrl_bound[1] == -Inf || opt.ctrl_bound[2] == Inf
        for pj = (length(ini_particle)+1):p_num
            particles[pj].data.ctrl =
                [[2 * rand(rng) - 1.0 for j = 1:ctrl_length] for i = 1:ctrl_num]
        end
    else
        a = opt.ctrl_bound[1]
        b = opt.ctrl_bound[2]
        for pj = (length(ini_particle)+1):p_num
            particles[pj].data.ctrl =
                [[(b - a) * rand(rng) + a for j = 1:ctrl_length] for i = 1:ctrl_num]
        end
    end

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
