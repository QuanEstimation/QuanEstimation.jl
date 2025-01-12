#### control optimization ####
function optimize!(opt::ControlOpt, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg
    
    pdata = param_data(scheme)
    
    if isnothing(ini_particle)
        ini_particle = ([opt.ctrl,],)
    end
    ini_particle = ini_particle[1]
    ctrl_length = get_ctrl_length(scheme) 
    ctrl_num = get_ctrl_num(scheme)
    particles = repeat(scheme, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num, opt.rng)
    pbest = zeros(ctrl_num, ctrl_length, p_num)
    gbest = zeros(ctrl_num, ctrl_length)
    
    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    # initialize 
    initial_ctrl!(opt, ini_particle, particles, p_num, opt.rng)

    scheme_copy = set_ctrl(scheme, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, scheme_copy)
    f_ini, f_comp = objective(obj, scheme)

    set_f!(output, f_ini)
    set_buffer!(output, pdata.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj, alg)

    for ei = 1:(max_episode[1]-1)
        for pj = 1:p_num
            f_out, f_now = objective(obj, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for di = 1:ctrl_num
                    for ni = 1:ctrl_length
                        pbest[di, ni, pj] = param_data(particles[pj]).ctrl[di][ni]
                    end
                end
            end
        end

        for pj = 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
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
                    control_coeff_pre[dk][ck] = param_data(particles[pk]).ctrl[dk][ck]
                    velocity[dk, ck, pk] =
                        c0 * velocity[dk, ck, pk] +
                        c1 *
                        rand(opt.rng) *
                        (pbest[dk, ck, pk] - param_data(particles[pk]).ctrl[dk][ck])
                    +c2 * rand(opt.rng) * (gbest[dk, ck] - param_data(particles[pk]).ctrl[dk][ck])
                    param_data(particles[pk]).ctrl[dk][ck] += velocity[dk, ck, pk]
                end
            end

            for dm = 1:ctrl_num
                for cm = 1:ctrl_length
                    param_data(particles[pk]).ctrl[dm][cm] = (
                        x ->
                            x < opt.ctrl_bound[1] ? opt.ctrl_bound[1] :
                            x > opt.ctrl_bound[2] ? opt.ctrl_bound[2] : x
                    )(
                        param_data(particles[pk]).ctrl[dm][cm],
                    )
                    velocity[dm, cm, pk] =
                        param_data(particles[pk]).ctrl[dm][cm] - control_coeff_pre[dm][cm]
                end
            end
        end
        if ei % max_episode[2] == 0
            pdata.ctrl = [gbest[k, :] for k = 1:ctrl_num]
            particles = repeat(scheme, p_num)
        end

        set_f!(output, fit_out)
        set_buffer!(output, [gbest[k, :] for k = 1:ctrl_num])
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### state optimization ####
function optimize!(opt::StateOpt, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg

    sdata = state_data(scheme)

    if isnothing(ini_particle)
        ini_particle = ([opt.psi], )
    end
    ini_particle = ini_particle[1]
    dim = get_dim(scheme)
    particles = repeat(scheme, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = 0.1.*rand(opt.rng, ComplexF64, dim, p_num)
    pbest = zeros(ComplexF64, dim, p_num)
    gbest = zeros(ComplexF64, dim)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    # initialization 
    initial_state!(ini_particle, particles, p_num, opt.rng)

    f_ini, _ = objective(obj, scheme)
    set_f!(output, f_ini)
    set_buffer!(output, sdata)
    set_io!(output, f_ini)
    show(opt, output, obj, alg)

    for ei in 1:(max_episode[1]-1)
        for pj in 1:p_num
            f_out, f_now = objective(obj, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for di in 1:dim
                    pbest[di,pj] = particles[pj].data.ψ0[di]
                end
            end
        end

        for pj in 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
                for dj in 1:dim
                    gbest[dj] = pbest[dj,pj]
                end
            end
        end

        for pk in 1:p_num
            psi_pre = zeros(ComplexF64, dim)
            for dk in 1:dim
                psi_pre[dk] = particles[pk].data.ψ0[dk]
                velocity[dk, pk] = c0*velocity[dk, pk] + c1*rand(opt.rng)*(pbest[dk, pk] - particles[pk].data.ψ0[dk]) + c2*rand(opt.rng)*(gbest[dk] - particles[pk].data.ψ0[dk])
                particles[pk].data.ψ0[dk] = particles[pk].data.ψ0[dk] + velocity[dk, pk]
            end
            particles[pk].data.ψ0 = particles[pk].data.ψ0/norm(particles[pk].data.ψ0)
    
            for dm in 1:dim
                velocity[dm, pk] = particles[pk].data.ψ0[dm] - psi_pre[dm]
            end
        end
        if ei%max_episode[2] == 0
            sdata = [gbest[i] for i in 1:dim]
            particles = repeat(scheme, p_num)
        end
        set_f!(output, fit_out)
        set_buffer!(output, [gbest[i] for i in 1:dim])
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### projective measurement optimization ####
function optimize!(opt::Mopt_Projection, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg

    sdata = state_data(scheme)

    if isnothing(ini_particle)
        ini_particle = ([opt.M], )
    end
    ini_particle = ini_particle[1]  
    dim = get_dim(scheme)
    M_num = length(opt.M)
    particles = [[zeros(ComplexF64, dim) for j in 1:M_num] for i in 1:p_num]

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = 0.1*rand(opt.rng, ComplexF64, M_num, dim, p_num)
    pbest = zeros(ComplexF64, M_num, dim, p_num)
    gbest = [zeros(ComplexF64, dim) for i in 1:M_num]

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    # initialization  
    initial_M!(ini_particle, particles, dim, p_num, M_num, opt.rng)

    M = [particles[1][i]*(particles[1][i])' for i in 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, scheme)

    obj_QFIM = QFIM_obj(obj)   
    f_opt, f_comp = objective(obj_QFIM, scheme)

    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_ini, f_opt)
    show(opt, output, obj, alg)

    for ei in 1:(max_episode[1]-1)
        for pj in 1:p_num
            M = [particles[pj][i]*(particles[pj][i])' for i in 1:M_num]
            obj_copy = set_M(obj, M)
            f_out, f_now = objective(obj_copy, scheme)
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for di in 1:M_num
                    for ni in 1:dim
                        pbest[di,ni,pj] = particles[pj][di][ni]
                    end
                end
            end
        end

        for pj in 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
                for dj in 1:M_num
                    for nj in 1:dim
                        gbest[dj][nj] = pbest[dj,nj,pj]
                    end
                end
            end
        end

        for pk in 1:p_num
            meas_pre = [zeros(ComplexF64, dim) for i in 1:M_num]
            for dk in 1:M_num
                for ck in 1:dim
                    meas_pre[dk][ck] = particles[pk][dk][ck]
        
                    velocity[dk, ck, pk] = c0*velocity[dk, ck, pk] + c1*rand(opt.rng)*(pbest[dk, ck, pk] - particles[pk][dk][ck]) 
                                               + c2*rand(opt.rng)*(gbest[dk][ck] - particles[pk][dk][ck])
                    particles[pk][dk][ck] += velocity[dk, ck, pk]
                end
            end
            particles[pk] = gramschmidt(particles[pk])
    
            for dm in 1:M_num
                for cm in 1:dim
                    velocity[dm, cm, pk] = particles[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [gbest[i]*(gbest[i])' for i in 1:M_num]
        set_f!(output, fit_out)
        set_buffer!(output, M)
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### find the optimal linear combination of a given set of POVM ####
function optimize!(opt::Mopt_LinearComb, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg
    (; B, POVM_basis, M_num) = opt
    basis_num = length(POVM_basis)
    if isnothing(ini_particle)
        ini_particle = ( [B], )
    end
    ini_particle = ini_particle[1]
    particles = [[zeros(basis_num) for j in 1:M_num] for i in 1:p_num]

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = 0.1*rand(opt.rng, Float64, M_num, basis_num, p_num)
    pbest = zeros(Float64, M_num, basis_num, p_num)
    gbest = zeros(Float64, M_num, basis_num)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    # initialization  
    initial_LinearComb!(ini_particle, particles, basis_num, M_num, p_num, opt.rng)

    obj_QFIM = QFIM_obj(obj)
    f_opt, f_comp = objective(obj_QFIM, scheme)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, scheme)

    M = [sum([particles[1][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, scheme)

    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_ini, f_povm, f_opt)
    show(opt, output, obj, alg)
    
    for ei in 1:(max_episode[1]-1)
        for pj in 1:p_num
            M = [sum([particles[pj][i][j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
            obj_copy = set_M(obj, M)
            f_out, f_now = objective(obj_copy, scheme)
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for di in 1:M_num
                    for ni in 1:basis_num
                        pbest[di,ni,pj] = particles[pj][di][ni]
                    end
                end
            end
        end

        for pj in 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
                for dj in 1:M_num
                    for nj in 1:basis_num
                        gbest[dj, nj] = pbest[dj,nj,pj]
                    end
                end
            end
        end 

        for pk in 1:p_num
            meas_pre = [zeros(Float64, basis_num) for i in 1:M_num]
            for dk in 1:M_num
                for ck in 1:basis_num
                    meas_pre[dk][ck] = particles[pk][dk][ck]
        
                    velocity[dk, ck, pk] = c0*velocity[dk, ck, pk] + c1*rand(opt.rng)*(pbest[dk, ck, pk] - particles[pk][dk][ck]) 
                                               + c2*rand(opt.rng)*(gbest[dk, ck] - particles[pk][dk][ck])
                    particles[pk][dk][ck] += velocity[dk, ck, pk]
                end
            end
            bound_LC_coeff!(particles[pk], opt.rng)
    
            for dm in 1:M_num
                for cm in 1:basis_num
                    velocity[dm, cm, pk] = particles[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [sum([gbest[i,j]*POVM_basis[j] for j in 1:basis_num]) for i in 1:M_num]
        set_f!(output, fit_out)
        set_buffer!(output, M)
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### find the optimal rotated measurement of a given set of POVM ####
function optimize!(opt::Mopt_Rotation, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg
    if isnothing(ini_particle)
        ini_particle = ([opt.s], )
    end
    ini_particle = ini_particle[1]
    (; s, POVM_basis, Lambda) = opt
    M_num = length(POVM_basis)
    dim = get_dim(scheme)
    suN = suN_generator(dim)
    Lambda = Matrix{ComplexF64}[]
    append!(Lambda, [Matrix{ComplexF64}(I,dim,dim)])
    append!(Lambda, [suN[i] for i in eachindex(suN)])

    # if isnothing(Lambda)
    #     Lambda = Matrix{ComplexF64}[]
    #     append!(Lambda, [Matrix{ComplexF64}(I,dim,dim)])
    #     append!(Lambda, [suN[i] for i in eachindex(suN)])
    # end
    
    particles = repeat(s, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity = 0.1*rand(opt.rng, Float64, dim^2, p_num)
    pbest = zeros(Float64, dim^2, p_num)
    gbest = zeros(Float64, dim^2)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    # initialization  
    particles = [zeros(dim^2) for i in 1:p_num]
    initial_Rotation!(ini_particle, particles, dim, p_num, opt.rng)
    
    obj_QFIM = QFIM_obj(obj)
    f_opt, f_comp = objective(obj_QFIM, scheme)
    obj_POVM = set_M(obj, POVM_basis)
    f_povm, f_comp = objective(obj_POVM, scheme)

    U = rotation_matrix(particles[1], Lambda)
    M = [U*POVM_basis[i]*U' for i in 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, scheme)
    set_f!(output, f_ini)
    set_buffer!(output, M)
    set_io!(output, f_ini, f_povm, f_opt)
    show(opt, output, obj, alg)

    for ei in 1:(max_episode[1]-1)
        for pj in 1:p_num
            U = rotation_matrix(particles[pj], Lambda)
            M = [U*POVM_basis[i]*U' for i in 1:M_num]
            obj_copy = set_M(obj, M)
            f_out, f_now = objective(obj_copy, scheme)
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for ni in 1:dim^2
                    pbest[ni,pj] = particles[pj][ni]
                end
            end
        end

        for pj in 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
                for nj in 1:dim^2
                    gbest[nj] = pbest[nj,pj]
                end
            end
        end

        for pk in 1:p_num
            meas_pre = zeros(Float64, dim^2)
    
            for ck in 1:dim^2
                meas_pre[ck] = particles[pk][ck]
        
                velocity[ck, pk] = c0*velocity[ck, pk] + c1*rand(opt.rng)*(pbest[ck, pk] - particles[pk][ck]) + c2*rand(opt.rng)*(gbest[ck] - particles[pk][ck])
                particles[pk][ck] += velocity[ck, pk]
            end
    
            bound_rot_coeff!(particles[pk])
    
            for cm in 1:dim^2
                velocity[cm, pk] = particles[pk][cm] - meas_pre[cm]
            end
        end
        U = rotation_matrix(gbest, Lambda)
        M = [U*POVM_basis[i]*U' for i in 1:M_num]
        set_f!(output, fit_out)
        set_buffer!(output, M)
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### state and control optimization ####
function optimize!(opt::StateControlOpt, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg
    if isnothing(ini_particle)
        ini_particle = ([opt.psi], [opt.ctrl,])
    end
    psi0, ctrl0 = ini_particle
    dim = get_dim(scheme)
    ctrl_length = get_ctrl_length(scheme)
    ctrl_num = get_ctrl_num(scheme)
    particles = repeat(scheme, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity_state = 0.1.*rand(opt.rng, ComplexF64, dim, p_num)
    pbest_state = zeros(ComplexF64, dim, p_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_ctrl = initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num, opt.rng)
    pbest_ctrl = zeros(ctrl_num, ctrl_length, p_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    # initialization  
    initial_state!(psi0, particles, p_num, opt.rng)
    initial_ctrl!(opt, ctrl0, particles, p_num, opt.rng)

    scheme_copy = set_ctrl!(scheme, [zeros(ctrl_length) for i=1:ctrl_num])
    f_noctrl, f_comp = objective(obj, scheme_copy)
    f_ini, f_comp = objective(obj, scheme)

    set_f!(output, f_ini)
    set_buffer!(output, particles[1].data.ψ0, particles[1].data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output, obj, alg)

    for ei in 1:(max_episode[1]-1)
        for pj in 1:p_num
            f_out, f_now = objective(obj, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for di in 1:dim
                    pbest_state[di,pj] = particles[pj].data.ψ0[di]
                end
    
                for di in 1:ctrl_num
                    for ni in 1:ctrl_length
                        pbest_ctrl[di,ni,pj] = param_data(particles[pj]).ctrl[di][ni]
                    end
                end
            end
        end
    
        for pj in 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
                for dj in 1:dim
                    gbest_state[dj] = pbest_state[dj,pj]
                end
    
                for dj in 1:ctrl_num
                    for nj in 1:ctrl_length
                        gbest_ctrl[dj, nj] = pbest_ctrl[dj,nj,pj]
                    end
                end
            end
        end  
    
        for pk in 1:p_num
            psi_pre = zeros(ComplexF64, dim)
            for dk in 1:dim
                psi_pre[dk] = particles[pk].data.ψ0[dk]
                velocity_state[dk, pk] = c0*velocity_state[dk, pk] + c1*rand(opt.rng)*(pbest_state[dk, pk] - particles[pk].data.ψ0[dk]) + 
                                         c2*rand(opt.rng)*(gbest_state[dk] - particles[pk].data.ψ0[dk])
                particles[pk].data.ψ0[dk] = particles[pk].data.ψ0[dk] + velocity_state[dk, pk]
            end
            particles[pk].data.ψ0 = particles[pk].data.ψ0/norm(particles[pk].data.ψ0)
            for dm in 1:dim
                velocity_state[dm, pk] = particles[pk].data.ψ0[dm] - psi_pre[dm]
            end
    
            control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
            for dk in 1:ctrl_num
                for ck in 1:ctrl_length
                    control_coeff_pre[dk][ck] = param_data(particles[pk]).ctrl[dk][ck]
                    velocity_ctrl[dk, ck, pk] = c0*velocity_ctrl[dk, ck, pk] + c1*rand(opt.rng)*(pbest_ctrl[dk, ck, pk] - param_data(particles[pk]).ctrl[dk][ck]) 
                                         + c2*rand(opt.rng)*(gbest_ctrl[dk, ck] - param_data(particles[pk]).ctrl[dk][ck])
                    param_data(particles[pk]).ctrl[dk][ck] += velocity_ctrl[dk, ck, pk]
                end
            end
    
            for dm in 1:ctrl_num
                for cm in 1:ctrl_length
                    param_data(particles[pk]).ctrl[dm][cm] = (x-> x < opt.ctrl_bound[1] ? opt.ctrl_bound[1] : x > opt.ctrl_bound[2] ? opt.ctrl_bound[2] : x)(param_data(particles[pk]).ctrl[dm][cm])
                    velocity_ctrl[dm, cm, pk] = param_data(particles[pk]).ctrl[dm][cm] - control_coeff_pre[dm][cm]
                end
            end
        end
        set_f!(output, fit_out)
        set_buffer!(output, gbest_state, gbest_ctrl)
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### state and measurement optimization ####
function optimize!(opt::StateMeasurementOpt, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg
    if isnothing(ini_particle)
        ini_particle = ([opt.psi], [opt.M])
    end
    psi0, measurement0 = ini_particle
    dim = get_dim(scheme)
    M_num = length(opt.M)
    particles = repeat(scheme, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity_state = 0.1.*rand(opt.rng, ComplexF64, dim, p_num)
    pbest_state = zeros(ComplexF64, dim, p_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_meas = 0.1*rand(opt.rng, ComplexF64, M_num, dim, p_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, p_num)
    gbest_meas = [zeros(ComplexF64, dim) for i in 1:M_num]

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    # initialization  
    initial_state!(psi0, particles, p_num, opt.rng)
    C_all = [[zeros(ComplexF64, dim) for j in 1:M_num] for i in 1:p_num]
    initial_M!(measurement0, C_all, dim, p_num, M_num, opt.rng)

    M = [C_all[1][i]*(C_all[1][i])' for i in 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, scheme)

    set_f!(output, f_ini)
    set_buffer!(output, particles[1].data.ψ0, M)
    set_io!(output, f_ini)
    show(opt, output, obj, alg)

    for ei in 1:(max_episode[1]-1)
        for pj in 1:p_num
            M = [C_all[pj][i]*(C_all[pj][i])' for i in 1:M_num]
            obj_copy = set_M(obj, M)
            f_out, f_now = objective(obj_copy, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for di in 1:dim
                    pbest_state[di,pj] = particles[pj].data.ψ0[di]
                end
    
                for di in 1:M_num
                    for ni in 1:dim
                        pbest_meas[di,ni,pj] = C_all[pj][di][ni]
                    end
                end
            end
        end
    
        for pj in 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
                for dj in 1:dim
                    gbest_state[dj] = pbest_state[dj,pj]
                end
    
                for dj in 1:M_num
                    for nj in 1:dim
                        gbest_meas[dj][nj] = pbest_meas[dj,nj,pj]
                    end
                end
            end
        end  
    
        for pk in 1:p_num
            psi_pre = zeros(ComplexF64, dim)
            for dk in 1:dim
                psi_pre[dk] = particles[pk].data.ψ0[dk]
                velocity_state[dk, pk] = c0*velocity_state[dk, pk] + c1*rand(opt.rng)*(pbest_state[dk, pk] - particles[pk].data.ψ0[dk]) + 
                                         c2*rand(opt.rng)*(gbest_state[dk] - particles[pk].data.ψ0[dk])
                particles[pk].data.ψ0[dk] = particles[pk].data.ψ0[dk] + velocity_state[dk, pk]
            end
            particles[pk].data.ψ0 = particles[pk].data.ψ0/norm(particles[pk].data.ψ0)
            for dm in 1:dim
                velocity_state[dm, pk] = particles[pk].data.ψ0[dm] - psi_pre[dm]
            end
    
            meas_pre = [zeros(ComplexF64, dim) for i in 1:M_num]
            for dk in 1:M_num
                for ck in 1:dim
                    meas_pre[dk][ck] = C_all[pk][dk][ck]
        
                    velocity_meas[dk, ck, pk] = c0*velocity_meas[dk, ck, pk] + c1*rand(opt.rng)*(pbest_meas[dk, ck, pk] - C_all[pk][dk][ck]) 
                                               + c2*rand(opt.rng)*(gbest_meas[dk][ck] - C_all[pk][dk][ck])
                                               C_all[pk][dk][ck] += velocity_meas[dk, ck, pk]
                end
            end
            C_all[pk] = gramschmidt(C_all[pk])
    
            for dm in 1:M_num
                for cm in 1:dim
                    velocity_meas[dm, cm, pk] = C_all[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [gbest_meas[i]*(gbest_meas[i])' for i in 1:M_num]
        set_f!(output, fit_out)
        set_buffer!(output, gbest_state, M)
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### control and measurement optimization ####
function optimize!(opt::ControlMeasurementOpt, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg
    if isnothing(ini_particle)
        ini_particle = ([opt.ctrl,], [opt.M])
    end
    ctrl0, measurement0 = ini_particle
    ctrl_length = get_ctrl_length(scheme)
    ctrl_num = get_ctrl_num(scheme)
    dim = get_dim(scheme)
    M_num = length(opt.M)
    particles = repeat(scheme, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity_ctrl = initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num, opt.rng)
    pbest_ctrl = zeros(ctrl_num, ctrl_length, p_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)
    velocity_meas = 0.1*rand(opt.rng, ComplexF64, M_num, dim, p_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, p_num)
    gbest_meas = [zeros(ComplexF64, dim) for i in 1:M_num]

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    initial_ctrl!(opt, ctrl0, particles, p_num, opt.rng)
    C_all = [[zeros(ComplexF64, dim) for j in 1:M_num] for i in 1:p_num]
    initial_M!(measurement0, C_all, dim, p_num, M_num, opt.rng)

    M = [C_all[1][i]*(C_all[1][i])' for i in 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, f_comp = objective(obj_copy, scheme)

    set_f!(output, f_ini)
    set_buffer!(output, particles[1].data.ctrl, M)
    set_io!(output, f_ini)
    show(opt, output, obj, alg)

    for ei in 1:(max_episode[1]-1)
        for pj in 1:p_num
            M = [C_all[pj][i]*(C_all[pj][i])' for i in 1:M_num]
            obj_copy = set_M(obj, M)
            f_out, f_now = objective(obj_copy, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for di in 1:ctrl_num
                    for ni in 1:ctrl_length
                        pbest_ctrl[di,ni,pj] = param_data(particles[pj]).ctrl[di][ni]
                    end
                end
                for di in 1:M_num
                    for ni in 1:dim
                        pbest_meas[di,ni,pj] = C_all[pj][di][ni]
                    end
                end
            end
        end
    
        for pj in 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
                for dj in 1:ctrl_num
                    for nj in 1:ctrl_length
                        gbest_ctrl[dj, nj] = pbest_ctrl[dj,nj,pj]
                    end
                end
                for dj in 1:M_num
                    for nj in 1:dim
                        gbest_meas[dj][nj] = pbest_meas[dj,nj,pj]
                    end
                end
            end
        end  
    
        for pk in 1:p_num
            control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
            for dk in 1:ctrl_num
                for ck in 1:ctrl_length
                    control_coeff_pre[dk][ck] = param_data(particles[pk]).ctrl[dk][ck]
                    velocity_ctrl[dk, ck, pk] = c0*velocity_ctrl[dk, ck, pk] + c1*rand(opt.rng)*(pbest_ctrl[dk, ck, pk] - param_data(particles[pk]).ctrl[dk][ck]) 
                                         + c2*rand(opt.rng)*(gbest_ctrl[dk, ck] - param_data(particles[pk]).ctrl[dk][ck])
                    param_data(particles[pk]).ctrl[dk][ck] += velocity_ctrl[dk, ck, pk]
                end
            end
    
            for dm in 1:ctrl_num
                for cm in 1:ctrl_length
                    param_data(particles[pk]).ctrl[dm][cm] = (x-> x < opt.ctrl_bound[1] ? opt.ctrl_bound[1] : x > opt.ctrl_bound[2] ? opt.ctrl_bound[2] : x)(param_data(particles[pk]).ctrl[dm][cm])
                    velocity_ctrl[dm, cm, pk] = param_data(particles[pk]).ctrl[dm][cm] - control_coeff_pre[dm][cm]
                end
            end
    
            meas_pre = [zeros(ComplexF64, dim) for i in 1:M_num]
            for dk in 1:M_num
                for ck in 1:dim
                    meas_pre[dk][ck] = C_all[pk][dk][ck]
        
                    velocity_meas[dk, ck, pk] = c0*velocity_meas[dk, ck, pk] + c1*rand(opt.rng)*(pbest_meas[dk, ck, pk] - C_all[pk][dk][ck]) 
                                               + c2*rand(opt.rng)*(gbest_meas[dk][ck] - C_all[pk][dk][ck])
                                               C_all[pk][dk][ck] += velocity_meas[dk, ck, pk]
                end
            end
            C_all[pk] = gramschmidt(C_all[pk])
    
            for dm in 1:M_num
                for cm in 1:dim
                    velocity_meas[dm, cm, pk] = C_all[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [gbest_meas[i]*(gbest_meas[i])' for i in 1:M_num]
        set_f!(output, fit_out)
        set_buffer!(output, gbest_ctrl, M)
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

#### state, control and measurement optimization ####
function optimize!(opt::StateControlMeasurementOpt, alg::PSO, obj, scheme, output)
    (; max_episode, p_num, ini_particle, c0, c1, c2) = alg
    if isnothing(ini_particle)
        ini_particle = ([opt.psi], [opt.ctrl,], [opt.M])
    end
    psi0, ctrl0, measurement0 = ini_particle
    ctrl_length = get_ctrl_length(scheme)
    ctrl_num = get_ctrl_num(scheme)
    dim = get_dim(scheme)
    M_num = length(opt.M)
    particles = repeat(scheme, p_num)

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    velocity_state = 0.1.*rand(opt.rng, ComplexF64, dim, p_num)
    pbest_state = zeros(ComplexF64, dim, p_num)
    gbest_state = zeros(ComplexF64, dim)
    velocity_ctrl = initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num, opt.rng)
    pbest_ctrl = zeros(ctrl_num, ctrl_length, p_num)
    gbest_ctrl = zeros(ctrl_num, ctrl_length)
    velocity_meas = 0.1*rand(opt.rng, ComplexF64, M_num, dim, p_num)
    pbest_meas = zeros(ComplexF64, M_num, dim, p_num)
    gbest_meas = [zeros(ComplexF64, dim) for i in 1:M_num]

    p_fit, p_out = zeros(p_num), zeros(p_num)
    fit, fit_out = 0.0, 0.0

    # initialization 
    initial_state!(psi0, particles, p_num, opt.rng)
    initial_ctrl!(opt, ctrl0, particles, p_num, opt.rng)
    C_all = [[zeros(ComplexF64, dim) for j in 1:M_num] for i in 1:p_num]
    initial_M!(measurement0, C_all, dim, p_num, M_num, opt.rng)

    M = [C_all[1][i]*(C_all[1][i])' for i in 1:M_num]
    obj_copy = set_M(obj, M)
    f_ini, _ = objective(obj_copy, scheme)

    set_f!(output, f_ini)
    set_buffer!(output,  particles[1].data.ψ0, particles[1].data.ctrl, M)
    set_io!(output, f_ini)
    show(opt, output, obj, alg)
    
    for ei in 1:(max_episode[1]-1)
        for pj in 1:p_num
            M = [C_all[pj][i]*(C_all[pj][i])' for i in 1:M_num]
            obj_copy = set_M(obj, M)
            f_out, f_now = objective(obj_copy, particles[pj])
            if f_now > p_fit[pj]
                p_fit[pj] = f_now
                p_out[pj] = f_out
                for di in 1:dim
                    pbest_state[di,pj] = particles[pj].data.ψ0[di]
                end
                for di in 1:ctrl_num
                    for ni in 1:ctrl_length
                        pbest_ctrl[di,ni,pj] = param_data(particles[pj]).ctrl[di][ni]
                    end
                end
                for di in 1:M_num
                    for ni in 1:dim
                        pbest_meas[di,ni,pj] = C_all[pj][di][ni]
                    end
                end
            end
        end
    
        for pj in 1:p_num
            if p_fit[pj] > fit
                fit = p_fit[pj]
                fit_out = p_out[pj]
                for dj in 1:dim
                    gbest_state[dj] = pbest_state[dj,pj]
                end
                for dj in 1:ctrl_num
                    for nj in 1:ctrl_length
                        gbest_ctrl[dj, nj] = pbest_ctrl[dj,nj,pj]
                    end
                end
                for dj in 1:M_num
                    for nj in 1:dim
                        gbest_meas[dj][nj] = pbest_meas[dj,nj,pj]
                    end
                end
            end
        end  
    
        for pk in 1:p_num
            psi_pre = zeros(ComplexF64, dim)
            for dk in 1:dim
                psi_pre[dk] = particles[pk].data.ψ0[dk]
                velocity_state[dk, pk] = c0*velocity_state[dk, pk] + c1*rand(opt.rng)*(pbest_state[dk, pk] - particles[pk].data.ψ0[dk]) + 
                                         c2*rand(opt.rng)*(gbest_state[dk] - particles[pk].data.ψ0[dk])
                particles[pk].data.ψ0[dk] = particles[pk].data.ψ0[dk] + velocity_state[dk, pk]
            end
            particles[pk].data.ψ0 = particles[pk].data.ψ0/norm(particles[pk].data.ψ0)
            for dm in 1:dim
                velocity_state[dm, pk] = particles[pk].data.ψ0[dm] - psi_pre[dm]
            end
    
            control_coeff_pre = [zeros(ctrl_length) for i in 1:ctrl_num]
            for dk in 1:ctrl_num
                for ck in 1:ctrl_length
                    control_coeff_pre[dk][ck] = param_data(particles[pk]).ctrl[dk][ck]
                    velocity_ctrl[dk, ck, pk] = c0*velocity_ctrl[dk, ck, pk] + c1*rand(opt.rng)*(pbest_ctrl[dk, ck, pk] - param_data(particles[pk]).ctrl[dk][ck]) 
                                         + c2*rand(opt.rng)*(gbest_ctrl[dk, ck] - param_data(particles[pk]).ctrl[dk][ck])
                    param_data(particles[pk]).ctrl[dk][ck] += velocity_ctrl[dk, ck, pk]
                end
            end
    
            for dm in 1:ctrl_num
                for cm in 1:ctrl_length
                    param_data(particles[pk]).ctrl[dm][cm] = (x-> x < opt.ctrl_bound[1] ? opt.ctrl_bound[1] : x > opt.ctrl_bound[2] ? opt.ctrl_bound[2] : x)(param_data(particles[pk]).ctrl[dm][cm])
                    velocity_ctrl[dm, cm, pk] = param_data(particles[pk]).ctrl[dm][cm] - control_coeff_pre[dm][cm]
                end
            end
    
            meas_pre = [zeros(ComplexF64, dim) for i in 1:M_num]
            for dk in 1:M_num
                for ck in 1:dim
                    meas_pre[dk][ck] = C_all[pk][dk][ck]
        
                    velocity_meas[dk, ck, pk] = c0*velocity_meas[dk, ck, pk] + c1*rand(opt.rng)*(pbest_meas[dk, ck, pk] - C_all[pk][dk][ck]) 
                                               + c2*rand(opt.rng)*(gbest_meas[dk][ck] - C_all[pk][dk][ck])
                    C_all[pk][dk][ck] += velocity_meas[dk, ck, pk]
                end
            end
            C_all[pk]= gramschmidt(C_all[pk])
    
            for dm in 1:M_num
                for cm in 1:dim
                    velocity_meas[dm, cm, pk] = C_all[pk][dm][cm] - meas_pre[dm][cm]
                end
            end
        end
        M = [gbest_meas[i]*(gbest_meas[i])' for i in 1:M_num]
        set_f!(output, fit_out)
        set_buffer!(output, gbest_state, gbest_ctrl, M)
        set_io!(output, fit_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end
