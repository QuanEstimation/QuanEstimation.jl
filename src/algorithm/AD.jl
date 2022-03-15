#### AD ####
function update!(opt::ControlOpt, alg::AD_Adam, obj, dynamics, output)
    (; max_episode, ϵ, beta1, beta2) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    
    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i in 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)
    
    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt, output,obj)
    
    output.f_list = [f_ini]
    for ei in 1:(max_episode-1)
        # δI = grad(obj, dynamics)
        δI = gradient(()->objective(obj, dynamics)[2], Flux.Params([dynamics.data.ctrl]))
        Adam_ctrl!(dynamics, δI[dynamics.data.ctrl], ϵ, beta1, beta2, obj.eps)
        bound!(dynamics.data.ctrl, opt.ctrl_bound)
        f_out, f_now = objective(obj, dynamics)

        set_f!(output, f_out)
        set_buffer!(output, [dynamics.data.ctrl])
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function update!(opt::ControlOpt, alg::AD, obj, dynamics, output)
    (; max_episode, ϵ) = alg
    ctrl_length = length(dynamics.data.ctrl[1])
    ctrl_num = length(dynamics.data.Hc)
    
    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i in 1:ctrl_num])
    f_noctrl, f_comp = objective(obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)

    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl, f_ini)
    show(opt,output,obj)

    output.f_list = [f_ini]
    for ei in 1:(max_episode-1)
        δI = gradient(()->objective(obj, dynamics)[2], Flux.Params([dynamics.data.ctrl]))
        dynamics.data.ctrl += ϵ*δI[dynamics.data.ctrl]
        bound!(dynamics.data.ctrl, opt.ctrl_bound)
        f_out, f_now = objective(obj, dynamics)
        
        set_f!(output, f_out)
        set_buffer!(output, [dynamics.data.ctrl])
        set_io!(output, f_out, ei)
        show(output, obj)
    end
    set_io!(output, output.f_list[end])
end

function grad(opt::ControlOpt,obj, dynamics)
    (;H0, dH, ρ0, tspan, decay_opt, γ, Hc, ctrl)= dynamics.data
    δI = gradient(x->objective_grad(obj, H0, dH, ρ0, tspan, decay_opt, γ, Hc, x),ctrl).|>real |>sum
end
