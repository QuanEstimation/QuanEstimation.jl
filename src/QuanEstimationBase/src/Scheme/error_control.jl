using Setfield

# function error_control_param(scheme::Scheme{S, Lindblad{HT, DT, CT, Expm, P}, M, E};output_error_scaling = 1e-6, input_error_scaling=1e-8, max_episode=1000) where {S, HT, DT, CT, Expm, P, M, E}
#     (;Parameterization) = scheme
#     (;data) = Parameterization


# end

function error_control_param(
    scheme::Scheme{S,Lindblad{HT,DT,CT,Expm,P},M,E};
    output_error_scaling = 1e-6,
    input_error_scaling = 1e-8,
    max_episode = 1000,
) where {S,HT,DT,CT,Expm,P,M,E}
    (; Parameterization) = scheme
    (; data) = Parameterization

    for _ = 1:max_episode
        tspan = data.tspan
        t0, t1, te = tspan[1], tspan[2], tspan[end]
        data = @set data.tspan = t0:(t1-t0)/2:te
        scheme = @set scheme.Parameterization = typeof(Parameterization)(data, nothing)
        δF = param_error_evaluation(scheme, input_error_scaling; verbose = false)
        @show δF[1]
        if δF[1] < output_error_scaling
            println("Parameterization error control is successful.")
            println("Current δF ≈ ", δF[1])
            break
        end
    end
end


function error_control_eps(scheme::Scheme; eps_scaling = 1e-8, max_episode = 100)
    eps_tp = eps_scaling
    eps_error = SLD_eps_error(scheme, eps_scaling)
    println("δF ≈ ", eps_error)
    if eps_error ≈ 0
        println("No need for eps error control")
        return nothing
    end

    for _ = 1:max_episode
        eps_tp = eps_tp / 10
        if eps_error <= 1e-16
            break
        end
        eps_error = SLD_eps_error(scheme, eps_scaling)
        println("eps=", eps_tp, " δF ≈ ", eps_error)
    end
    return nothing
end

function error_control(
    scheme::Scheme;
    output_error_scaling = 1e-6,
    input_error_scaling = 1e-8,
    eps_scaling = 1e-6,
    max_episode = 1000,
)
    error_control_param(
        scheme,
        output_error_scaling = output_error_scaling,
        input_error_scaling = input_error_scaling,
        max_episode = max_episode,
    )
    error_control_eps(scheme, eps_scaling = eps_scaling, max_episode = max_episode)
end
