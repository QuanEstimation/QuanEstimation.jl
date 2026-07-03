using Setfield

raw"""

    error_control_param(scheme::Scheme; output_error_scaling=1e-6, input_error_scaling=1e-8, max_episode=10)

Refine the time discretization of a Lindblad dynamics parameterization until the parameterization error ``\delta F`` falls below `output_error_scaling`. Doubles the number of time steps each episode.
"""
function error_control_param(
    scheme::Scheme{S,LindbladDynamics{HT,DT,CT,Expm,P},M,E};
    output_error_scaling = 1e-6,
    input_error_scaling = 1e-8,
    max_episode = 10,
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


raw"""

    error_control_eps(scheme::Scheme; SLD_eps=1e-8, max_episode=10)

Iteratively reduce the SLD eigenvalue threshold `SLD_eps` until the resulting QFIM error ``\delta F`` stabilizes.
"""
function error_control_eps(scheme::Scheme; SLD_eps = 1e-8, max_episode = 10)
    eps_tp = SLD_eps
    eps_error = SLD_eps_error(scheme, SLD_eps)
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
        eps_error = SLD_eps_error(scheme, eps_tp)
        println("eps=", eps_tp, " δF ≈ ", eps_error)
    end
    return nothing
end

"""

    error_control(scheme::Scheme; objective="QFIM", output_error_scaling=1e-6, input_error_scaling=1e-8, SLD_eps=1e-6, max_episode=10)

Run both parameterization and SLD-epsilon error control for the given scheme. Currently supports `objective="QFIM"`.
"""
function error_control(
    scheme::Scheme;
    objective::Union{Symbol, String}="QFIM",
    output_error_scaling = 1e-6,
    input_error_scaling = 1e-8,
    SLD_eps = 1e-6,
    max_episode = 10,
)
    if objective == "QFIM" || objective == :QFIM
        if objective isa String
            objective = Symbol(objective)
        end

        error_control_param(
            scheme;
            output_error_scaling = output_error_scaling,
            input_error_scaling = input_error_scaling,
            max_episode = max_episode,
        )
        error_control_eps(scheme; SLD_eps = SLD_eps, max_episode = max_episode)
    else
        throw(ArgumentError("Objective not supported."))
    end
end
