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

Deprecated.  Tighten the relative spectral guard starting from `SLD_eps` by a
factor of ten per episode until the truncation-bias bound reported by
[`NumericalDiagnostics`](@ref) stabilizes, or until the default guard
`default_guard(Float64)` is reached.
"""
function error_control_eps(scheme::Scheme; SLD_eps = 1e-8, max_episode = 10)
    Base.depwarn(
        "`error_control_eps` is deprecated; pass `guard`/`tol` to `QFIM` and inspect " *
        "`diagnostics=true` instead.",
        :error_control_eps,
    )
    rho, drho = evolve(scheme)
    floor_min = default_guard(Float64)
    g = max(Float64(SLD_eps), floor_min)
    bias = Inf
    for _ = 1:max_episode
        _, diag = QFIM(rho, drho; guard = g, diagnostics = true)
        bias_new = diag.truncation_bias_bound
        println("guard=", g, " δF ≈ ", bias_new)
        if isfinite(bias) &&
           abs(bias_new - bias) <= 1e-3 * max(abs(bias), floatmin(Float64))
            println("Truncation bias stabilized; error control is successful.")
            return nothing
        end
        bias = bias_new
        g <= floor_min && break
        g = max(g / 10, floor_min)
    end
    return nothing
end

"""
    error_control(scheme::Scheme; objective="QFIM", output_error_scaling=1e-6, input_error_scaling=1e-8, SLD_eps=1e-6, max_episode=10)

Run both parameterization and SLD-guard error control for the given scheme. Currently supports `objective="QFIM"`.
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
