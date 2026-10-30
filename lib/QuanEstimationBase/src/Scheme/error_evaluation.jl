"""
    error_evaluation(scheme::Scheme; verbose=true, objective=:QFIM, input_error_scaling=1e-8, SLD_eps=1e-6, abstol=1e-6, reltol=1e-3)

Evaluate the total error scaling for a scheme, combining both parameterization error and SLD truncation-bias error.
raw"""
function error_evaluation(
    scheme::Scheme;
    verbose::Bool = true,
    objective = :QFIM,
    input_error_scaling = 1e-8,
    SLD_eps = 1e-6,
    abstol = 1e-6,
    reltol = 1e-3,
)
    println("Error evaluation for $(nameof(typeof(scheme)))")
    param_error = param_error_evaluation(
        scheme,
        input_error_scaling;
        verbose = verbose,
        objective = objective,
        abstol = abstol,
        reltol = reltol,
    )
    eps_error = SLD_eps_error(scheme, SLD_eps)
    println("\nOverall error scaling ≈ ", param_error + eps_error)
end

raw"""
    param_error_evaluation(scheme::Scheme, input_error_scaling; verbose=true, objective=:QFIM, abstol=1e-6, reltol=1e-3)

Estimate the QFIM error ``\delta F`` due to finite-precision input data (controls, state) via gradient-based propagation. For ODE-based dynamics, returns ``\mathrm{abstol} + \mathrm{reltol} \cdot \mathrm{input\_error\_scaling}``.
raw"""
function param_error_evaluation(
    scheme::Scheme{S,LindbladDynamics{HT,DT,CT,Expm,P},M,E},
    input_error_scaling;
    verbose::Bool = true,
    objective = :QFIM,
    abstol = 1e-6,
    reltol = 1e-3,
) where {S,HT,DT,CT,P,M,E}
    ctrl = scheme.Parameterization.data.ctrl
    rho = scheme.StatePreparation.data
    grad = Zygote.gradient(() -> QFIM(scheme)[1], Zygote.Params([ctrl, rho]))
    gs_c = grad[ctrl]
    gs_s = grad[rho]
    δF =
        sum([norm(g) * input_error_scaling for g in gs_c]) + sum([norm(g) * input_error_scaling for g in gs_s])
    if verbose
        println("\nError evaluation for Lindblad dynamics")
        println(
            "Source: input data (ctrl, Hc, decay etc.) error scaling = $(input_error_scaling)",
        )
        println("δF ≈ ", δF)
    end

    return δF
end

function param_error_evaluation(
    scheme::Scheme{S,LindbladDynamics{HT,DT,CT,Ode,P},M,E},
    input_error_scaling;
    verbose::Bool = true,
    objective = :QFIM,
    abstol = 1e-6,
    reltol = 1e-3,
) where {S,HT,DT,CT,P,M,E}
    δF = abstol + reltol * input_error_scaling
    if verbose
        println("\nError evaluation for Lindblad dynamics")
        println(
            "Source: input data (ctrl, Hc, decay etc.) error scaling = $(input_error_scaling)",
        )
        println("δF ≈ ", δF)
    end
    return δF
end

raw"""
    SLD_eps_error(scheme, eps)

Deprecated.  Return the truncation-bias bound of the QFIM evaluation with the
legacy relative spectral floor `eps`.  The bound is read from the
[`NumericalDiagnostics`](@ref) record of `QFIM(scheme; eps=eps, diagnostics=true)`.
raw"""
function SLD_eps_error(scheme, eps)
    Base.depwarn(
        "`SLD_eps_error` is deprecated; use `QFIM(scheme; eps=..., diagnostics=true)` and " *
        "inspect `diag.truncation_bias_bound` instead.",
        :SLD_eps_error,
    )
    println("\nError evaluation for SLD calculation")
    println("Source: eps = $(eps)")
    _, diag = QFIM(scheme; eps = eps, diagnostics = true)
    δF = diag.truncation_bias_bound
    println("δF ≈ ", δF)
    return δF
end

function _sld_with_error(ρ::Matrix{T}, dρ::Matrix{T}; spec::PrecisionSpec) where {T<:Complex}
    sol = _sld_solve(ρ, dρ; spec = spec)
    return sol.L, sol.L + sol.L_dropped
end

"""
    QFIM_with_error(scheme::Scheme; verbose=false, eps=nothing)

Deprecated shim around the guarded SLD solver.  Return `(F, δF)` where `F` is
assembled from the minimum-norm SLD operators and `δF = F_all - F` is the
change produced by adding the dropped sub-floor blocks back into the SLD
operators.
"""
function QFIM_with_error(scheme::Scheme; verbose::Bool = false, eps = nothing)
    Base.depwarn(
        "`QFIM_with_error` is deprecated; use `QFIM` with the `guard`/`tol` keywords and " *
        "`diagnostics=true` instead.",
        :QFIM_with_error,
    )
    rho, drho = evolve(scheme)
    spec = precision_spec(eps)
    p_num = length(drho)
    sols = [_sld_solve(rho, x; spec = spec) for x in drho]
    Ls = [s.L for s in sols]
    Ls_all = [s.L + s.L_dropped for s in sols]
    F = [
        real(tr(0.5 * rho * (Ls[i] * Ls[j] + Ls[j] * Ls[i]))) for i = 1:p_num, j = 1:p_num
    ]
    F_all = [
        real(tr(0.5 * rho * (Ls_all[i] * Ls_all[j] + Ls_all[j] * Ls_all[i]))) for
        i = 1:p_num, j = 1:p_num
    ]
    δF = F_all - F
    if verbose
        println("\nError evaluation for SLD calculation")
        println("Source: eps = $(eps)")
        println("δF ≈ ", δF[1])
    end
    return F, δF
end

"""
    SLD_with_error(ρ, dρ::Matrix; eps=nothing)

Deprecated shim for the symmetric logarithmic derivative.  Return the tuple
`(L, L_all)` where `L` is the guarded (minimum-norm) SLD and `L_all` adds back
the contribution of the eigenblocks removed by the relative spectral floor.
"""
function SLD_with_error(ρ::Matrix{T}, dρ::Matrix{T}; eps = nothing) where {T<:Complex}
    Base.depwarn(
        "`SLD_with_error` is deprecated; use `SLD(ρ, dρ; guard=..., tol=..., " *
        "diagnostics=true)` instead.",
        :SLD_with_error,
    )
    spec = precision_spec(eps)
    return _sld_with_error(ρ, dρ; spec = spec)
end

function SLD_with_error(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = nothing) where {T<:Complex}
    Base.depwarn(
        "`SLD_with_error` is deprecated; use `SLD(ρ, dρ; guard=..., tol=..., " *
        "diagnostics=true)` instead.",
        :SLD_with_error,
    )
    spec = precision_spec(eps)
    results = [_sld_with_error(ρ, x; spec = spec) for x in dρ]
    return [r[1] for r in results], [r[2] for r in results]
end

"""
    QFIM_SLD_with_error(ρ, dρ::Matrix; eps=nothing)

Deprecated.  Forward to [`QFIM_SLD`](@ref).
"""
function QFIM_SLD_with_error(ρ::Matrix{T}, dρ::Matrix{T}; eps = nothing) where {T<:Complex}
    Base.depwarn(
        "`QFIM_SLD_with_error` is deprecated; use `QFIM_SLD(ρ, dρ; guard=..., tol=...)` " *
        "instead.",
        :QFIM_SLD_with_error,
    )
    return QuanEstimationBase.QFIM_SLD(ρ, dρ; eps = eps)
end

"""
    QFIM_RLD_with_error(ρ, dρ::Matrix; eps=nothing)

Deprecated.  Forward to [`QFIM_RLD`](@ref).
"""
function QFIM_RLD_with_error(ρ::Matrix{T}, dρ::Matrix{T}; eps = nothing) where {T<:Complex}
    Base.depwarn(
        "`QFIM_RLD_with_error` is deprecated; use `QFIM_RLD(ρ, dρ; guard=..., tol=...)` " *
        "instead.",
        :QFIM_RLD_with_error,
    )
    return QuanEstimationBase.QFIM_RLD(ρ, dρ; eps = eps)
end

"""
    QFIM_LLD_with_error(ρ, dρ::Matrix; eps=nothing)

Deprecated.  Forward to [`QFIM_LLD`](@ref).
"""
function QFIM_LLD_with_error(ρ::Matrix{T}, dρ::Matrix{T}; eps = nothing) where {T<:Complex}
    Base.depwarn(
        "`QFIM_LLD_with_error` is deprecated; use `QFIM_LLD(ρ, dρ; guard=..., tol=...)` " *
        "instead.",
        :QFIM_LLD_with_error,
    )
    return QuanEstimationBase.QFIM_LLD(ρ, dρ; eps = eps)
end

raw"""
    QFIM_pure_with_error(ρ, ∂ρ_∂x)

Deprecated.  Forward to [`QFIM_pure`](@ref).
"""
function QFIM_pure_with_error(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    Base.depwarn(
        "`QFIM_pure_with_error` is deprecated; use `QFIM_pure(ρ, ∂ρ_∂x)` instead.",
        :QFIM_pure_with_error,
    )
    return QuanEstimationBase.QFIM_pure(ρ, ∂ρ_∂x)
end

"""
    QFIM_SLD_with_error(ρ, dρ::Vector{Matrix}; eps=nothing)

Deprecated.  Forward to [`QFIM_SLD`](@ref).
"""
function QFIM_SLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = nothing,
) where {T<:Complex}
    Base.depwarn(
        "`QFIM_SLD_with_error` is deprecated; use `QFIM_SLD(ρ, dρ; guard=..., tol=...)` " *
        "instead.",
        :QFIM_SLD_with_error,
    )
    return QuanEstimationBase.QFIM_SLD(ρ, dρ; eps = eps)
end

"""
    QFIM_RLD_with_error(ρ, dρ::Vector{Matrix}; eps=nothing)

Deprecated.  Forward to [`QFIM_RLD`](@ref).
"""
function QFIM_RLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = nothing,
) where {T<:Complex}
    Base.depwarn(
        "`QFIM_RLD_with_error` is deprecated; use `QFIM_RLD(ρ, dρ; guard=..., tol=...)` " *
        "instead.",
        :QFIM_RLD_with_error,
    )
    return QuanEstimationBase.QFIM_RLD(ρ, dρ; eps = eps)
end

"""
    QFIM_LLD_with_error(ρ, dρ::Vector{Matrix}; eps=nothing)

Deprecated.  Forward to [`QFIM_LLD`](@ref).
"""
function QFIM_LLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = nothing,
) where {T<:Complex}
    Base.depwarn(
        "`QFIM_LLD_with_error` is deprecated; use `QFIM_LLD(ρ, dρ; guard=..., tol=...)` " *
        "instead.",
        :QFIM_LLD_with_error,
    )
    return QuanEstimationBase.QFIM_LLD(ρ, dρ; eps = eps)
end

"""
    QFIM_pure_with_error(ρ, ∂ρ_∂x::Vector{Matrix})

Deprecated.  Forward to [`QFIM_pure`](@ref).
"""
function QFIM_pure_with_error(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    Base.depwarn(
        "`QFIM_pure_with_error` is deprecated; use `QFIM_pure(ρ, ∂ρ_∂x)` instead.",
        :QFIM_pure_with_error,
    )
    return QuanEstimationBase.QFIM_pure(ρ, ∂ρ_∂x)
end
