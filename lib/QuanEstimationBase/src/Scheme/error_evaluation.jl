"""

    error_evaluation(scheme::Scheme; verbose=true, objective=:QFIM, input_error_scaling=1e-8, SLD_eps=1e-6, abstol=1e-6, reltol=1e-3)

Evaluate the total error scaling for a scheme, combining both parameterization error and SLD eigenvalue threshold error.
raw"""
function error_evaluation(
    scheme::Scheme;
    verbose::Bool = true,
    objective=:QFIM,
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
    objective=:QFIM,
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
    objective=:QFIM,
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

Compute the QFIM error ``\delta F`` induced by the SLD eigenvalue threshold `eps`.
"""
function SLD_eps_error(scheme, eps)
    println("\nError evaluation for SLD calculation")
    println("Source: eps = $(eps)")
    _, δF = QFIM_with_error(scheme, eps = eps)
    println("δF ≈ ", δF[1])
    return δF[1]
end  # function SLD_eps_error

"""

    QFIM_with_error(scheme::Scheme; verbose=false, eps=GLOBAL_EPS)

Compute the QFIM and its error due to SLD truncation.
"""
function QFIM_with_error(scheme::Scheme; verbose::Bool = false, eps = GLOBAL_EPS)
    rho, drho = evolve(scheme)

    p_num = length(drho)
    SLD_tp, SLD_tp_err = SLD_with_error(rho, drho; eps = eps)
    F =
        (
            [0.5 * rho] .* (
                kron(SLD_tp, reshape(SLD_tp, 1, p_num)) +
                kron(reshape(SLD_tp, 1, p_num), SLD_tp)
            )
        ) .|>
        tr .|>
        real
    δF =
        real.(
            tr.(
                [0.5 * rho] .* (
                    kron(SLD_tp_err, reshape(SLD_tp_err, 1, p_num)) +
                    kron(reshape(SLD_tp_err, 1, p_num), SLD_tp_err)
                )
            )
        ) - F
    return F, δF
end

"""

    SLD_with_error(ρ, dρ; eps=GLOBAL_EPS)

Compute the symmetric logarithmic derivative (SLD) together with its error contribution from eigenvalues below the threshold `eps`.
"""
function SLD_with_error(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}

    dim = size(ρ)[1]

    ρ_h = (ρ + ρ') / 2
    val, vec = eigen(ρ_h)
    val = val |> real
    SLD_eig = zeros(T, dim, dim)
    SLD_eig_err = zeros(T, dim, dim)
    for fi = 1:dim
        for fj = 1:dim
            if val[fi] + val[fj] > eps
                SLD_eig[fi, fj] = 2 * (vec[:, fi]' * dρ * vec[:, fj]) / (val[fi] + val[fj])
            else
                SLD_eig_err[fi, fj] =
                    2 * (vec[:, fi]' * dρ * vec[:, fj]) / (val[fi] + val[fj])
            end
        end
    end
    SLD_eig[findall(SLD_eig == Inf)] .= 0.0
    SLD_eig[findall(abs.(SLD_eig) .> 1e10)] .= 0.0

    return vec * (SLD_eig * vec'), vec * ((SLD_eig + SLD_eig_err) * vec')
end

function SLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    temp = [SLD_with_error(ρ, x; eps = eps) for x in dρ]
    return [t[1] for t in temp], [t[2] for t in temp]
end



"""

    QFIM_SLD_with_error(ρ, dρ; eps=GLOBAL_EPS)

Compute the QFIM element via the SLD approach, with error estimation.
"""
function QFIM_SLD_with_error(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    SLD_tp = SLD(ρ, dρ; eps = eps)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

"""

    QFIM_RLD_with_error(ρ, dρ; eps=GLOBAL_EPS)

Compute the QFIM element via the right logarithmic derivative (RLD), with error estimation.
"""
function QFIM_RLD_with_error(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    R = RLD(ρ, dρ; eps = eps)
    F = tr(ρ * R * R')
    F |> real
end

"""

    QFIM_LLD_with_error(ρ, dρ; eps=GLOBAL_EPS)

Compute the QFIM element via the left logarithmic derivative (LLD), with error estimation.
raw"""
function QFIM_LLD_with_error(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    L = LLD(ρ, dρ; eps = eps)
    F = tr(ρ * L' * L)
    F |> real
end

raw"""

    QFIM_pure_with_error(ρ, ∂ρ_∂x)

Compute the QFIM element for a pure state ``\rho = |\psi\rangle\langle\psi|``. For pure states, ``L = 2\partial\rho/\partial x``.
"""
function QFIM_pure_with_error(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    SLD = 2 * ∂ρ_∂x
    SLD2_tp = SLD * SLD
    F = tr(ρ * SLD2_tp)
    F |> real
end

#==========================================================#
####################### calculate QFIM #####################
"""
    QFIM_SLD_with_error(ρ, dρ::Vector{Matrix}; eps=GLOBAL_EPS)

Multi-parameter QFIM via SLD with error estimation.

See also: [`QFIM_SLD_with_error(ρ, dρ::Matrix; eps)`](@ref) for the single-parameter version.
"""
function QFIM_SLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = (x -> SLD(ρ, x; eps = eps)).(dρ)
    return [real(tr(0.5 * ρ * (LD_tp[i] * LD_tp[j] + LD_tp[j] * LD_tp[i]))) for i in 1:p_num, j in 1:p_num]
end

"""
    QFIM_RLD_with_error(ρ, dρ::Vector{Matrix}; eps=GLOBAL_EPS)

Multi-parameter QFIM via RLD with error estimation.

See also: [`QFIM_RLD_with_error(ρ, dρ::Matrix; eps)`](@ref) for the single-parameter version.
"""
function QFIM_RLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    p_num = length(dρ)
    R = RLD(ρ, dρ; eps = eps)
    return [tr(ρ * R[i] * R[j]') for i in 1:p_num, j in 1:p_num]
end

"""
    QFIM_LLD_with_error(ρ, dρ::Vector{Matrix}; eps=GLOBAL_EPS)

Multi-parameter QFIM via LLD with error estimation.

See also: [`QFIM_LLD_with_error(ρ, dρ::Matrix; eps)`](@ref) for the single-parameter version.
"""
function QFIM_LLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    p_num = length(dρ)
    L = LLD(ρ, dρ; eps = eps)
    return [tr(ρ * L[i]' * L[j]) for i in 1:p_num, j in 1:p_num]
end

"""
    QFIM_pure_with_error(ρ, ∂ρ_∂x::Vector{Matrix})

Multi-parameter QFIM for pure states with error estimation.

See also: [`QFIM_pure_with_error(ρ, ∂ρ_∂x::Matrix)`](@ref) for the single-parameter version.
"""
function QFIM_pure_with_error(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    p_num = length(∂ρ_∂x)
    sld = [2 * ∂ρ_∂x[i] for i = 1:p_num]
    return [real(tr(0.5 * ρ * (sld[i] * sld[j] + sld[j] * sld[i]))) for i in 1:p_num, j in 1:p_num]
end
