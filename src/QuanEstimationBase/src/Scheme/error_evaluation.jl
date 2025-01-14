function error_evaluation(
    scheme::Scheme;
    verbose::Bool = true,
    input_error_scaling = 1e-8,
    SLD_eps = 1e-6,
    abstol = 1e-6,
    reltol = 1e-3,
)
    println("Error evaluation for $(typeof(scheme))")
    param_error = param_error_evaluation(
        scheme,
        input_error_scaling;
        abstol = abstol,
        reltol = reltol,
        verbose = verbose,
    )
    eps_error = SLD_eps_error(scheme, SLD_eps)
    println("\nOverall error scaling ≈ ", param_error + eps_error)
end

function param_error_evaluation(
    scheme::Scheme{S,Lindblad{HT,DT,CT,Expm,P},M,E},
    input_error_scaling;
    verbose::Bool = true,
    abstol = 1e-6,
    reltol = 1e-3,
) where {S,HT,DT,CT,P,M,E}
    ctrl = scheme.Parameterization.data.ctrl
    rho = scheme.StatePreparation.data
    grad = Flux.gradient(() -> QFIM(scheme)[1], Flux.Params([ctrl, rho]))
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
    scheme::Scheme{S,Lindblad{HT,DT,CT,Ode,P},M,E},
    input_error_scaling;
    verbose::Bool = true,
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

function SLD_eps_error(scheme, eps)
    println("\nError evaluation for SLD calculation")
    println("Source: eps = $(eps)")
    F, δF = QFIM_with_error(scheme, eps = eps)
    println("δF ≈ ", δF[1])
    return δF[1]
end  # function SLD_eps_error

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

function SLD_with_error(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}

    dim = size(ρ)[1]

    val, vec = eigen(ρ)
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

function QFIM_RLD_with_error(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    RLD_tp = pinv(ρ, reltol = eps) * dρ
    F = tr(ρ * RLD_tp * RLD_tp')
    F |> real
end

function QFIM_LLD_with_error(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    LLD_tp = (dρ * pinv(ρ, reltol = eps))'
    F = tr(ρ * LLD_tp * LLD_tp')
    F |> real
end

function QFIM_pure_with_error(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    SLD = 2 * ∂ρ_∂x
    SLD2_tp = SLD * SLD
    F = tr(ρ * SLD2_tp)
    F |> real
end

#==========================================================#
####################### calculate QFIM #####################
function QFIM_SLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = (x -> SLD(ρ, x; eps = eps)).(dρ)
    (
        [0.5 * ρ] .*
        (kron(LD_tp, reshape(LD_tp, 1, p_num)) + kron(reshape(LD_tp, 1, p_num), LD_tp))
    ) .|>
    tr .|>
    real
end

function QFIM_RLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = (x -> (pinv(ρ, reltol = eps) * x)).(dρ)
    LD_dag = [LD_tp[i]' for i = 1:p_num]
    ([ρ] .* (kron(LD_tp, reshape(LD_dag, 1, p_num)))) .|> tr
end

function QFIM_LLD_with_error(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = (x -> (x * pinv(ρ, reltol = eps))').(dρ)
    LD_dag = [LD_tp[i]' for i = 1:p_num]
    ([ρ] .* (kron(LD_tp, reshape(LD_dag, 1, p_num)))) .|> tr
end

function QFIM_liouville_with_error(ρ, dρ)
    p_num = length(dρ)
    LD_tp = SLD_lio
    uville(ρ, dρ)
    (
        [0.5 * ρ] .*
        (kron(LD_tp, reshape(LD_tp, 1, p_num)) + kron(reshape(LD_tp, 1, p_num), LD_tp))
    ) .|>
    tr .|>
    real
end

function QFIM_pure_with_error(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    p_num = length(∂ρ_∂x)
    sld = [2 * ∂ρ_∂x[i] for i = 1:p_num]
    (
        [0.5 * ρ] .*
        (kron(sld, reshape(sld, 1, p_num)) + kron(reshape(sld, 1, p_num), sld))
    ) .|>
    tr .|>
    real
end
