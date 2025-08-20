function liouville_commu(H)
    kron(one(H), H) - kron(H |> transpose, one(H))
end

function liouville_dissip(Γ)
    kron(Γ |> conj, Γ) - 0.5 * kron((Γ |> transpose) * (Γ |> conj), Γ |> one) -
    0.5 * kron(Γ |> one, Γ' * Γ)
end

function dissipation(Γ::V, γ::Vector{R}, t::Int = 1) where {V<:AbstractVector,R<:Real}
    [γ[i] * liouville_dissip(Γ[i]) for i in eachindex(Γ)] |> sum
end

function liouvillian(H::Matrix{T}, decay_opt::AbstractVector, γ, t = 1) where {T<:Complex}
    freepart = liouville_commu(H)
    dissp = norm(γ) + 1 ≈ 1 ? freepart |> zero : dissipation(decay_opt, γ, t)
    -1.0im * freepart + dissp
end

function Htot(H0::T, Hc::V, ctrl) where {T<:Matrix{ComplexF64},V<:AbstractVector}
    [H0 + sum([ctrl[i][t] * Hc[i] for i in eachindex(Hc)]) for t in eachindex(ctrl[1])]
end

# function Htot(
#     H0::T,
#     Hc::V,
#     ctrl::Vector{R},
# ) where {T<:AbstractArray,V<:AbstractVector,R<:Real}
#     H0 + ([ctrl[i] * Hc[i] for i in eachindex(ctrl)] |> sum)
# end


# function Htot(H0::V1, Hc::V2, ctrl) where {V1<:AbstractVector,V2<:AbstractVector}
#     H0 + ([ctrl[i] * Hc[i] for i in eachindex(ctrl)] |> sum)
# end

function expL(H, decay_opt, γ, dt, tj = 1)
    Ld = dt * liouvillian(H, decay_opt, γ, tj)
    exp(Ld)
end

function expL(H, dt)
    freepart = liouville_commu(H)
    Ld = -1.0im * dt * freepart
    exp(Ld)
end

##========== matrix exponential (expm) ==========##
@doc raw"""

    expm(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractMatrix, dH::AbstractMatrix; decay::Union{AbstractVector, Nothing}=nothing, Hc::Union{AbstractVector, Nothing}=nothing, ctrl::Union{AbstractVector, Nothing}=nothing)

When applied to the case of single parameter. 
"""
function expm(
    tspan::AbstractVector,
    ρ0::AbstractMatrix,
    H0::AbstractVecOrMat,
    dH::AbstractMatrix;
    decay::Union{AbstractVector,Nothing} = nothing,
    Hc::Union{AbstractVector,Nothing} = nothing,
    ctrl::Union{AbstractVector,Nothing} = nothing,
)
    expm(tspan, ρ0, H0, [dH]; decay = decay, Hc = Hc, ctrl = ctrl)
end

@doc raw"""

    expm(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractMatrix, dH::AbstractVector; decay::Union{AbstractVector, Nothing}=nothing, Hc::Union{AbstractVector, Nothing}=nothing, ctrl::Union{AbstractVector, Nothing}=nothing)

The dynamics of a density matrix is of the form  ``\partial_t\rho=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}\left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right)``, where ``\rho`` is the evolved density matrix, ``H`` is the Hamiltonian of the system, ``\Gamma_i`` and ``\gamma_i`` are the ``i\mathrm{th}`` decay operator and the corresponding decay rate.
- `tspan`: Time length for the evolution.
- `ρ0`: Initial state (density matrix).
- `H0`: Free Hamiltonian.
- `dH`: Derivatives of the free Hamiltonian with respect to the unknown parameters to be estimated. For example, dH[0] is the derivative vector on the first parameter.
- `decay`: Decay operators and the corresponding decay rates. Its input rule is decay=[[``\Gamma_1``, ``\gamma_1``], [``\Gamma_2``, ``\gamma_2``],...], where ``\Gamma_1`` ``(\Gamma_2)`` represents the decay operator and ``\gamma_1`` ``(\gamma_2)`` is the corresponding decay rate.
- `Hc`: Control Hamiltonians.
- `ctrl`: Control coefficients.
"""
function expm(
    tspan::AbstractVector,
    ρ0::AbstractMatrix,
    H0::AbstractVecOrMat,
    dH::AbstractVector;
    decay::Union{AbstractVector,Nothing} = nothing,
    Hc::Union{AbstractVector,Nothing} = nothing,
    ctrl::Union{AbstractVector,Nothing} = nothing,
)
    dim = size(ρ0, 1)
    tnum = length(tspan)
    if isnothing(decay)
        decay_opt = [zeros(ComplexF64, dim, dim)]
        γ = [0.0]
    else
        decay_opt = [decay[1] for decay in decay]
        γ = [decay[2] for decay in decay]
    end

    if isnothing(Hc)
        Hc = [zeros(ComplexF64, dim, dim)]
        ctrl0 = [zeros(tnum - 1)]
    elseif isnothing(ctrl)
        ctrl0 = [zeros(tnum - 1)]
    else
        ctrl_num = length(Hc)
        ctrl_length = length(ctrl)
        if ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences",
                ),
            )
        elseif ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.",
                ),
            )
        end

        ratio_num = ceil((length(tspan) - 1) / length(ctrl[1]))
        if length(tspan) - 1 % length(ctrl[1]) != 0
            tnum = ratio_num * length(ctrl[1]) |> Int
            tspan = range(tspan[1], tspan[end], length = tnum + 1)
        end
        ctrl0 = ctrl
    end
    param_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl0[1])) |> Int
    ## TODO reconstruct repeat feature
    ctrl = [repeat(ctrl0[i], outer = ctrl_interval) for i = 1:ctrl_num]

    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:param_num]

    Δt = tspan[2] - tspan[1]

    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in eachindex(tspan)]
    ∂ρt_∂x_all = [
        [Vector{ComplexF64}(undef, (length(H0))^2) for j = 1:param_num] for
        i = 1:length(tspan)
    ]
    ρt_all[1] = ρ0 |> vec
    for pj = 1:param_num
        ∂ρt_∂x_all[1][pj] = ρt_all[1] |> zero
    end

    for t in eachindex(tspan)[2:end]
        exp_L = expL(H[t-1], decay_opt, γ, Δt, t - 1)
        ρt_all[t] = exp_L * ρt_all[t-1]
        for pj = 1:param_num
            ∂ρt_∂x_all[t][pj] =
                -im * Δt * dH_L[pj] * ρt_all[t] + exp_L * ∂ρt_∂x_all[t-1][pj]
        end
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

function expm(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,NonDecay,NonControl,S,N},M,E},
) where {HT,M,E,S,N}
    (; hamiltonian, tspan) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(hamiltonian)

    return expm(tspan, ρ0, H0, dH)
end

function expm(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,Decay,NonControl,S,N},M,E},
) where {HT,M,E,S,N}
    (; hamiltonian, tspan, decay) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(hamiltonian)

    return expm(tspan, ρ0, H0, dH; decay = decay)
end

function expm(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,NonDecay,Control,S,N},M,E},
) where {HT,M,E,S,N}
    (; hamiltonian, tspan, Hc, ctrl, decay) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(hamiltonian)

    return expm(tspan, ρ0, H0, dH; decay = decay, Hc = Hc, ctrl = ctrl)
end

function expm(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,Decay,Control,S,N},M,E},
) where {HT,M,E,S,N}
    (; hamiltonian, tspan, decay, Hc, ctrl) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(hamiltonian)

    return expm(tspan, ρ0, H0, dH; decay = decay, Hc = Hc, ctrl = ctrl)
end

expm(tspan, ρ0, H0, dH, decay) = expm(tspan, ρ0, H0, dH; decay = decay)
expm(tspan, ρ0, H0, dH, decay, Hc) = expm(tspan, ρ0, H0, dH; decay = decay, Hc = Hc)
expm(tspan, ρ0, H0, dH, decay, Hc, ctrl) =
    expm(tspan, ρ0, H0, dH; decay = decay, Hc = Hc, ctrl = ctrl)

function secondorder_derivative(
    tspan::AbstractVector,
    ρ0::AbstractMatrix,
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    dH_∂x::AbstractVector,
    decay_opt::Vector{Matrix{T}},
    γ,
    Hc::Vector{Matrix{T}},
    ctrl::Vector{Vector{R}},
) where {T<:Complex,R<:Real}

    param_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]

    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
    dH_L = [liouville_commu(dH_∂x[i]) for i = 1:param_num]

    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:param_num]
    ∂2ρt_∂x = [ρt |> zero for i = 1:param_num]
    for t in eachindex(tspan)[2:end]
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        exp_L = expL(H[t-1], decay_opt, γ, Δt, t - 1)
        ρt = exp_L * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
        ∂2ρt_∂x =
            [
                (-im * Δt * dH_L[i] + Δt * Δt * dH_L[i] * dH_L[i]) * ρt -
                2 * im * Δt * dH_L[i] * ∂ρt_∂x[i] for i = 1:param_num
            ] + [exp_L] .* ∂2ρt_∂x
    end
    # ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat, ∂2ρt_∂x |> vec2mat
end

##========== solve ordinary differential equation (ODE) ==========##
@doc raw"""

    ode(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractMatrix, dH::AbstractMatrix; decay::Union{AbstractVector, Nothing}=nothing, Hc::Union{AbstractVector, Nothing}=nothing, ctrl::Union{AbstractVector, Nothing}=nothing)

When applied to the case of single parameter. 
"""
function ode(
    tspan::AbstractVector,
    ρ0::AbstractMatrix,
    H0::AbstractVecOrMat,
    dH::AbstractMatrix;
    decay::Union{AbstractVector,Nothing} = nothing,
    Hc::Union{AbstractVector,Nothing} = nothing,
    ctrl::Union{AbstractVector,Nothing} = nothing,
)
    ode(tspan, ρ0, H0, [dH]; decay = decay, Hc = Hc, ctrl = ctrl)
end

@doc raw"""

    ode(tspan::AbstractVector, ρ0::AbstractMatrix, H0::AbstractVector, dH::AbstractVector; decay::Union{AbstractVector, Nothing}=nothing, Hc::Union{AbstractVector, Nothing}=nothing, ctrl::Union{AbstractVector, Nothing}=nothing)

The dynamics of a density matrix is of the form  ``\partial_t\rho=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}\left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right)``, where ``\rho`` is the evolved density matrix, ``H`` is the Hamiltonian of the system, ``\Gamma_i`` and ``\gamma_i`` are the ``i\mathrm{th}`` decay operator and the corresponding decay rate.
- `tspan`: Time length for the evolution.
- `ρ0`: Initial state (density matrix).
- `H0`: Free Hamiltonian.
- `dH`: Derivatives of the free Hamiltonian with respect to the unknown parameters to be estimated. For example, dH[0] is the derivative vector on the first parameter.
- `decay`: Decay operators and the corresponding decay rates. Its input rule is decay=[[``\Gamma_1``, ``\gamma_1``], [``\Gamma_2``, ``\gamma_2``],...], where ``\Gamma_1`` ``(\Gamma_2)`` represents the decay operator and ``\gamma_1`` ``(\gamma_2)`` is the corresponding decay rate.
- `Hc`: Control Hamiltonians.
- `ctrl`: Control coefficients.
"""
function ode(
    tspan::AbstractVector,
    ρ0::AbstractMatrix,
    H0::AbstractVecOrMat,
    dH::AbstractVector;
    decay::Union{AbstractVector,Nothing} = nothing,
    Hc::Union{AbstractVector,Nothing} = nothing,
    ctrl::Union{AbstractVector,Nothing} = nothing,
)

    ##==========initialization==========##
    dim = size(ρ0, 1)
    ρ0 = complex.(ρ0)
    tnum = length(tspan)
    if isnothing(decay)
        Γ = [zeros(ComplexF64, dim, dim)]
        γ = [0.0]
    else
        Γ = [decay[1] for decay in decay]
        γ = [decay[2] for decay in decay]
    end

    if isnothing(Hc)
        Hc = [zeros(ComplexF64, dim, dim)]
        ctrl0 = [zeros(tnum)]
    elseif isnothing(ctrl)
        ctrl0 = [zeros(tnum)]
    else
        ctrl_num = length(Hc)
        ctrl_length = length(ctrl)
        if ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences",
                ),
            )
        elseif ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.",
                ),
            )
        end

        ratio_num = ceil((length(tspan)) / length(ctrl[1]))
        if length(tspan) % length(ctrl[1]) != 0
            tnum = ratio_num * length(ctrl[1]) |> Int
            tspan = range(tspan[1], tspan[end], length = tnum)
        end
        ctrl0 = ctrl
    end
    param_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = (length(tspan) / length(ctrl0[1])) |> Int
    ctrl = [repeat(ctrl0[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]

    H(ctrl) = Htot(H0, Hc, ctrl)
    dt = tspan[2] - tspan[1]
    t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1
    ρt_func!(ρ, ctrl, t) =
        -im * (H(ctrl)[t2Num(t)] * ρ - ρ * H(ctrl)[t2Num(t)]) + (
            [
                γ[i] * (Γ[i] * ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ρ + ρ * Γ[i]' * Γ[i])) for
                i in eachindex(Γ)
            ] |> sum
        )
    prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]), ctrl)
    ρt = solve(prob_ρ, Tsit5(), saveat = dt).u

    ∂ρt_func!(∂ρ, (pa, ctrl), t) =
        -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) -
        im * (H(ctrl)[t2Num(t)] * ∂ρ - ∂ρ * H(ctrl)[t2Num(t)]) + (
            [
                γ[i] * (Γ[i] * ∂ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ∂ρ + ∂ρ * Γ[i]' * Γ[i]))
                for i in eachindex(Γ)
            ] |> sum
        )

    ∂ρt_tp = []
    for pa = 1:param_num
        prob_∂ρ = ODEProblem(∂ρt_func!, ρ0 |> zero, (tspan[1], tspan[end]), (pa, ctrl))
        push!(∂ρt_tp, solve(prob_∂ρ, Tsit5(), saveat = dt).u)
    end
    ∂ρt = [[∂ρt_tp[i][j] for i = 1:param_num] for j in eachindex(tspan)]
    ρt, ∂ρt
end

# ode(tspan, ρ0, H0, dH, decay) = ode(tspan, ρ0, H0, dH; decay = decay)
# ode(tspan, ρ0, H0, dH, decay, Hc) = ode(tspan, ρ0, H0, dH; decay = decay, Hc = Hc)
# ode(tspan, ρ0, H0, dH, decay, Hc, ctrl) =
#     ode(tspan, ρ0, H0, dH; decay = decay, Hc = Hc, ctrl = ctrl)

function ode(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,NonDecay,NonControl,S,N},M,E},
) where {HT,M,E,S,N}
    (; hamiltonian, tspan) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(hamiltonian)

    return ode(tspan, ρ0, H0, dH)
end

function ode(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,Decay,NonControl,S,N},M,E},
) where {HT,M,E,S,N}
    (; hamiltonian, tspan, decay) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(hamiltonian)

    return ode(tspan, ρ0, H0, dH; decay = decay)
end

function ode(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,NonDecay,Control,S,N},M,E},
) where {HT,M,E,S,N}
    (; hamiltonian, tspan, Hc, ctrl, decay) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(hamiltonian)

    return ode(tspan, ρ0, H0, dH; decay = decay, Hc = Hc, ctrl = ctrl)
end

function ode(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,Decay,Control,S,N},M,E},
) where {HT,M,E,S,N}
    (; hamiltonian, tspan, decay, Hc, ctrl, decay) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(hamiltonian)

    return ode(tspan, ρ0, H0, dH; decay = decay, Hc = Hc, ctrl = ctrl)
end

function evaluate_hamiltonian(
    scheme::Scheme{S,LindbladDynamics{HT,DT,CT,ST,Nothing},TS},
) where {S,HT,DT,CT,ST,TS}
    (; hamiltonian,) = param_data(scheme)
    return evaluate_hamiltonian(hamiltonian)
end

function evaluate_hamiltonian(scheme::Scheme)
    (; hamiltonian,) = param_data(scheme)
    params = hamiltonian.params
    H0 = hamiltonian.H0(params...)
    dH = hamiltonian.dH(params...)
    return H0, dH
end

function evaluate_hamiltonian(ham::Hamiltonian{T,Vector{T},N}) where {T,N}
    H0 = getfield(ham, :H0)
    dH = getfield(ham, :dH)
    return H0, dH
end

function evaluate_hamiltonian(ham::Hamiltonian{F1,F2,N}) where {F1<:Function,F2<:Function,N}
    params = getfield(ham, :params)
    H0 = ham.H0(params...)
    dH = ham.dH(params...)
    return H0, dH
end

#### evolution of pure states under time-independent Hamiltonian without noise and controls ####
function evolve(
    scheme::Scheme{Ket,LindbladDynamics{HT,NonDecay,NonControl,Expm,P},M,E},
) where {HT,M,E,P}
    (; tspan) = param_data(scheme)
    ψ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)
    param_num = length(dH)

    Δt = tspan[2] - tspan[1]
    U = exp(-im * H0 * Δt)
    ψt = ψ0
    ∂ψ∂x = [ψ0 |> zero for i = 1:param_num]
    for i in eachindex(tspan)[2:end]
        ψt = U * ψt
        ∂ψ∂x = [-im * Δt * dH[i] * ψt for i = 1:param_num] + [U] .* ∂ψ∂x
    end
    ρt = ψt * ψt'
    ∂ρt_∂x = [(∂ψ∂x[i] * ψt' + ψt * ∂ψ∂x[i]') for i = 1:param_num]
    ρt, ∂ρt_∂x
end

function evolve(
    scheme::Scheme{Ket,LindbladDynamics{HT,NonDecay,NonControl,Ode,P},M,E},
) where {HT,M,E,P}
    (; tspan) = param_data(scheme)
    ψ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)
    param_num = length(dH)

    dt = tspan[2] - tspan[1]
    ψt_func!(ψ, p, t) = -im * H0 * ψ
    prob_ψ = ODEProblem(ψt_func!, ψ0, (tspan[1], tspan[end]))
    ψt = solve(prob_ψ, Tsit5(), saveat = dt).u

    t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1
    ∂ψt_func!(∂ψ, pa, t) = -im * dH[pa] * ψt[t2Num(t)] - im * H0 * ∂ψ
    ∂ψ∂x = typeof(ψ0)[]
    for pa = 1:param_num
        prob_∂ψ = ODEProblem(∂ψt_func!, ψ0 |> zero, (tspan[1], tspan[end]), pa)
        push!(∂ψ∂x, solve(prob_∂ψ, Tsit5(), saveat = dt).u[end])
    end
    ρt = ψt[end] * ψt[end]'
    ∂ρt_∂x = [(∂ψ∂x[i] * ψt[end]' + ψt[end] * ∂ψ∂x[i]') for i = 1:param_num]
    ρt, ∂ρt_∂x
end

#### evolution of density matrix under time-independent Hamiltonian without noise and controls ####
function evolve(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,NonDecay,NonControl,Expm,P},M,E},
) where {HT,M,E,P}
    (; tspan) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)
    param_num = length(dH)


    Δt = tspan[2] - tspan[1]
    exp_L = expL(H0, Δt)
    dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:param_num]
    for _ = 2:length(tspan)
        ρt = exp_L * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
    end
    # ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end


function evolve(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,NonDecay,NonControl,Ode,P},M,E},
) where {HT,M,E,P}
    (; tspan) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)
    param_num = length(dH)

    dt = tspan[2] - tspan[1]
    t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1

    ρt_func!(ρ, p, t) = -im * (H0 * ρ - ρ * H0)
    prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]))
    ρt = solve(prob_ρ, Tsit5(), saveat = dt).u

    ∂ρt_func!(∂ρ, pa, t) =
        -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) - im * (H0 * ∂ρ - ∂ρ * H0)
    ∂ρt_∂x = typeof(ρ0)[]
    for pa = 1:param_num
        prob_∂ρ = ODEProblem(∂ρt_func!, ρ0 |> zero, (tspan[1], tspan[end]), pa)
        push!(∂ρt_∂x, solve(prob_∂ρ, Tsit5(), saveat = dt).u[end])
    end
    ρt[end], ∂ρt_∂x
end


#### evolution of pure states under time-independent Hamiltonian  
#### with noise but without controls
function evolve(
    scheme::Scheme{Ket,LindbladDynamics{HT,Decay,NonControl,Expm,P},M,E},
) where {HT,M,E,P}
    (; tspan, decay) = param_data(scheme)
    ψ0 = state_data(scheme)
    decay_opt, γ = [d[1] for d in decay], [d[2] for d in decay]
    H0, dH = evaluate_hamiltonian(scheme)
    param_num = length(dH)

    ρt = (ψ0 * ψ0') |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:param_num]
    Δt = tspan[2] - tspan[1]
    exp_L = expL(H0, decay_opt, γ, Δt, 1)
    dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
    for _ = 2:length(tspan)
        ρt = exp_L * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
    end

    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end


function evolve(
    scheme::Scheme{Ket,LindbladDynamics{HT,Decay,NonControl,Ode,P},M,E},
) where {HT,M,E,P}
    (; tspan, decay) = param_data(scheme)
    ψ0 = state_data(scheme)
    decay_opt, γ = [d[1] for d in decay], [d[2] for d in decay]
    H0, dH = evaluate_hamiltonian(scheme)
    param_num = length(dH)

    Γ = decay_opt
    ρ0 = (ψ0 * ψ0')
    dt = tspan[2] - tspan[1]
    t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1
    ρt_func!(ρ, p, t) =
        -im * (H0 * ρ - ρ * H0) + (
            [
                γ[i] * (Γ[i] * ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ρ + ρ * Γ[i]' * Γ[i])) for
                i in eachindex(Γ)
            ] |> sum
        )
    prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]))
    ρt = solve(prob_ρ, Tsit5(), saveat = dt).u

    ∂ρt_func!(∂ρ, pa, t) =
        -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) - im * (H0 * ∂ρ - ∂ρ * H0) + (
            [
                γ[i] * (Γ[i] * ∂ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ∂ρ + ∂ρ * Γ[i]' * Γ[i]))
                for i in eachindex(Γ)
            ] |> sum
        )

    ∂ρt_∂x = typeof(ρ0)[]
    for pa = 1:param_num
        prob_∂ρ = ODEProblem(∂ρt_func!, ρ0 |> zero, (tspan[1], tspan[end]), pa)
        push!(∂ρt_∂x, solve(prob_∂ρ, Tsit5(), saveat = dt).u[end])
    end
    ρt[end], ∂ρt_∂x
end

#### evolution of density matrix under time-independent Hamiltonian  
#### with noise but without controls
function evolve(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,Decay,NonControl,Expm,P},M,E},
) where {HT,M,E,P}
    (; tspan, decay) = param_data(scheme)
    ρ0 = state_data(scheme)
    decay_opt, γ = [d[1] for d in decay], [d[2] for d in decay]
    H0, dH = evaluate_hamiltonian(scheme)
    param_num = length(dH)

    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:param_num]
    Δt = tspan[2] - tspan[1]
    exp_L = expL(H0, decay_opt, γ, Δt, 1)
    dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
    for t in eachindex(tspan)[2:end]
        ρt = exp_L * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function evolve(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,Decay,NonControl,Ode,P},M,E},
) where {HT,M,E,P}
    (; tspan, decay) = param_data(scheme)
    ρ0 = state_data(scheme)
    decay_opt, γ = [d[1] for d in decay], [d[2] for d in decay]
    H0, dH = evaluate_hamiltonian(scheme)
    param_num = length(dH)

    Γ = decay_opt
    dt = tspan[2] - tspan[1]
    t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1
    ρt_func!(ρ, p, t) =
        -im * (H0 * ρ - ρ * H0) + (
            [
                γ[i] * (Γ[i] * ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ρ + ρ * Γ[i]' * Γ[i])) for
                i in eachindex(Γ)
            ] |> sum
        )
    prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]))
    ρt = solve(prob_ρ, Tsit5(), saveat = dt).u

    ∂ρt_func!(∂ρ, pa, t) =
        -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) - im * (H0 * ∂ρ - ∂ρ * H0) + (
            [
                γ[i] * (Γ[i] * ∂ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ∂ρ + ∂ρ * Γ[i]' * Γ[i]))
                for i in eachindex(Γ)
            ] |> sum
        )

    ∂ρt_∂x = typeof(ρ0)[]
    for pa = 1:param_num
        prob_∂ρ = ODEProblem(∂ρt_func!, ρ0 |> zero, (tspan[1], tspan[end]), pa)
        push!(∂ρt_∂x, solve(prob_∂ρ, Tsit5(), saveat = dt).u[end])
    end
    ρt[end], ∂ρt_∂x
end



#### evolution of density matrix under time-independent Hamiltonian 
#### with controls but without noise #### 
function evolve(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,NonDecay,Control,Expm,P},M,E},
) where {HT,M,E,P}
    (; tspan, Hc, ctrl) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)
    H = Matrix{ComplexF64}[]

    param_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl =
        collect.([repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num])
    append!(H, Htot(H0, Hc, ctrl))
    dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:param_num]
    for t in eachindex(tspan)[2:end]
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        exp_L = expL(H[t-1], Δt)
        ρt = exp_L * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
    end
    # ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function evolve(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,NonDecay,Control,Ode,P},M,E},
) where {HT,M,E,P}
    (; tspan, Hc, ctrl) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)

    param_num = length(dH)
    ctrl_num = length(Hc)
    dt = tspan[2] - tspan[1]
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl =
        [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec |> Array for i = 1:ctrl_num]
    push!.(ctrl, [0.0 for i = 1:ctrl_num])
    H(ctrl) = Htot(H0, Hc, ctrl)
    t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1
    ρt_func!(ρ, ctrl, t) = -im * (H(ctrl)[t2Num(t)] * ρ - ρ * H(ctrl)[t2Num(t)])
    prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]), ctrl)
    ρt = solve(prob_ρ, Tsit5(), saveat = dt).u

    ∂ρt_func!(∂ρ, (pa, ctrl), t) =
        -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) -
        im * (H(ctrl)[t2Num(t)] * ∂ρ - ∂ρ * H(ctrl)[t2Num(t)])
    ∂ρt_∂x = typeof(ρ0)[]
    for pa = 1:param_num
        prob_∂ρ = ODEProblem(∂ρt_func!, ρ0 |> zero, (tspan[1], tspan[end]), (pa, ctrl))
        push!(∂ρt_∂x, solve(prob_∂ρ, Tsit5(), saveat = dt).u[end])
    end
    ρt[end], ∂ρt_∂x
end

#### evolution of density matrix under time-independent Hamiltonian with noise and controls #### 
function evolve(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,Decay,Control,Expm,P},M,E},
) where {HT,M,E,P}
    (; tspan, decay, Hc, ctrl) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)
    decay_opt, γ = [d[1] for d in decay], [d[2] for d in decay]

    param_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]
    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for _ = 1:param_num]
    for t in eachindex(tspan)[2:end]
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        exp_L = expL(H[t-1], decay_opt, γ, Δt, t - 1)
        ρt = exp_L * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
    end
    # ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function evolve(
    scheme::Scheme{DensityMatrix,LindbladDynamics{HT,Decay,Control,Ode,P},M,E},
) where {HT,M,E,P}
    (; tspan, decay, Hc, ctrl, abstol, reltol) = param_data(scheme)
    ρ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)
    decay_opt, γ = [d[1] for d in decay], [d[2] for d in decay]

    Γ = decay_opt
    param_num = length(dH)
    dt = tspan[2] - tspan[1]
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl =
        [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec |> Array for i = 1:ctrl_num]
    push!.(ctrl, [0.0 for i = 1:ctrl_num])
    H(ctrl) = Htot(H0, Hc, ctrl)
    t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1

    ρt_func!(ρ, ctrl, t) =
        -im * (H(ctrl)[t2Num(t)] * ρ - ρ * H(ctrl)[t2Num(t)]) + (
            [
                γ[i] * (Γ[i] * ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ρ + ρ * Γ[i]' * Γ[i])) for
                i in eachindex(Γ)
            ] |> sum
        )
    prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]), ctrl)
    ρt = solve(prob_ρ, Tsit5(), saveat = dt; abstol = abstol, reltol = reltol).u

    ∂ρt_func!(∂ρ, (pa, ctrl), t) =
        -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) -
        im * (H(ctrl)[t2Num(t)] * ∂ρ - ∂ρ * H(ctrl)[t2Num(t)]) + (
            [
                γ[i] * (Γ[i] * ∂ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ∂ρ + ∂ρ * Γ[i]' * Γ[i]))
                for i in eachindex(Γ)
            ] |> sum
        )

    ∂ρt_∂x = typeof(ρ0)[]
    for pa = 1:param_num
        prob_∂ρ = ODEProblem(∂ρt_func!, ρ0 |> zero, (tspan[1], tspan[end]), (pa, ctrl))
        push!(
            ∂ρt_∂x,
            solve(prob_∂ρ, Tsit5(), saveat = dt; abstol = abstol, reltol = reltol).u[end],
        )
    end
    ρt[end], ∂ρt_∂x
end

#### evolution of state under time-independent Hamiltonian with noise and controls #### 
function evolve(scheme::Scheme{Ket,LindbladDynamics{HT,Decay,Control,Expm,P},M,E}) where {HT,M,E,P}
    (; tspan, decay, Hc, ctrl) = param_data(scheme)
    ρ0 = state_data(scheme)|>x->x*x'
    H0, dH = evaluate_hamiltonian(scheme)
    decay_opt, γ = [d[1] for d in decay], [d[2] for d in decay]

    param_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]
    H = Htot(H0, Hc, ctrl)
    dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
    ρt = vec(ρ0)
    ∂ρt_∂x = [ρt |> zero for _ = 1:param_num]
    for t in eachindex(tspan)[2:end]
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        exp_L = expL(H[t-1], decay_opt, γ, Δt, t - 1)
        ρt = exp_L * ρt
        ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
    end
    # ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

function evolve(scheme::Scheme{Ket,LindbladDynamics{HT,Decay,Control,Ode,P},M,E}) where {HT,M,E,P}
    (; tspan, decay, Hc, ctrl) = param_data(scheme)
    ψ0 = state_data(scheme)
    H0, dH = evaluate_hamiltonian(scheme)
    decay_opt, γ = [d[1] for d in decay], [d[2] for d in decay]

    ρ0 = (ψ0 * ψ0')
    Γ = decay_opt
    param_num = length(dH)
    dt = tspan[2] - tspan[1]
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl =
        [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec |> Array for i = 1:ctrl_num]
    push!.(ctrl, [0.0 for i = 1:ctrl_num])
    H(ctrl) = Htot(H0, Hc, ctrl)
    t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1
    ρt_func!(ρ, ctrl, t) =
        -im * (H(ctrl)[t2Num(t)] * ρ - ρ * H(ctrl)[t2Num(t)]) + (
            [
                γ[i] * (Γ[i] * ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ρ + ρ * Γ[i]' * Γ[i])) for
                i in eachindex(Γ)
            ] |> sum
        )
    prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]), ctrl)
    ρt = solve(prob_ρ, Tsit5(), saveat = dt).u

    ∂ρt_func!(∂ρ, (pa, ctrl), t) =
        -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) -
        im * (H(ctrl)[t2Num(t)] * ∂ρ - ∂ρ * H(ctrl)[t2Num(t)]) + (
            [
                γ[i] * (Γ[i] * ∂ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ∂ρ + ∂ρ * Γ[i]' * Γ[i]))
                for i in eachindex(Γ)
            ] |> sum
        )

    ∂ρt_∂x = typeof(ρ0)[]
    for pa = 1:param_num
        prob_∂ρ = ODEProblem(∂ρt_func!, ρ0 |> zero, (tspan[1], tspan[end]), (pa, ctrl))
        push!(∂ρt_∂x, solve(prob_∂ρ, Tsit5(), saveat = dt).u[end])
    end
    ρt[end], ∂ρt_∂x
end
# #### evolution of pure states under time-dependent Hamiltonian without noise and controls ####
# function evolve(scheme::Scheme{Ket, LindbladDynamics{Hamiltonian{Vector{Matrix{T}}, Vector{Vector{Matrix{T}}}, N},NonDecay, NonControl, Expm, P}, M, E}) where {T, N, M, E} 
#     (; hamiltonian, tspan) = param_data(scheme)
#     ψ0 = state_data(scheme)
#     H0, dH = evaluate_hamiltonian(hamiltonian)
#     param_num = N
#     dH_L = [liouville_commu.(dh) for dh in dH]
#     ρt = (ψ0 * ψ0') |> vec
#     ∂ρt_∂x = [ρt |> zero for _ in 1:param_num]
#     for t in eachindex(tspan)[2:end]
#         Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
#         exp_L = expL(H0[t-1], Δt)
#         ρt = exp_L * ρt
#         ∂ρt_∂x = [-im * Δt * dhl[t] * ρt for dhl in dH_L] + [exp_L] .* ∂ρt_∂x
#     end
#     # ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
#     ρt |> vec2mat, ∂ρt_∂x |> vec2mat
# end

# function evolve(scheme::Scheme{Ket, LindbladDynamics{Hamiltonian{Vector{Matrix{T}}, Vector{Vector{Matrix{T}}}, N},NonDecay, NonControl, Ode, P}, M, E}) where {T, N, M, E} 
#     (; hamiltonian, tspan) = param_data(scheme)
#     ψ0 = state_data(scheme)
#     H0, dH = evaluate_hamiltonian(hamiltonian)
#     param_num = N
#     dt = tspan[2] - tspan[1] 
#     t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1

#     ψt_func!(ψ, p, t) = -im*H0[t2Num(t)]*ψ
#     prob_ψ = ODEProblem(ψt_func!, ψ0, (tspan[1], tspan[end]))
#     ψt = solve(prob_ψ, Tsit5(), saveat=dt).u

#     ∂ψt_func!(∂ψ, pa, t) = -im * dH[pa] * ψt[t2Num(t)] - im * H0[t2Num(t)] * ∂ψ
#     ∂ψ∂x = typeof(ψ0)[]
#     for pa in 1:param_num
#         prob_∂ψ = ODEProblem(∂ψt_func!, ψ0|>zero, (tspan[1], tspan[end]), pa)
#         push!(∂ψ∂x, solve(prob_∂ψ, Tsit5(), saveat=dt).u[end])
#     end
#     ρt = ψt[end]*ψt[end]'
#     ∂ρt_∂x = [(∂ψ∂x[i] * ψt[end]' + ψt[end] * ∂ψ∂x[i]') for i = 1:param_num]
#     ρt, ∂ρt_∂x 
# end

# #### evolution of density matrix under time-dependent Hamiltonian without noise and controls ####
# function evolve(data::Lindblad_noiseless_timedepend{Expm, P})
#     (; H0, dH, ρ0, tspan) = data

#     param_num = length(dH)
#     dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
#     ρt = ρ0 |> vec
#     ∂ρt_∂x = [ρt |> zero for i = 1:param_num]
#     for t in eachindex(tspan)[2:end]
#         Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
#         exp_L = expL(H0[t-1], Δt)
#         ρt = exp_L * ρt
#         ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
#     end
#     # ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
#     ρt |> vec2mat, ∂ρt_∂x |> vec2mat
# end

# function evolve(data::Lindblad_noiseless_timedepend{Ode, P})
#     (; H0, dH, ρ0, tspan) = data
#     param_num = length(dH)
#     dt = tspan[2] - tspan[1] 
#     t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1

#     ρt_func!(ρ, p, t) = -im*H0[t2Num(t)]*ρ+im*ρ*H0[t2Num(t)]
#     prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]))
#     ρt = solve(prob_ρ, Tsit5(), saveat=dt).u

#     ∂ρt_func!(∂ρ, pa, t) = -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) -im * (H0[t2Num(t)] * ∂ρ - ∂ρ * H0[t2Num(t)])
#     ∂ρt_∂x = typeof(ρ0)[]
#     for pa in 1:param_num
#         prob_∂ρ = ODEProblem(∂ρt_func!, ρ0|>zero, (tspan[1], tspan[end]), pa)
#         push!(∂ρt_∂x, solve(prob_∂ρ, Tsit5(), saveat=dt).u[end])
#     end
#     ρt[end], ∂ρt_∂x
# end

# #### evolution of pure states under time-dependent Hamiltonian  
# #### with noise but without controls
# function evolve(data::Lindblad_noisy_timedepend_pure{Expm, P})
#     (; H0, dH, ψ0, tspan, decay_opt, γ) = data

#     param_num = length(dH)
#     dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
#     ρt = (ψ0 * ψ0') |> vec
#     ∂ρt_∂x = [ρt |> zero for i = 1:param_num]
#     for t in eachindex(tspan)[2:end]
#         Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
#         exp_L = expL(H0[t-1], decay_opt, γ, Δt, t-1)
#         ρt = exp_L * ρt
#         ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
#     end
#     # ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
#     ρt |> vec2mat, ∂ρt_∂x |> vec2mat
# end

# function evolve(data::Lindblad_noisy_timedepend_pure{Ode, P})
#     (; H0, dH, ψ0, tspan, decay_opt, γ) = data
#     Γ = decay_opt
#     param_num = length(dH)
#     ρ0 = (ψ0 * ψ0')
#     dt = tspan[2] - tspan[1] 
#     t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1
#     ρt_func!(ρ, p, t) = -im * (H0[t2Num(t)] * ρ - ρ * H0[t2Num(t)]) + 
#                  ([γ[i] * (Γ[i] * ρ * Γ[i]' - 0.5*(Γ[i]' * Γ[i] * ρ + ρ * Γ[i]' * Γ[i] )) for i in eachindex(Γ)] |> sum)
#     prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]))
#     ρt = solve(prob_ρ, Tsit5(), saveat=dt).u

#     ∂ρt_func!(∂ρ, pa, t) = -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) -im * (H0[t2Num(t)] * ∂ρ - ∂ρ * H0[t2Num(t)]) + 
#                  ([γ[i] * (Γ[i] * ∂ρ * Γ[i]' - 0.5*(Γ[i]' * Γ[i] * ∂ρ + ∂ρ * Γ[i]' * Γ[i] )) for i in eachindex(Γ)] |> sum)

#     ∂ρt_∂x = typeof(ρ0)[]
#     for pa in 1:param_num
#         prob_∂ρ = ODEProblem(∂ρt_func!, ρ0|>zero, (tspan[1], tspan[end]), pa)
#         push!(∂ρt_∂x, solve(prob_∂ρ, Tsit5(), saveat=dt).u[end])
#     end
#     ρt[end], ∂ρt_∂x
# end

# #### evolution of density matrix under time-dependent Hamiltonian  
# #### with noise but without controls
# function evolve(data::Lindblad_noisy_timedepend{Expm, P})
#     (; H0, dH, ρ0, tspan, decay_opt, γ) = data

#     param_num = length(dH)
#     dH_L = [liouville_commu(dH[i]) for i = 1:param_num]
#     ρt = ρ0 |> vec
#     ∂ρt_∂x = [ρt |> zero for i = 1:param_num]
#     for t in eachindex(tspan)[2:end]
#         Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
#         exp_L = expL(H0[t-1], decay_opt, γ, Δt, t-1)
#         ρt = exp_L * ρt
#         ∂ρt_∂x = [-im * Δt * dH_L[i] * ρt for i = 1:param_num] + [exp_L] .* ∂ρt_∂x
#     end
#     # ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
#     ρt |> vec2mat, ∂ρt_∂x |> vec2mat
# end

# function evolve(data::Lindblad_noisy_timedepend{Ode, P})
#     (; H0, dH, ρ0, tspan, decay_opt, γ) = data
#     Γ = decay_opt
#     param_num = length(dH)
#     dt = tspan[2] - tspan[1] 
#     t2Num(t) = Int(round((t - tspan[1]) / dt)) + 1
#     ρt_func!(ρ, p, t) = -im * (H0[t2Num(t)] * ρ - ρ * H0[t2Num(t)]) + 
#                  ([γ[i] * (Γ[i] * ρ * Γ[i]' - 0.5*(Γ[i]' * Γ[i] * ρ + ρ * Γ[i]' * Γ[i] )) for i in eachindex(Γ)] |> sum)
#     prob_ρ = ODEProblem(ρt_func!, ρ0, (tspan[1], tspan[end]))
#     ρt = solve(prob_ρ, Tsit5(), saveat=dt).u

#     ∂ρt_func!(∂ρ, pa, t) = -im * (dH[pa] * ρt[t2Num(t)] - ρt[t2Num(t)] * dH[pa]) -im * (H0[t2Num(t)] * ∂ρ - ∂ρ * H0[t2Num(t)]) + 
#                  ([γ[i] * (Γ[i] * ∂ρ * Γ[i]' - 0.5*(Γ[i]' * Γ[i] * ∂ρ + ∂ρ * Γ[i]' * Γ[i] )) for i in eachindex(Γ)] |> sum)

#     ∂ρt_∂x = typeof(ρ0)[]
#     for pa in 1:param_num
#         prob_∂ρ = ODEProblem(∂ρt_func!, ρ0|>zero, (tspan[1], tspan[end]), pa)
#         push!(∂ρt_∂x, solve(prob_∂ρ, Tsit5(), saveat=dt).u[end])
#     end
#     ρt[end], ∂ρt_∂x
# end

# function propagate(
#     dynamics::LindbladDynamics{noisy,controlled,dm,P},
#     ρₜ::AbstractMatrix,
#     dρₜ::AbstractVector,
#     a::Vector{R},
#     t::Real,
# ) where {R<:Real,P}
#     (; H0, dH, decay_opt, γ, Hc, tspan, ctrl) = dynamics.data
#     Δt = tspan[t] - tspan[t-1]
#     ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
#     param_num = length(dH)
#     H = Htot(H0, Hc, a)
#     dH_L = [liouville_commu(dH) for dH in dH]
#     exp_L = expL(H, decay_opt, γ, Δt)
#     dρₜ_next = [dρₜ|>vec for dρₜ in dρₜ]
#     ρₜ_next = ρₜ
#     for i in 1:ctrl_interval
#         ρₜ_next = exp_L * vec(ρₜ_next)
#         for para = 1:param_num
#             dρₜ_next[para] = -im * Δt * dH_L[para] * ρₜ_next + exp_L * dρₜ_next[para]
#         end
#     end
#     ρₜ_next |> vec2mat, dρₜ_next |> vec2mat
# end

# function propagate(
#     dynamics::LindbladDynamics{noiseless,controlled,dm,P},
#     ρₜ::AbstractMatrix,
#     dρₜ::AbstractVector,
#     a::Vector{R},
#     t::Real,
# ) where {R<:Real,P}
#     (; H0, dH, Hc, tspan, ctrl) = dynamics.data
#     Δt = tspan[t] - tspan[t-1]
#     ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
#     param_num = length(dH)
#     H = Htot(H0, Hc, a)
#     dH_L = [liouville_commu(dH) for dH in dH]
#     exp_L = expL(H, Δt)
#     dρₜ_next = [dρₜ|>vec for dρₜ in dρₜ]
#     ρₜ_next = ρₜ
#     for i in 1:ctrl_interval
#         ρₜ_next = exp_L * vec(ρₜ_next)
#         for para = 1:param_num
#             dρₜ_next[para] = -im * Δt * dH_L[para] * ρₜ_next + exp_L * dρₜ_next[para]
#         end
#     end
#     ρₜ_next |> vec2mat, dρₜ_next |> vec2mat
# end

# function propagate(ρₜ, dρₜ, dynamics, ctrl, t = 1, ctrl_interval = 1)
#     Δt = dynamics.data.tspan[t+1] - system.data.tspan[t]
#     propagate(dynamics, ρₜ, dρₜ, ctrl, Δt, ctrl_interval)
# end
# 
# function propagate!(system)
#     system.ρ, system.∂ρ_∂x = propagate(system.freeHamiltonian, system.dH, system.ρ0,
#                             system.decay_opt, system.γ, system.Hc, 
#                             system.ctrl, system.tspan)
# # end
