using Convex
function decomposition(A)
    C = bunchkaufman(A; check = false)
    R = sqrt(Array(C.D)) * C.U'C.P
    return R
end

function HCRB(scheme::AbstractScheme; C = nothing, eps = GLOBAL_EPS)
    if isnothing(C)
        C = I(get_param_num(scheme))
    end
    rho, drho = evolve(scheme)
    return HCRB(rho, drho, C; eps = eps)
end

"""

    HCRB(ρ::AbstractMatrix, dρ::AbstractVector, C::AbstractMatrix; eps=GLOBAL_EPS)

Caltulate the Holevo Cramer-Rao bound (HCRB) via the semidefinite program (SDP).
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix on the unknown parameters to be estimated. For example, drho[0] is the derivative vector on the first parameter.
- `W`: Weight matrix.
- `eps`: Machine epsilon.
"""
function HCRB(ρ::AbstractMatrix, dρ::AbstractVector, C::AbstractMatrix; eps = GLOBAL_EPS)
    if length(dρ) == 1
        println(
            "In the single-parameter scenario, the HCRB is equivalent to the QFI. This function will return the value of the QFI.",
        )
        f = QFIM_SLD(ρ, dρ[1]; eps = eps)
        return f
    elseif rank(C) == 1
        println(
            "For rank-one wight matrix, the HCRB is equivalent to QFIM. This function will return the value of Tr(WF^{-1}).",
        )
        F = QFIM_SLD(ρ, dρ; eps = eps)
        return tr(C * pinv(F))
    else
        Holevo_bound(ρ, dρ, C; eps = eps)
    end
end

function Holevo_bound(
    ρ::AbstractMatrix,
    dρ::AbstractVector,
    C::AbstractMatrix;
    eps = GLOBAL_EPS,
)

    dim = size(ρ)[1]
    num = dim * dim
    para_num = length(dρ)
    suN = suN_generator(dim) / sqrt(2)
    Lambda = [Matrix{ComplexF64}(I, dim, dim) / sqrt(2)]
    append!(Lambda, [suN[i] for i in eachindex(suN)])
    vec_∂ρ = [[0.0 for i = 1:num] for j = 1:para_num]

    for pa = 1:para_num
        for ra = 2:num
            vec_∂ρ[pa][ra] = (dρ[pa] * Lambda[ra]) |> tr |> real
        end
    end
    S = zeros(ComplexF64, num, num)
    for a = 1:num
        for b = 1:num
            S[a, b] = (Lambda[a] * Lambda[b] * ρ) |> tr
        end
    end

    accu = length(string(Int(ceil(1 / eps)))) - 1
    R = decomposition(round.(digits = accu, S))

    #=========optimization variables===========#
    V = Variable(para_num, para_num)
    X = Variable(num, para_num)
    #============add constraints===============#
    constraints = [[V X'*R'; R*X Matrix{Float64}(I, num, num)] ⪰ 0]
    for i = 1:para_num
        for j = 1:para_num
            if i == j
                constraints += [X[:, i]' * vec_∂ρ[j] == 1]
            else
                constraints += [X[:, i]' * vec_∂ρ[j] == 0]
            end
        end
    end
    problem = minimize(tr(C * V), constraints)
    Convex.solve!(problem, SCS.Optimizer, silent_solver = true)
    return evaluate(tr(C * V))
end

function Holevo_bound_obj(
    ρ::AbstractMatrix,
    dρ::AbstractVector,
    C::AbstractMatrix;
    eps = GLOBAL_EPS,
)
    return Holevo_bound(ρ, dρ, C; eps = eps)[1]
end

"""

    NHB(ρ::AbstractMatrix, dρ::AbstractVector, W::AbstractMatrix)

Nagaoka-Hayashi bound (NHB) via the semidefinite program (SDP).
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix on the unknown parameters to be estimated. For example, drho[0] is the derivative vector on the first parameter.
- `W`: Weight matrix.
"""
function NHB(ρ::AbstractMatrix, dρ::AbstractVector, W::AbstractMatrix)

    dim = size(ρ)[1]
    para_num = length(dρ)

    #=========optimization variables===========#
    L_tp = [[Variable() for i = 1:para_num] for j = 1:para_num]
    for para_i = 1:para_num
        for para_j = para_i:para_num
            L_tp[para_i][para_j] = ComplexVariable(dim, dim)
            constraints = [transpose(conj(L_tp[para_i][para_j])) == L_tp[para_i][para_j]]
            L_tp[para_j][para_i] = L_tp[para_i][para_j]
        end
    end
    L = vcat([hcat(L_tp[i]...) for i = 1:para_num]...)
    X = [ComplexVariable(dim, dim) for j = 1:para_num]

    #============add constraints===============#
    constraints += [[L vcat(X...); hcat(X...) Matrix{Float64}(I, dim, dim)] ⪰ 0]
    for i = 1:para_num
        constraints += [tr(X[i] * ρ[i]) == 0]
        constraints += [transpose(conj(X[i])) == X[i]]
        for j = 1:para_num
            if i == j
                constraints += [tr(X[i] * dρ[j]) == 1]
            else
                constraints += [tr(X[i] * dρ[j]) == 0]
            end
        end
    end
    problem = minimize(real(tr(kron(W, ρ) * L)), constraints)
    Convex.solve!(problem, SCS.Optimizer, silent_solver = true)
    return evaluate(real(tr(kron(W, ρ) * L)))
end
