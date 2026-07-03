using Convex

@doc raw"""
    decomposition(A)

Perform the LDLᵀ decomposition (Bunch-Kaufman) of a symmetric matrix ``A``
and return the upper-triangular factor ``R`` such that ``R^{\mathsf{T}}R \approx A``.

This is used internally by [`Holevo_bound`](@ref) to factorize the correlation
matrix ``S`` in the SDP formulation of the HCRB.

# Arguments

- `A`: Symmetric matrix to decompose.

# Returns

- `Matrix`: The factor ``R`` satisfying ``R^{\mathsf{T}}R \approx A``.

# See Also

- [`Holevo_bound`](@ref): HCRB via SDP using this decomposition.
raw"""
function decomposition(A)
    C = bunchkaufman(A; check = false)
    R = sqrt(Array(C.D)) * C.U'C.P
    return R
end

@doc raw"""
    HCRB(scheme::AbstractScheme; W=nothing, eps=GLOBAL_EPS)

Compute the Holevo Cramér-Rao bound (HCRB) from a Scheme.

Evolves the state encoded in the scheme and calls [`HCRB`](@ref) on the
resulting density matrix and derivatives.

# Arguments

- `scheme::AbstractScheme`: The estimation scheme.
- `W::Union{AbstractMatrix,UniformScaling,Nothing}=nothing`: Weight matrix.
  Defaults to the identity ``\mathbb{I}``.
- `eps::Float64=GLOBAL_EPS`: Epsilon threshold.

# Returns

- `Float64`: The HCRB value ``\mathrm{Tr}(W V)``.

# See Also

- [`HCRB`](@ref): Direct HCRB computation from density matrix.
- [`NHB`](@ref): Nagaoka-Hayashi bound (alternative bound).
"""
function HCRB(scheme::AbstractScheme; W = nothing, eps = GLOBAL_EPS)
    if isnothing(W)
        W = I(get_param_num(scheme))
    end
    rho, drho = evolve(scheme)
    return HCRB(rho, drho, W; eps = eps)
end

@doc raw"""
    NHB(scheme::AbstractScheme; W=nothing)

Compute the Nagaoka-Hayashi bound (NHB) from a Scheme.

Evolves the state encoded in the scheme and calls [`NHB`](@ref) on the
resulting density matrix and derivatives.

# Arguments

- `scheme::AbstractScheme`: The estimation scheme.
- `W::Union{AbstractMatrix,UniformScaling,Nothing}=nothing`: Weight matrix.
  Defaults to the identity ``\mathbb{I}``.

# Returns

- `Float64`: The NHB value.

# See Also

- [`NHB`](@ref): Direct NHB computation via SDP.
- [`HCRB`](@ref): Holevo Cramér-Rao bound (tighter bound).
"""
function NHB(scheme::AbstractScheme; W = nothing)
    if isnothing(W)
        W = I(get_param_num(scheme))
    end
    rho, drho = evolve(scheme)
    return NHB(rho, drho, W)
end

raw"""

    HCRB(ρ::AbstractMatrix, dρ::AbstractVector, W::AbstractMatrix; eps=GLOBAL_EPS)

Caltulate the Holevo Cramer-Rao bound (HCRB) via the semidefinite program (SDP).
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix on the unknown parameters to be estimated. For example, drho[0] is the derivative vector on the first parameter.
- `W`: Weight matrix.
- `eps`: Machine epsilon.
raw"""
function HCRB(ρ::AbstractMatrix, dρ::AbstractVector, W::AbstractMatrix; eps = GLOBAL_EPS)
    if length(dρ) == 1
        println(
            "In the single-parameter scenario, the HCRB is equivalent to the QFI. This function will return the value of the QFI.",
        )
        f = QFIM_SLD(ρ, dρ[1]; eps = eps)
        return f
    elseif rank(W) == 1
        println(
            "For rank-one wight matrix, the HCRB is equivalent to QFIM. This function will return the value of Tr(WF^{-1}).",
        )
        F = QFIM_SLD(ρ, dρ; eps = eps)
        return tr(W * pinv(F))
    else
        Holevo_bound(ρ, dρ, W; eps = eps)
    end
end

@doc raw"""
    Holevo_bound(ρ::AbstractMatrix, dρ::AbstractVector, W::AbstractMatrix; eps=GLOBAL_EPS)

Compute the Holevo Cramér-Rao bound (HCRB) via semidefinite programming (SDP).

The HCRB is given by the optimization

```math
\min_{V, X}\;\mathrm{Tr}(W V)
```

subject to the positive semi-definite constraint

```math
\begin{pmatrix}
V & X^\dagger R^\dagger \\
R X & \mathbb{I}
\end{pmatrix} \succeq 0,
```

and the orthogonality conditions

```math
X_i^\dagger \mathrm{vec}(\partial_j\rho) = \delta_{ij},
```

where ``R`` is the LDLᵀ factor of the correlation matrix
``S_{ab} = \mathrm{Tr}(\Lambda_a\Lambda_b\rho)``, and ``\Lambda_a`` are the
su(d) generators.

# Arguments

- `ρ::AbstractMatrix`: Density matrix.
- `dρ::AbstractVector`: Vector of derivatives ``\partial_a\rho``, one per parameter.
- `W::AbstractMatrix`: Weight matrix (must satisfy ``\mathrm{rank}(W)>1`` for
  non-trivial SDP; otherwise falls back to QFIM).
- `eps::Float64=GLOBAL_EPS`: Epsilon threshold for LDLᵀ rounding.

# Returns

- `Float64`: The HCRB value ``\min\mathrm{Tr}(W V)``.

# Note

For single-parameter or rank-1 weight matrices, this function falls back to
the QFI/QFIM and does not invoke the SDP solver.

# See Also

- [`HCRB`](@ref): Top-level dispatch (handles fallback logic).
- [`decomposition`](@ref): LDLᵀ factorization of ``S``.
- [`NHB`](@ref): Nagaoka-Hayashi bound (alternative SDP bound).
"""
function Holevo_bound(
    ρ::AbstractMatrix,
    dρ::AbstractVector,
    W::AbstractMatrix;
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
    problem = minimize(tr(W * V), constraints)
    Convex.solve!(problem, SCS.Optimizer(), silent_solver = true)
    return evaluate(tr(W * V))
end

"""
    Holevo_bound_obj(ρ::AbstractMatrix, dρ::AbstractVector, W::AbstractMatrix; eps=GLOBAL_EPS)

Wrapper around [`Holevo_bound`](@ref) that returns only the scalar objective
value, extracting the first element from the Convex.jl result.

# Arguments

- Same as [`Holevo_bound`](@ref).

# Returns

- `Float64`: The scalar HCRB value.

# See Also

- [`Holevo_bound`](@ref): Full HCRB computation via SDP.
"""
function Holevo_bound_obj(
    ρ::AbstractMatrix,
    dρ::AbstractVector,
    W::AbstractMatrix;
    eps = GLOBAL_EPS,
)
    return Holevo_bound(ρ, dρ, W; eps = eps)[1]
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
    Convex.solve!(problem, SCS.Optimizer(), silent_solver = true)
    return evaluate(real(tr(kron(W, ρ) * L)))
end
