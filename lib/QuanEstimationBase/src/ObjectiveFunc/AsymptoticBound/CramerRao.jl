using ChainRulesCore
const σ_x = [0.0 1.0; 1.0 0.0im]
const σ_y = [0.0 -1.0im; 1.0im 0.0]
const σ_z = [1.0 0.0im; 0.0 -1.0]

############## logarrithmic derivative ###############
@doc raw"""

	SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}

Calculate the symmetric logarrithmic derivatives (SLDs). The SLD operator ``L_a`` is defined 
as``\partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)``, where ``\rho`` is the parameterized density matrix. 
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the SLD operator. Options can be: "original" (default) and "eigen" .
- `eps`: Machine epsilon.
"""
function SLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    rep = "original",
    eps = GLOBAL_EPS,
) where {T<:Complex}
    (x -> SLD(ρ, x; rep = rep, eps = eps)).(dρ)
end

raw"""

	SLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}

When applied to the case of single parameter.
raw"""
function SLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    rep = "original",
    eps = GLOBAL_EPS,
) where {T<:Complex}

    dim = size(ρ)[1]
    SLD = Matrix{ComplexF64}(undef, dim, dim)

    ρ_h = (ρ + ρ') / 2
    val, vec = eigen(ρ_h)
    val = val |> real
    SLD_eig = zeros(T, dim, dim)
    for fi = 1:dim
        for fj = 1:dim
            if val[fi] + val[fj] > eps
                SLD_eig[fi, fj] = 2 * (vec[:, fi]' * dρ * vec[:, fj]) / (val[fi] + val[fj])
            end
        end
    end
    SLD_eig[findall(SLD_eig == Inf)] .= 0.0
    SLD_eig[findall(abs.(SLD_eig) .> 1e10)] .= 0.0

    if rep == "original"
        SLD = vec * (SLD_eig * vec')
        SLD = (SLD + SLD') / 2
    elseif rep == "eigen"
        SLD = SLD_eig
    else
        throw(ArgumentError("The rep should be chosen in {'original', 'eigen'}."))
    end
    return SLD
end

@doc raw"""
    ChainRulesCore.rrule(::typeof(SLD), ρ::Matrix{T}, dρ::Matrix{T}; eps=GLOBAL_EPS)

Custom reverse-mode automatic differentiation (AD) rule for the SLD operator.

This rule enables backpropagation through `SLD(ρ, dρ)` in AD frameworks
(Zygote, etc.). The forward pass computes ``L = \mathrm{SLD}(\rho, \partial\rho)``.
The pullback maps the cotangent ``\bar{L}`` to cotangents w.r.t. ``\rho`` and
``\partial\rho``.

# Mathematical Definition

Given ``\bar{L}``, the pullback computes:

```math
\bar{\rho} = -L\bar{L} - \bar{L}L,\qquad
\bar{\partial\rho} = 2\bar{L}.
```

# Arguments

- `::typeof(SLD)`: The function signature for dispatch.
- `ρ::Matrix{T}`: Density matrix.
- `dρ::Matrix{T}`: Derivative of the density matrix.
- `eps::Float64=GLOBAL_EPS`: Epsilon threshold for eigenvalue truncation.

# Returns

- `Tuple{L, pullback}`: The SLD ``L`` and a pullback closure.

# See Also

- [`SLD`](@ref): Forward SLD computation.
raw"""
function ChainRulesCore.rrule(::typeof(SLD), ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    L = SLD(ρ, dρ; eps = eps)
    function SLD_pullback(L̄)
        Ḡ = SLD(Array(ρ), Matrix{ComplexF64}(L̄) / 2)
        return ChainRulesCore.NoTangent(), -Ḡ * L - L * Ḡ, 2 * Ḡ
    end
    return L, SLD_pullback
end

@doc raw"""
    SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}; eps=GLOBAL_EPS) where {T<:Complex}

Compute the symmetric logarithmic derivative (SLD) in Liouville space via
the pseudo-inverse of the commutator super-operator.

# Mathematical Definition

In Liouville (vectorized) representation, the SLD defining equation
``\frac{1}{2}(L\rho + \rho L) = \partial\rho`` becomes

```math
\mathrm{vec}(L) = 2\bigl(\rho^{\mathsf{T}}\otimes\mathbb{I}
    + \mathbb{I}\otimes\rho\bigr)^{-1}\mathrm{vec}(\partial\rho).
```

The ``d^2\times d^2`` matrix ``(\rho^{\mathsf{T}}\otimes\mathbb{I} + \mathbb{I}\otimes\rho)``
is inverted via `pinv` with relative tolerance `eps`.

# Arguments

- `ρ::Matrix{T}`: Density matrix.
- `∂ρ_∂x::Matrix{T}`: Derivative ``\partial_x\rho``.
- `eps::Float64=GLOBAL_EPS`: Relative tolerance for pseudo-inverse.

# Returns

- `Matrix{ComplexF64}`: The SLD operator ``L`` reshaped from Liouville vector.

# See Also

- [`SLD`](@ref): Standard SLD via eigenbasis decomposition.
- [`SLD_qr`](@ref): Liouville-space SLD via QR decomposition.
"""
function SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), rtol = eps) * vec(∂ρ_∂x) |> vec2mat
end

"""
    SLD_liouville(ρ::Vector{T}, ∂ρ_∂x::Vector{T}; eps=GLOBAL_EPS) where {T<:Complex}

Vector-input convenience wrapper that reshapes vectorized density matrices
to matrix form and calls [`SLD_liouville`](@ref).
"""
function SLD_liouville(ρ::Vector{T}, ∂ρ_∂x::Vector{T}; eps = GLOBAL_EPS) where {T<:Complex}
    SLD_liouville(ρ |> vec2mat, ∂ρ_∂x |> vec2mat; eps = eps)
end

raw"""
    SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}; eps=GLOBAL_EPS) where {T<:Complex}

Multi-parameter convenience wrapper that broadcasts the single-parameter
[`SLD_liouville`](@ref) over each derivative matrix.
raw"""
function SLD_liouville(
    ρ::Matrix{T},
    ∂ρ_∂x::Vector{Matrix{T}};
    eps = GLOBAL_EPS,
) where {T<:Complex}

    (x -> SLD_liouville(ρ, x; eps = eps)).(∂ρ_∂x)
end

@doc raw"""
    SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}

Compute the symmetric logarithmic derivative (SLD) in Liouville space via
QR decomposition instead of pseudo-inverse.

This is an alternative to [`SLD_liouville`](@ref) that solves the linear
system ``(\rho^{\mathsf{T}}\otimes\mathbb{I} + \mathbb{I}\otimes\rho)\,\mathrm{vec}(L)
= 2\,\mathrm{vec}(\partial\rho)`` using `qr` (QR factorization with
column pivoting), which can be more numerically stable than `pinv`.

# Mathematical Definition

Same Liouville-space equation as [`SLD_liouville`](@ref):

```math
\mathrm{vec}(L) = 2\bigl(\rho^{\mathsf{T}}\otimes\mathbb{I}
    + \mathbb{I}\otimes\rho\bigr)^{-1}\mathrm{vec}(\partial\rho).
```

# Arguments

- `ρ::Matrix{T}`: Density matrix.
- `∂ρ_∂x::Matrix{T}`: Derivative ``\partial_x\rho``.

# Returns

- `Matrix{ComplexF64}`: The SLD operator ``L``.

# See Also

- [`SLD_liouville`](@ref): Liouville-space SLD via pseudo-inverse.
- [`SLD`](@ref): Standard SLD via eigenbasis decomposition.
raw"""
function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), ColumnNorm()) \ vec(∂ρ_∂x)) |>
    vec2mat
end

@doc raw"""

    RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}

Calculate the right logarrithmic derivatives (RLDs). The RLD operator is defined as 
``\partial_{a}\rho=\rho \mathcal{R}_a``, where ``\rho`` is the parameterized density matrix.  
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the RLD operator. Options can be: "original" (default) and "eigen".
- `eps`: Machine epsilon.
"""
function RLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    rep = "original",
    eps = GLOBAL_EPS,
) where {T<:Complex}
    (x -> RLD(ρ, x; rep = rep, eps = eps)).(dρ)
end

raw"""

	RLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}

When applied to the case of single parameter.
raw"""
function RLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    rep = "original",
    eps = GLOBAL_EPS,
) where {T<:Complex}

    dim = size(ρ)[1]
    RLD = Matrix{ComplexF64}(undef, dim, dim)

    ρ_h = (ρ + ρ') / 2
    val, vec = eigen(ρ_h)
    val = val |> real
    RLD_eig = zeros(T, dim, dim)
    for fi = 1:dim
        for fj = 1:dim
            term_tp = (vec[:, fi]' * dρ * vec[:, fj])
            if val[fi] > eps
                RLD_eig[fi, fj] = term_tp / val[fi]
            else
                if abs(term_tp) > eps
                    throw(
                        ErrorException(
                            "The RLD does not exist. It only exist when the support of drho is contained in the support of rho.",
                        ),
                    )
                end
            end
        end
    end
    RLD_eig[findall(RLD_eig == Inf)] .= 0.0
    RLD_eig[findall(abs.(RLD_eig) .> 1e10)] .= 0.0

    if rep == "original"
        RLD = vec * (RLD_eig * vec')
    elseif rep == "eigen"
        RLD = RLD_eig
    end
    return RLD
end


@doc raw"""

    LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}

Calculate the left logarrithmic derivatives (LLDs). The LLD operator is defined as ``\partial_{a}\rho=\mathcal{L}_a\rho``,
where ``\mathcal{L}_a = \mathcal{R}_a^\dagger`` and ``\mathcal{R}_a`` is the RLD operator.  
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the LLD operator. Options can be: "original" (default) and "eigen".
- `eps`: Machine epsilon.
"""
function LLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    rep = "original",
    eps = GLOBAL_EPS,
) where {T<:Complex}
    (x -> RLD(ρ, x; rep = rep, eps = eps)').(dρ)
end

"""

    LLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=GLOBAL_EPS) where {T<:Complex}

When applied to the case of single parameter.
raw"""
function LLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    rep = "original",
    eps = GLOBAL_EPS,
) where {T<:Complex}
    return RLD(ρ, dρ; rep = rep, eps = eps)'
end


#========================================================#
####################### calculate QFI ####################
@doc raw"""
    QFIM_SLD(ρ::Matrix{T}, dρ::Matrix{T}; eps=GLOBAL_EPS) where {T<:Complex}

Compute the single-parameter quantum Fisher information (QFI) via the symmetric
logarithmic derivative (SLD).

# Mathematical Definition

The SLD-based QFI for a single parameter ``x`` is given by

```math
F_x = \mathrm{Tr}(\rho L_x^2) = \frac{1}{2}\mathrm{Tr}\bigl(\rho\{L_x, L_x\}\bigr),
```

where ``L_x`` is the SLD operator defined by
``\frac{1}{2}(\rho L_x + L_x\rho) = \partial_x\rho``.

# Arguments

- `ρ::Matrix{T}`: Density matrix (positive semi-definite).
- `dρ::Matrix{T}`: Derivative ``\partial_x\rho`` with respect to the single parameter.
- `eps::Float64=GLOBAL_EPS`: Threshold for treating near-zero eigenvalues as zero.

# Returns

- `Float64`: The real part of ``\mathrm{Tr}(\rho L_x^2)``.

# See Also

- [`SLD`](@ref): Symmetric logarithmic derivative operator.
- [`QFIM_SLD`](@ref) (multi-parameter): Multi-parameter SLD-based QFIM.
raw"""
function QFIM_SLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    SLD_tp = SLD(ρ, dρ; eps = eps)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end

@doc raw"""
    QFIM_RLD(ρ::Matrix{T}, dρ::Matrix{T}; eps=GLOBAL_EPS) where {T<:Complex}

Compute the single-parameter quantum Fisher information (QFI) via the right
logarithmic derivative (RLD).

# Mathematical Definition

The RLD-based QFI for a single parameter ``x`` is

```math
F_x^{\mathrm{RLD}} = \mathrm{Tr}\bigl(\rho \mathcal{R}_x \mathcal{R}_x^\dagger\bigr),
```

where ``\mathcal{R}_x`` is the RLD operator defined by ``\partial_x\rho = \rho\mathcal{R}_x``.

# Arguments

- `ρ::Matrix{T}`: Density matrix (positive semi-definite).
- `dρ::Matrix{T}`: Derivative ``\partial_x\rho`` with respect to the single parameter.
- `eps::Float64=GLOBAL_EPS`: Threshold for treating near-zero eigenvalues as zero.

# Returns

- `Float64`: The real part of ``\mathrm{Tr}(\rho\mathcal{R}_x\mathcal{R}_x^\dagger)``.

# See Also

- [`RLD`](@ref): Right logarithmic derivative operator.
- [`QFIM_RLD`](@ref) (multi-parameter): Multi-parameter RLD-based QFIM.
- [`QFIM_LLD`](@ref): LLD-based QFI (equivalent to RLD-based QFI).
raw"""
function QFIM_RLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    R = RLD(ρ, dρ; eps = eps)
    return real(tr(ρ * R * R'))
end

@doc raw"""
    QFIM_LLD(ρ::Matrix{T}, dρ::Matrix{T}; eps=GLOBAL_EPS) where {T<:Complex}

Compute the single-parameter quantum Fisher information (QFI) via the left
logarithmic derivative (LLD).

# Mathematical Definition

The LLD-based QFI for a single parameter ``x`` is

```math
F_x^{\mathrm{LLD}} = \mathrm{Tr}\bigl(\rho \mathcal{L}_x^\dagger \mathcal{L}_x\bigr)
= \mathrm{Tr}\bigl(\rho \mathcal{R}_x \mathcal{R}_x^\dagger\bigr),
```

where ``\mathcal{L}_x = \mathcal{R}_x^\dagger`` is the LLD operator and
``\partial_x\rho = \mathcal{L}_x\rho``.

# Arguments

- `ρ::Matrix{T}`: Density matrix (positive semi-definite).
- `dρ::Matrix{T}`: Derivative ``\partial_x\rho`` with respect to the single parameter.
- `eps::Float64=GLOBAL_EPS`: Threshold for treating near-zero eigenvalues as zero.

# Returns

- `Float64`: The real part of ``\mathrm{Tr}(\rho\mathcal{L}_x^\dagger\mathcal{L}_x)``.

# See Also

- [`LLD`](@ref): Left logarithmic derivative operator.
- [`QFIM_LLD`](@ref) (multi-parameter): Multi-parameter LLD-based QFIM.
- [`QFIM_RLD`](@ref): RLD-based QFI (equivalent to LLD-based QFI).
raw"""
function QFIM_LLD(ρ::Matrix{T}, dρ::Matrix{T}; eps = GLOBAL_EPS) where {T<:Complex}
    L = LLD(ρ, dρ; eps = eps)
    return real(tr(ρ * L' * L))
end

@doc raw"""
    QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}

Compute the single-parameter quantum Fisher information (QFI) for a pure state.

# Mathematical Definition

For a pure state ``\rho = |\psi\rangle\langle\psi|`` with
``\partial_x\rho = |\partial_x\psi\rangle\langle\psi| + |\psi\rangle\langle\partial_x\psi|``,
the QFI simplifies to

```math
F_x = 4\bigl(\langle\partial_x\psi|\partial_x\psi\rangle
- |\langle\partial_x\psi|\psi\rangle|^2\bigr).
```

In the code, the SLD is obtained as ``L_x = 2\partial_x\rho``, which is valid
for pure states.

# Arguments

- `ρ::Matrix{T}`: Pure-state density matrix.
- `∂ρ_∂x::Matrix{T}`: Derivative ``\partial_x\rho`` with respect to the single parameter.

# Returns

- `Float64`: The real part of ``\mathrm{Tr}(\rho L_x^2)``.

# See Also

- [`QFIM_pure`](@ref) (multi-parameter): Multi-parameter pure-state QFIM.
- [`QFIM_SLD`](@ref): SLD-based QFI for general (mixed) states.
raw"""
function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T<:Complex}
    SLD = 2 * ∂ρ_∂x
    SLD2_tp = SLD * SLD
    F = tr(ρ * SLD2_tp)
    F |> real
end

#==========================================================#
####################### calculate QFIM #####################
@doc raw"""
    QFIM_SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps=GLOBAL_EPS) where {T<:Complex}

Compute the multi-parameter quantum Fisher information matrix (QFIM) via
the symmetric logarithmic derivative (SLD).

# Mathematical Definition

For multiple parameters ``\mathbf{x} = (x_1, \dots, x_p)``, the SLD-based
QFIM entries are

```math
F_{ab} = \frac{1}{2}\mathrm{Tr}\bigl(\rho\{L_a, L_b\}\bigr)
       = \mathrm{Re}\,\mathrm{Tr}(\rho L_a L_b),
```

where ``L_a`` is the SLD operator for parameter ``a`` satisfying
``\frac{1}{2}(L_a\rho + \rho L_a) = \partial_a\rho``.

# Arguments

- `ρ::Matrix{T}`: Density matrix (positive semi-definite).
- `dρ::Vector{Matrix{T}}`: Vector of derivatives ``\partial_a\rho``, one per parameter.
- `eps::Float64=GLOBAL_EPS`: Threshold for treating near-zero eigenvalues as zero.

# Returns

- `Matrix{Float64}`: The ``p\times p`` QFIM with entries ``F_{ab}``.

# See Also

- [`SLD`](@ref): Symmetric logarithmic derivative operator.
- [`QFIM_SLD`](@ref) (single-parameter): Single-parameter SLD-based QFI.
raw"""
function QFIM_SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    p_num = length(dρ)
    LD_tp = (x -> SLD(ρ, x; eps = eps)).(dρ)
    return [real(tr(0.5 * ρ * (LD_tp[i] * LD_tp[j] + LD_tp[j] * LD_tp[i]))) for i in 1:p_num, j in 1:p_num]
end

@doc raw"""
    QFIM_RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps=GLOBAL_EPS) where {T<:Complex}

Compute the multi-parameter quantum Fisher information matrix (QFIM) via
the right logarithmic derivative (RLD).

# Mathematical Definition

The RLD-based QFIM entries are

```math
F_{ab}^{\mathrm{RLD}} = \mathrm{Tr}\bigl(\rho \mathcal{R}_a \mathcal{R}_b^\dagger\bigr),
```

where ``\mathcal{R}_a`` is the RLD operator satisfying
``\partial_a\rho = \rho\mathcal{R}_a``. The RLD QFIM is complex-Hermitian;
this function returns the full complex matrix (no ``\mathrm{Re}`` cast).

# Arguments

- `ρ::Matrix{T}`: Density matrix (positive semi-definite).
- `dρ::Vector{Matrix{T}}`: Vector of derivatives ``\partial_a\rho``, one per parameter.
- `eps::Float64=GLOBAL_EPS`: Threshold for treating near-zero eigenvalues as zero.

# Returns

- `Matrix{ComplexF64}`: The ``p\times p`` RLD QFIM (complex-Hermitian).

# See Also

- [`RLD`](@ref): Right logarithmic derivative operator.
- [`QFIM_RLD`](@ref) (single-parameter): Single-parameter RLD-based QFI.
- [`QFIM_LLD`](@ref): LLD-based QFIM (``F_{ab}^{\mathrm{LLD}}=\mathrm{Tr}(\rho\mathcal{L}_a^\dagger\mathcal{L}_b)=\mathrm{Tr}(\rho\mathcal{R}_a\mathcal{R}_b^\dagger)``).
raw"""
function QFIM_RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    p_num = length(dρ)
    R = RLD(ρ, dρ; eps = eps)
    return [tr(ρ * R[i] * R[j]') for i in 1:p_num, j in 1:p_num]
end

@doc raw"""
    QFIM_LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps=GLOBAL_EPS) where {T<:Complex}

Compute the multi-parameter quantum Fisher information matrix (QFIM) via
the left logarithmic derivative (LLD).

# Mathematical Definition

The LLD-based QFIM entries are

```math
F_{ab}^{\mathrm{LLD}} = \mathrm{Tr}\bigl(\rho \mathcal{L}_a^\dagger \mathcal{L}_b\bigr)
= \mathrm{Tr}\bigl(\rho \mathcal{R}_a \mathcal{R}_b^\dagger\bigr),
```

where ``\mathcal{L}_a = \mathcal{R}_a^\dagger`` is the LLD operator and
``\partial_a\rho = \mathcal{L}_a\rho``. The LLD and RLD QFIMs coincide.

# Arguments

- `ρ::Matrix{T}`: Density matrix (positive semi-definite).
- `dρ::Vector{Matrix{T}}`: Vector of derivatives ``\partial_a\rho``, one per parameter.
- `eps::Float64=GLOBAL_EPS`: Threshold for treating near-zero eigenvalues as zero.

# Returns

- `Matrix{ComplexF64}`: The ``p\times p`` LLD QFIM (complex-Hermitian).

# See Also

- [`LLD`](@ref): Left logarithmic derivative operator.
- [`QFIM_LLD`](@ref) (single-parameter): Single-parameter LLD-based QFI.
- [`QFIM_RLD`](@ref): RLD-based QFIM (identical to LLD-based QFIM).
raw"""
function QFIM_LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps = GLOBAL_EPS) where {T<:Complex}
    p_num = length(dρ)
    L = LLD(ρ, dρ; eps = eps)
    return [tr(ρ * L[i]' * L[j]) for i in 1:p_num, j in 1:p_num]
end

@doc raw"""
    QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}

Compute the multi-parameter quantum Fisher information matrix (QFIM) for a pure state.

# Mathematical Definition

For a pure state ``\rho = |\psi\rangle\langle\psi|``, the QFIM entries are

```math
F_{ab} = 4\,\mathrm{Re}\bigl(
    \langle\partial_a\psi|\partial_b\psi\rangle
    - \langle\partial_a\psi|\psi\rangle\langle\psi|\partial_b\psi\rangle
\bigr).
```

The factor ``\mathrm{Re}(\cdot)`` is essential; the imaginary part cancels in
the trace formula. In the code, the SLD is computed as ``L_a = 2\partial_a\rho``,
which is valid for pure states, then the symmetric trace formula is used.

# Arguments

- `ρ::Matrix{T}`: Pure-state density matrix.
- `∂ρ_∂x::Vector{Matrix{T}}`: Vector of derivatives ``\partial_a\rho``, one per parameter.

# Returns

- `Matrix{Float64}`: The ``p\times p`` QFIM with entries ``F_{ab}`` (real, symmetric,
  positive semi-definite).

# See Also

- [`QFIM_pure`](@ref) (single-parameter): Single-parameter pure-state QFI.
- [`QFIM_SLD`](@ref): SLD-based QFIM for general (mixed) states.
"""
function QFIM_pure(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T<:Complex}
    p_num = length(∂ρ_∂x)
    sld = [2 * ∂ρ_∂x[i] for i = 1:p_num]
    return [real(tr(0.5 * ρ * (sld[i] * sld[j] + sld[j] * sld[i]))) for i in 1:p_num, j in 1:p_num]
end

#======================================================#
#################### calculate CFIM ####################
@doc raw"""

	CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps=GLOBAL_EPS) where {T<:Complex}

Calculate the classical Fisher information matrix (CFIM). 
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `eps`: Machine epsilon.
"""
function CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps = GLOBAL_EPS) where {T<:Complex}
    m_num = length(M)
    p_num = length(dρ)
    [
        real(tr(ρ * M[i])) < eps ? zeros(ComplexF64, p_num, p_num) :
        (kron(tr.(dρ .* [M[i]]), reshape(tr.(dρ .* [M[i]]), 1, p_num)) / tr(ρ * M[i])) for
        i = 1:m_num
    ] |>
    sum .|>
    real
end

"""

	CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps=GLOBAL_EPS) where {T<:Complex}

When applied to the case of single parameter. Calculate the classical Fisher information (CFI). 
"""
function CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps = GLOBAL_EPS) where {T<:Complex}
    m_num = length(M)
    F = 0.0
    for i = 1:m_num
        mp = M[i]
        p = real(tr(ρ * mp))
        dp = real(tr(dρ * mp))
        cadd = 0.0
        if p > eps
            cadd = (dp * dp) / p
        end
        F += cadd
    end
    real(F)
end

"""

	CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; M=nothing, eps=GLOBAL_EPS) where {T<:Complex}

When the set of POVM is not given. Calculate the CFIM with SIC-POVM. The SIC-POVM is generated from the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).
"""
function CFIM(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    M = nothing,
    eps = GLOBAL_EPS,
) where {T<:Complex}
    M = SIC(size(ρ)[1])
    m_num = length(M)
    p_num = length(dρ)
    [
        real(tr(ρ * M[i])) < eps ? zeros(ComplexF64, p_num, p_num) :
        (kron(tr.(dρ .* [M[i]]), reshape(tr.(dρ .* [M[i]]), 1, p_num)) / tr(ρ * M[i])) for
        i = 1:m_num
    ] |>
    sum .|>
    real
end

"""

	CFIM(ρ::Matrix{T}, dρ::Matrix{T}; eps=GLOBAL_EPS) where {T<:Complex}

When applied to the case of single parameter and the set of POVM is not given. Calculate the CFI with SIC-POVM. 
"""
function CFIM(ρ::Matrix{T}, dρ::Matrix{T}; M = nothing, eps = GLOBAL_EPS) where {T<:Complex}
    M = SIC(size(ρ)[1])
    m_num = length(M)
    F = 0.0
    for i = 1:m_num
        mp = M[i]
        p = real(tr(ρ * mp))
        dp = real(tr(dρ * mp))
        cadd = 0.0
        if p > eps
            cadd = (dp * dp) / p
        end
        F += cadd
    end
    real(F)
end

@doc raw"""
    CFIM(scheme::Scheme; full_trajectory=false, LDtype=:SLD, exportLD=false, eps=GLOBAL_EPS)

Compute the classical Fisher information (matrix) from a full ``Scheme``.

This is the top-level dispatch that extracts the measurement POVM from the
scheme, evolves the state, and evaluates the CFIM.

# Arguments

- `scheme::Scheme`: The estimation scheme bundling probe, dynamics, measurement,
  and estimation strategy.
- `full_trajectory::Bool=false`: If `true`, return CFIM for each time step in
  the trajectory (uses matrix exponential). If `false`, return CFIM at the
  final time (uses ODE/expm evolution).
- `LDtype::Symbol=:SLD`: Placeholder (ignored for CFIM); kept for API uniformity
  with `QFIM`.
- `exportLD::Bool=false`: Placeholder (ignored for CFIM).
- `eps::Float64=GLOBAL_EPS`: Threshold for treating near-zero probabilities as zero.

# Returns

- `Matrix{Float64}` or `Vector{Matrix{Float64}}`: The CFIM. If `full_trajectory=true`,
  returns a vector of matrices, one per time step.

# See Also

- [`CFIM`](@ref): Direct matrix CFIM computation.
- [`QFIM`](@ref): Quantum Fisher information (matrix) from a Scheme.
"""
function CFIM(
    scheme::Scheme;
    full_trajectory = false,
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = GLOBAL_EPS,
)
    M = meas_data(scheme)
    if full_trajectory
        rho, drho = expm(scheme)
        return [CFIM(r, dr, M; eps = eps) for (r, dr) in zip(rho, drho)]
    else
        rho, drho = evolve(scheme)
        return CFIM(rho, drho, M; eps = eps)
    end
end

## QFI with exportLD
"""

    QFIM(ρ::Matrix{T}, dρ::Matrix{T}; LDtype=:SLD, exportLD::Bool= false, eps=GLOBAL_EPS) where {T<:Complex}

Calculation of the quantum Fisher information (QFI) for all types. 
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.
- `exportLD`: export logarithmic derivatives apart from F.
- `eps`: Machine epsilon.
"""
function QFIM(
    ρ::Matrix{T},
    dρ::Matrix{T};
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = GLOBAL_EPS,
) where {T<:Complex}

    if LDtype == :SLD
        F = QFIM_SLD(ρ, dρ; eps = eps)
    elseif LDtype == :RLD
        F = QFIM_RLD(ρ, dρ; eps = eps)
    elseif LDtype == :LLD
        F = QFIM_LLD(ρ, dρ; eps = eps)
    else
        throw(ArgumentError("LDtype must be :SLD, :RLD, or :LLD"))
    end
    if exportLD == false
        return F
    else
        if LDtype == :SLD
            LD = SLD(ρ, dρ; eps = eps)
        elseif LDtype == :RLD
            LD = RLD(ρ, dρ; eps = eps)
        elseif LDtype == :LLD
            LD = LLD(ρ, dρ; eps = eps)
        end
        return F, LD
    end
end

"""

    QFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; LDtype=:SLD, exportLD::Bool= false, eps=GLOBAL_EPS) where {T<:Complex}

When applied to the case of single parameter. Calculation of the quantum Fisher information (QFI) for all types.
"""
function QFIM(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = GLOBAL_EPS,
) where {T<:Complex}

    if LDtype == :SLD
        F = QFIM_SLD(ρ, dρ; eps = eps)
    elseif LDtype == :RLD
        F = QFIM_RLD(ρ, dρ; eps = eps)
    elseif LDtype == :LLD
        F = QFIM_LLD(ρ, dρ; eps = eps)
    else
        throw(ArgumentError("The LDtype should be chosen in {'SLD', 'RLD', 'LLD'}."))
    end

    if exportLD == false
        return F
    else
        if LDtype == :SLD
            LD = SLD(ρ, dρ; eps = eps)
        elseif LDtype == :RLD
            LD = RLD(ρ, dρ; eps = eps)
        elseif LDtype == :LLD
            LD = LLD(ρ, dρ; eps = eps)
        end
        return F, LD
    end
end

@doc raw"""
    QFIM(scheme::Scheme; full_trajectory=false, LDtype=:SLD, exportLD=false, eps=GLOBAL_EPS)

Compute the quantum Fisher information (matrix) from a full ``Scheme``.

This is the top-level dispatch that evolves the state encoded in the scheme
and evaluates the QFIM using the chosen logarithmic derivative type.

# Arguments

- `scheme::Scheme`: The estimation scheme bundling probe, dynamics, measurement,
  and estimation strategy.
- `full_trajectory::Bool=false`: If `true`, return QFIM for each time step in
  the trajectory (uses matrix exponential). If `false`, return QFIM at the
  final time (uses ODE/expm evolution).
- `LDtype::Symbol=:SLD`: Type of logarithmic derivative. Options are `:SLD`
  (default), `:RLD`, and `:LLD`.
- `exportLD::Bool=false`: If `true`, also return the logarithmic derivative
  operators alongside the QFIM.
- `eps::Float64=GLOBAL_EPS`: Threshold for treating near-zero eigenvalues as zero.

# Returns

- `Union{Float64,Matrix{Float64}}`: The QFI (single parameter) or QFIM (multiple
  parameters). If `exportLD=true`, returns a tuple `(F, LD)`.

# See Also

- [`QFIM`](@ref): Direct matrix QFIM computation.
- [`CFIM`](@ref): Classical Fisher information from a Scheme.
"""
function QFIM(
    scheme::Scheme;
    full_trajectory = false,
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = GLOBAL_EPS,
)
    if full_trajectory
        rho, drho = expm(scheme)
        return [
            QFIM(r, dr; LDtype = LDtype, exportLD = exportLD, eps = eps) for
            (r, dr) in zip(rho, drho)
        ]
    else
        rho, drho = evolve(scheme)
        return QFIM(rho, drho; LDtype = LDtype, exportLD = exportLD, eps = eps)
    end
end



"""

    QFIM_Kraus(ρ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector; LDtype=:SLD, exportLD::Bool=false, eps=GLOBAL_EPS)

Calculation of the quantum Fisher information (QFI) and quantum Fisher information matrix (QFIM) with Kraus operator(s) for all types.
- `ρ0`: Density matrix.
- `K`: Kraus operator(s).
- `dK`: Derivatives of the Kraus operator(s) on the unknown parameters to be estimated. For example, dK[0] is the derivative vector on the first parameter.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.
- `exportLD`: Whether or not to export the values of logarithmic derivatives. If set True then the the values of logarithmic derivatives will be exported.
- `eps`: Machine epsilon.
"""
function QFIM_Kraus(
    ρ0::AbstractMatrix,
    K::AbstractVector,
    dK::AbstractVector;
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = GLOBAL_EPS,
)
    para_num = length(dK[1])
    dK = [[dK[i][j] for i in eachindex(K)] for j = 1:para_num]
    ρ = [K * ρ0 * K' for K in K] |> sum
    dρ = [[dK * ρ0 * K' + K * ρ0 * dK' for (K, dK) in zip(K, dK)] |> sum for dK in dK]
    F = QFIM(ρ, dρ; LDtype = LDtype, exportLD = exportLD, eps = eps)
    if para_num == 1
        # single-parameter scenario
        return F[1, 1]
    else
        # multiparameter scenario
        return F
    end
end

"""

	QFIM_Bloch(r, dr; eps=GLOBAL_EPS)

Calculate the SLD based quantum Fisher information (QFI) or quantum Fisher information matrix (QFIM) in Bloch representation.
- `r`: Parameterized Bloch vector.
- `dr`: Derivative(s) of the Bloch vector with respect to the unknown parameters to be estimated. For example, dr[1] is the derivative vector with respect to the first parameter.
- `eps`: Machine epsilon.
"""
function QFIM_Bloch(r, dr; eps = GLOBAL_EPS)
    para_num = length(dr)
    QFIM_res = zeros(para_num, para_num)

    dim = Int(sqrt(length(r) + 1))
    Lambda = suN_generator(dim)
    if dim == 2
        r_norm = norm(r)^2
        if abs(r_norm - 1.0) < eps
            for para_i = 1:para_num
                for para_j = para_i:para_num
                    QFIM_res[para_i, para_j] = real(dr[para_i]' * dr[para_j])
                    QFIM_res[para_j, para_i] = QFIM_res[para_i, para_j]
                end
            end
        else
            for para_i = 1:para_num
                for para_j = para_i:para_num
                    QFIM_res[para_i, para_j] = real(
                        dr[para_i]' * dr[para_j] +
                        (r' * dr[para_i]) * (r' * dr[para_j]) / (1 - r_norm),
                    )
                    QFIM_res[para_j, para_i] = QFIM_res[para_i, para_j]
                end
            end
        end
    else
        rho = (Matrix(I, dim, dim) + sqrt(dim * (dim - 1) / 2) * r' * Lambda) / dim
        G = zeros(ComplexF64, dim^2 - 1, dim^2 - 1)
        for row_i = 1:dim^2-1
            for col_j = row_i:dim^2-1
                anti_commu = Lambda[row_i] * Lambda[col_j] + Lambda[col_j] * Lambda[row_i]
                G[row_i, col_j] = 0.5 * tr(rho * anti_commu)
                G[col_j, row_i] = G[row_i, col_j]
            end
        end

        mat_tp = G * dim / (2 * (dim - 1)) - r * r'
        mat_inv = pinv(mat_tp)

        for para_m = 1:para_num
            for para_n = para_m:para_num
                QFIM_res[para_m, para_n] = real(dr[para_n]' * mat_inv * dr[para_m])
                QFIM_res[para_n, para_m] = QFIM_res[para_m, para_n]
            end
        end
    end
    if para_num == 1
        return QFIM_res[1, 1]
    else
        return QFIM_res
    end
end

"""

    FIM(p::Vector{R}, dp::Vector{R}; eps=GLOBAL_EPS) where {R<:Real}

When applied to the case of single parameter and the set of POVM is not given. Calculate the classical Fisher information for classical scenarios. 
"""
function FIM(p::Vector{R}, dp::Vector{R}; eps = GLOBAL_EPS) where {R<:Real}
    m_num = length(p)
    F = 0.0
    for i = 1:m_num
        p_tp = p[i]
        dp_tp = dp[i]
        cadd = 0.0
        if p_tp > eps
            cadd = (dp_tp * dp_tp) / p_tp
        end
        F += cadd
    end
    real(F)
end

"""

    FIM(p::Vector{R}, dp::Vector{R}; eps=GLOBAL_EPS) where {R<:Real}

Calculation of the classical Fisher information matrix for classical scenarios. 
- `p`: The probability distribution.
- `dp`: Derivatives of the probability distribution on the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `eps`: Machine epsilon.
"""
function FIM(p::Vector{R}, dp::Vector{Vector{R}}; eps = GLOBAL_EPS) where {R<:Real}
    m_num = length(p)
    para_num = length(dp[1])

    FIM_res = zeros(para_num, para_num)
    for pj = 1:m_num
        p_tp = p[pj]
        Cadd = zeros(para_num, para_num)
        if p_tp > eps
            for para_i = 1:para_num
                dp_i = dp[pj][para_i]
                for para_j = para_i:para_num
                    dp_j = dp[pj][para_j]
                    Cadd[para_i, para_j] = real(dp_i * dp_j / p_tp)
                    Cadd[para_j, para_i] = real(dp_i * dp_j / p_tp)
                end
            end
            FIM_res += Cadd
        end
    end
    if length(dp[1]) == 1
        # single-parameter scenario
        return FIM_res[1, 1]
    else
        # multiparameter scenario
        return FIM_res
    end

end

raw"""

    FI_Expt(y1, y2, dx; ftype=:norm)

Calculate the classical Fisher information (CFI) based on the experiment data.
- `y1`: Experimental data obtained at the truth value (x).
- `y1`: Experimental data obtained at x+dx.
- `dx`: A known small drift of the parameter.
- `ftype`: The distribution the data follows. Options are: norm, gamma, rayleigh, and poisson.
raw"""
function FI_Expt(y1, y2, dx; ftype = :norm)
    Fc = 0.0
    if ftype == :norm
        p1_norm = fit(Normal, y1)
        p2_norm = fit(Normal, y2)
        f_norm(x) = sqrt(pdf(p1_norm, x) * pdf(p2_norm, x))
        fidelity, err = quadgk(f_norm, -Inf, Inf)
        Fc = 8 * (1 - fidelity) / dx^2
    elseif ftype == :gamma
        p1_gamma = fit(Gamma, y1)
        p2_gamma = fit(Gamma, y2)
        f_gamma(x) = sqrt(pdf(p1_gamma, x) * pdf(p2_gamma, x))
        fidelity, err = quadgk(f_gamma, 0.0, Inf)
        Fc = 8 * (1 - fidelity) / dx^2
    elseif ftype == :rayleigh
        p1_rayl = fit(Rayleigh, y1)
        p2_rayl = fit(Rayleigh, y2)
        f_rayl(x) = sqrt(pdf(p1_rayl, x) * pdf(p2_rayl, x))
        fidelity, err = quadgk(f_rayl, 0.0, Inf)
        Fc = 8 * (1 - fidelity) / dx^2
    elseif ftype == :poisson
        p1_pois = pdf.(fit(Poisson, y1), range(0, maximum(y1), step = 1))
        p2_pois = pdf.(fit(Poisson, y2), range(0, maximum(y2), step = 1))
        p1_pois, p2_pois = p1_pois / sum(p1_pois), p2_pois / sum(p2_pois)
        fidelity = sum([sqrt(p1_pois[i] * p2_pois[i]) for i in eachindex(p1_pois)])
        Fc = 8 * (1 - fidelity) / dx^2
    else
        println("supported values for ftype are 'norm', 'poisson', 'gamma' and 'rayleigh'")
    end
    return Fc
end


#======================================================#
################# Gaussian States QFIM #################
@doc raw"""
    Williamson_form(A::AbstractMatrix)

Perform the Williamson decomposition of a positive-definite ``2N\times 2N``
covariance matrix ``\sigma``.

# Mathematical Definition

Any positive-definite real symmetric matrix ``\sigma`` can be decomposed as

```math
\sigma = S\,\mathrm{diag}(\nu_1,\dots,\nu_N,\nu_1,\dots,\nu_N)\,S^{\mathsf{T}},
```

where ``S`` is a symplectic matrix and ``\nu_k > 0`` are the symplectic
eigenvalues. This function returns ``S`` and the vector ``(\nu_1,\dots,\nu_N)``.

The algorithm uses: ``B = \sqrt{\sigma} J \sqrt{\sigma}``, Schur decomposition
of ``B`` to extract imaginary parts of eigenvalues, and constructs ``S`` from
the resulting diagonalising transformation.

# Arguments

- `A::AbstractMatrix`: The ``2N\times 2N`` covariance matrix ``\sigma`` (must be
  positive-definite).

# Returns

- `S::Matrix`: The symplectic matrix ``S``.
- `c::Vector`: The symplectic eigenvalues ``(\nu_1,\dots,\nu_N)``.

# See Also

- [`QFIM_Gauss`](@ref): Gaussian-state QFIM using Williamson form.
raw"""
function Williamson_form(A::AbstractMatrix)
    n = size(A)[1] // 2 |> Int
    J = zeros(n, n) |> x -> [x one(x); -one(x) x]
    A_sqrt = sqrt(A)
    B = A_sqrt * J * A_sqrt
    P = one(A) |> x -> [x[:, 1:2:2n-1] x[:, 2:2:2n]]
    t, Q, vals = schur(B)
    c = sort(filter(x -> imag(x) > 0, vals); by=imag) .|> imag
    D = c |> diagm |>complex |> x -> x^(-0.5)
    S = (J * A_sqrt * Q * P * [zeros(n, n) -D; D zeros(n, n)] |> transpose |> inv) * transpose(P)
    return S, c
end

const a_Gauss = [im * σ_y, σ_z, σ_x |> one, σ_x]

@doc raw"""
    A_Gauss(m::Int)

Construct the auxiliary tensor ``\mathbf{A}`` for the Gaussian-state QFIM
calculation.

For an ``N``-mode Gaussian state, this function builds the set of
``N\times N`` tensor-product basis matrices ``|j\rangle\langle k|\otimes\sigma_l``
where ``\sigma_l`` are the Pauli matrices (and identity). These are used by
[`G_Gauss`](@ref) to construct the inverse-covariance-weighted factors.

# Arguments

- `m::Int`: Number of modes (NOT the covariance matrix). The dimension of
  the Hilbert space is ``m``.

# Returns

- `Vector{Matrix}`: A vector of ``4\times m^2`` auxiliary matrices ``A^{(l)}_{jk}``.

# See Also

- [`G_Gauss`](@ref): Constructs the ``G`` matrix for each parameter.
- [`QFIM_Gauss`](@ref): Gaussian-state QFIM.
raw"""
function A_Gauss(m::Int)
    e = bases(m)
    s = e .* e'
    a_Gauss .|> x -> [kron(s, x) / sqrt(2) for s in s]
end

@doc raw"""
    G_Gauss(S::M, dC::VM, c::V) where {M<:AbstractMatrix,V,VM<:AbstractVector}

Construct the Gaussian QFIM kernel matrices ``G_x`` for each parameter ``x``.

For a Gaussian state with covariance matrix ``\sigma`` decomposed via
[`Williamson_form`](@ref) as ``\sigma = S D S^{\mathsf{T}}``, the Gaussian
QFIM entries are

```math
F_{ab} = \mathrm{Tr}(G_a\,\partial_b\sigma) + (\partial_a\bar{R})^{\mathsf{T}}\sigma^{-1}(\partial_b\bar{R}),
```

where the matrices ``G_x`` are built from the symplectic decomposition:

```math
G_x = \sum_{j,k,l} \frac{\mathrm{Tr}\bigl[\sigma^{-1}(\partial_x\sigma)\sigma^{-1}A^{(l)}_{jk}\bigr]}
{4\nu_j\nu_k + (-1)^l}\;
\sigma^{-1}A^{(l)}_{jk}\sigma^{-1}.
```

The denominator uses ``(-1)^l`` (note: literature Eq. (946) in
arXiv:1907.08037v3 uses ``(-1)^{l+1}``; the sign difference arises from
indexing convention).

# Arguments

- `S::AbstractMatrix`: Symplectic matrix from Williamson decomposition.
- `dC::AbstractVector`: Vector of derivative matrices ``\partial_x C`` of the
  covariance matrix (``C_{ij} = \sigma_{ij} - \bar{R}_i\bar{R}_j``).
- `c::AbstractVector`: Symplectic eigenvalues ``\nu_1,\dots,\nu_m``.

# Returns

- `Vector{Matrix}`: Vector of ``G_x`` matrices, one per parameter.

# See Also

- [`Williamson_form`](@ref): Williamson decomposition.
- [`A_Gauss`](@ref): Auxiliary ``A`` matrices.
- [`QFIM_Gauss`](@ref): Gaussian-state QFIM.
"""
function G_Gauss(S::M, dC::VM, c::V) where {M<:AbstractMatrix,V,VM<:AbstractVector}
    para_num = length(dC)
    m = size(S)[1] // 2 |> Int
    As = A_Gauss(m)
    gs = [
        [[inv(S) * ∂ₓC * inv(transpose(S)) * a' |> tr for a in A] for A in As] for ∂ₓC in dC
    ]
    G = [zero(S) for _ = 1:para_num]

    for i = 1:para_num
        for j = 1:m
            for k = 1:m
                for l = 1:4
                    G[i] +=
                        gs[i][l][j, k] / (4 * c[j]c[k] + (-1)^l) *
                        inv(transpose(S)) *
                        As[l][j, k] *
                        inv(S)
                end
            end
        end
    end
    return G
end

"""

	QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM<:AbstractVecOrMat}

Calculate the SLD based quantum Fisher information matrix (QFIM) with gaussian states.  
- `R̄` : First-order moment.
- `dR̄`: Derivatives of the first-order moment with respect to the unknown parameters to be estimated. For example, dR[1] is the derivative vector on the first parameter. 
- `D`: Second-order moment.
- `dD`: Derivatives of the second-order moment with respect to the unknown parameters to be estimated. 
- `eps`: Machine epsilon.
"""
function QFIM_Gauss(R̄::V, dR̄::VV, D::M, dD::VM) where {V,VV,M,VM<:AbstractVecOrMat}
    para_num = length(dR̄)
    quad_num = length(R̄)
    C = [D[i, j] - R̄[i]R̄[j] for i = 1:quad_num, j = 1:quad_num]
    dC = [
        [dD[k][i, j] - dR̄[k][i]R̄[j] - R̄[i]dR̄[k][j] for i = 1:quad_num, j = 1:quad_num] for k = 1:para_num
    ]

    S, cs = Williamson_form(C)
    Gs = G_Gauss(S, dC, cs)
    F = [
        tr(Gs[i] * dC[j]) + transpose(dR̄[i]) * inv(C) * dR̄[j] for i = 1:para_num,
        j = 1:para_num
    ]

    if para_num == 1
        return F[1, 1] |> real
    else
        return F |> real
    end
end
