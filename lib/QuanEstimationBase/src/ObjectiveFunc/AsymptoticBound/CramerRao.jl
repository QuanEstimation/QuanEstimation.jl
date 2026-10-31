using ChainRulesCore
const σ_x = [0.0 1.0; 1.0 0.0im]
const σ_y = [0.0 -1.0im; 1.0im 0.0]
const σ_z = [1.0 0.0im; 0.0 -1.0]

# =====================================================================
# Internal numerical core for the logarithmic derivatives.
#
# The public functions below keep their historical signatures; the
# bodies delegate to the solvers in this section, which implement the
# relative spectral floor, the guarded division semantics, compensated
# residual evaluation, one-step refinement and the diagnostic record.
# =====================================================================

"""
    SpectralSolver

Eigen-decomposition data of a density matrix together with the
relative spectral floor actually used.
"""
struct SpectralSolver
    ρh::Matrix{ComplexF64}
    λ::Vector{Float64}
    vec::Matrix{ComplexF64}
    λmax::Float64
    τ::Float64
    sep::Float64
    flags::Vector{Symbol}
end

function _spectral_solver(
    ρ::AbstractMatrix{<:Complex};
    spec::PrecisionSpec = PrecisionSpec(Float64),
)
    d = size(ρ, 1)
    h_err = opnorm(ρ - ρ')
    ρh = Matrix(Hermitian((ρ + ρ') / 2))
    F = eigen(Hermitian(ρh))
    λ = real.(F.values)
    vec = F.vectors
    λmax = maximum(abs, λ)
    τ = spectral_floor(spec, λmax)
    flags = _untracked() do
        f = Symbol[]
        if h_err > spec.tol * max(λmax, 1.0)
            push!(f, :hermiticity_corrected)
        end
        if any(<(0), λ)
            push!(f, :negative_eigenvalue)
        end
        f
    end
    sep = Inf
    @inbounds for j = 1:d
        for i = 1:d
            den = λ[i] + λ[j]
            if den > τ && den < sep
                sep = den
            end
        end
    end
    isfinite(sep) || (sep = max(τ, floatmin(Float64)))
    return SpectralSolver(ρh, λ, vec, λmax, τ, sep, flags)
end

"""
    SLDSolution

Result of the guarded SLD solve: operator (original and eigen bases),
residual and diagnostic record.  The spectral solver data is retained
for the adjoint (rrule) computation.
"""
struct SLDSolution
    L::Matrix{ComplexF64}
    L_eig::Matrix{ComplexF64}
    L_dropped::Matrix{ComplexF64}
    residual::Matrix{ComplexF64}
    diag::NumericalDiagnostics
    ss::SpectralSolver
end

"""
    _lyapunov_correct(ss, R)

Solve ``\\rho\\,\\delta L + \\delta L\\,\\rho = R`` in the eigenbasis of
`ss`, with the same relative spectral floor used for the SLD itself.
"""
function _lyapunov_correct(ss::SpectralSolver, R::AbstractMatrix{<:Complex})
    d = size(R, 1)
    Reig = ss.vec' * R * ss.vec
    Ceig = zeros(ComplexF64, d, d)
    @inbounds for j = 1:d
        for i = 1:d
            den = ss.λ[i] + ss.λ[j]
            if den > ss.τ
                Ceig[i, j] = Reig[i, j] / den
            end
        end
    end
    δ = ss.vec * Ceig * ss.vec'
    return Matrix(Hermitian((δ + δ') / 2))
end

"""
    _sld_solve(ρ, σ; spec)

Guarded solve of the symmetric-logarithmic-derivative equation
``(\\rho L + L\\rho)/2 = \\sigma``.

The kernel-`kernel` block is set to zero (minimum-norm convention), the
dropped contribution is booked as a truncation bias, the backward error
is measured with compensated arithmetic, and one step of refinement is
attempted when `spec.refine` is set.
"""
function _sld_solve(
    ρ::AbstractMatrix{<:Complex},
    σ::AbstractMatrix{<:Complex};
    spec::PrecisionSpec = PrecisionSpec(Float64),
)
    ss = _spectral_solver(ρ; spec = spec)
    d = size(ρ, 1)
    σh = Matrix(Hermitian((σ + σ') / 2))
    rhs_correction = opnorm(σ - σh)
    σeig = ss.vec' * σh * ss.vec
    Leig = zeros(ComplexF64, d, d)
    Ddrop = zeros(ComplexF64, d, d)
    nkept = 0
    kernel_mass = 0.0
    f_drop = 0.0
    @inbounds for j = 1:d
        for i = 1:d
            den = ss.λ[i] + ss.λ[j]
            if den > ss.τ
                Leig[i, j] = 2 * σeig[i, j] / den
                nkept += 1
            elseif den > 0
                Ddrop[i, j] = 2 * σeig[i, j] / den
                f_drop += 2 * abs2(σeig[i, j]) / den
            elseif abs(σeig[i, j]) > 0
                kernel_mass = max(kernel_mass, abs(σeig[i, j]))
                f_drop += 2 * abs2(σeig[i, j]) / max(ss.τ, floatmin(Float64))
            end
        end
    end
    L = ss.vec * Leig * ss.vec'
    L = Matrix(Hermitian((L + L') / 2))
    L_dropped = ss.vec * Ddrop * ss.vec'
    trunc_bound = max(opnorm(L_dropped), kernel_mass)
    residual = lyapunov_residual(ss.ρh, L, 2 .* σh)
    scale = 2 * opnorm(ss.ρh) * opnorm(L) + 2 * opnorm(σh) + floatmin(Float64)
    backward = opnorm(residual) / scale
    forward = opnorm(residual) / ss.sep
    layer = :float64
    flags = _untracked(() -> copy(ss.flags))
    if any(!, isfinite.(L))
        _push_flag!(flags, :nonfinite_solution)
    end
    if trunc_bound > spec.tol * max(opnorm(σh), 1.0)
        _push_flag!(flags, :truncation_nonnegligible)
    end
    if rhs_correction > spec.tol * max(opnorm(σh), 1.0)
        _push_flag!(flags, :hermiticity_corrected)
    end
    # divergence detector: unresolved kernel-sector data, or a dropped
    # Fisher contribution far above the data scale
    if kernel_mass > spec.tol * max(1.0, opnorm(σh))
        _push_flag!(flags, :rank_change_suspected)
    end
    if f_drop > max(1.0, opnorm(σh)) / max(spec.guard, spec.u)
        _push_flag!(flags, :rank_change_suspected)
    end
    if spec.refine && nkept > 0 && max(backward, forward) > spec.tol
        # Iterate until the correction reaches the rounding level of L.
        # The residual itself stalls at its own compensated floor and is
        # not monotone there, so it must not be used as the acceptance
        # test; the correction norm is the error proxy.
        correction_norm = Inf
        for _ = 1:10
            δL = _lyapunov_correct(ss, residual)
            correction_norm = opnorm(δL)
            L = Matrix(Hermitian((L + δL + (L + δL)') / 2))
            residual = lyapunov_residual(ss.ρh, L, 2 .* σh)
            backward = opnorm(residual) / scale
            forward = opnorm(residual) / ss.sep
            layer = :compensated
            correction_norm <= 4 * spec.u * max(opnorm(L), floatmin(Float64)) && break
        end
        if correction_norm > sqrt(spec.u) * max(opnorm(L), floatmin(Float64))
            _push_flag!(flags, :stagnated)
        end
    end
    positive = ss.λ[ss.λ .> 0]
    λmin⁺ = isempty(positive) ? ss.λmax : minimum(positive)
    κ = ss.λmax / max(λmin⁺, floatmin(Float64))
    input_prop =
        ss.sep > 0 ? (spec.input_error * (opnorm(σh) + opnorm(ss.ρh) * opnorm(L))) / ss.sep : Inf
    diag = NumericalDiagnostics(
        :SLD;
        precision_layer = layer,
        spectral_floor = ss.τ,
        numerical_rank = nkept,
        backward_error = backward,
        forward_error_bound = forward,
        input_error_propagation = input_prop,
        truncation_bias_bound = trunc_bound,
        condition_number = κ,
        branch = :generic,
        dropped_fisher_mass = f_drop,
        flags = flags,
        bottleneck = classify_bottleneck(;
            forward_error_bound = forward,
            input_error_propagation = input_prop,
            truncation_bias_bound = trunc_bound,
        ),
    )
    return SLDSolution(L, Leig, L_dropped, residual, diag, ss)
end

"""
    RLDSolution

Result of the guarded RLD solve with its diagnostic record.
"""
struct RLDSolution
    R::Matrix{ComplexF64}
    diag::NumericalDiagnostics
end

"""
    _rld_solve(ρ, σ; spec, label)

Guarded solve of the right-logarithmic-derivative equation
``\\sigma = \\rho R``.

When the support condition ``(I - \\rho\\rho^{+})\\sigma = 0`` is
violated the equation has no solution (class B); an `ArgumentError` is
thrown, reporting the parameter label and the violation magnitude.
"""
function _rld_solve(
    ρ::AbstractMatrix{<:Complex},
    σ::AbstractMatrix{<:Complex};
    spec::PrecisionSpec = PrecisionSpec(Float64),
    label::AbstractString = "the parameter",
)
    ss = _spectral_solver(ρ; spec = spec)
    d = size(ρ, 1)
    σeig = ss.vec' * σ * ss.vec
    Reig = zeros(ComplexF64, d, d)
    violation = 0.0
    @inbounds for j = 1:d
        for i = 1:d
            if ss.λ[i] > ss.τ
                Reig[i, j] = σeig[i, j] / ss.λ[i]
            else
                violation = max(violation, abs(σeig[i, j]))
            end
        end
    end
    σscale = max(opnorm(σ), floatmin(Float64))
    threshold = max(spec.guard, spec.input_error) * σscale
    if violation > threshold
        throw(
            ArgumentError(
                "The RLD/LLD does not exist for $label: the support of the derivative " *
                "is not contained in the support of the density matrix. " *
                "Violation magnitude = $(violation), threshold = $(threshold) " *
                "(σ scale = $(σscale), spectral floor = $(ss.τ)).",
            ),
        )
    end
    R = ss.vec * Reig * ss.vec'
    positive = ss.λ[ss.λ .> 0]
    λmin⁺ = isempty(positive) ? ss.λmax : minimum(positive)
    κ = ss.λmax / max(λmin⁺, floatmin(Float64))
    input_prop = spec.input_error * (opnorm(R) + σscale / max(λmin⁺, floatmin(Float64)))
    diag = NumericalDiagnostics(
        :RLD;
        precision_layer = :float64,
        spectral_floor = ss.τ,
        numerical_rank = count(>(ss.τ), ss.λ),
        input_error_propagation = input_prop,
        condition_number = κ,
        branch = :generic,
        flags = _untracked(() -> copy(ss.flags)),
        bottleneck = input_prop > 0 ? :input_data : :none,
    )
    return RLDSolution(R, diag)
end

"""
    _qfim_truncation_bound(ss, dρ)

Frobenius-type bound on the QFIM contribution of the dropped eigenblocks.
"""
function _qfim_truncation_bound(ss::SpectralSolver, dρ::AbstractVector)
    d = size(ss.λ, 1)
    p = length(dρ)
    σeig = [ss.vec' * Matrix(Hermitian((dρ[a] + dρ[a]') / 2)) * ss.vec for a = 1:p]
    bound = 0.0
    for a = 1:p, b = 1:p
        s = 0.0
        @inbounds for j = 1:d, i = 1:d
            den = ss.λ[i] + ss.λ[j]
            if den <= ss.τ
                den_eff = den > 0 ? den : ss.τ
                s += 2 * abs(σeig[a][i, j]) * abs(σeig[b][j, i]) / den_eff
            end
        end
        bound = max(bound, s)
    end
    return bound
end

"""
    _qfim_sigma_formula(ss, dρ)

Assemble the QFIM directly from the eigenvalue formula
``F_{ab} = 2\\sum (\\sigma_a)_{ij}(\\sigma_b)_{ji}/(\\lambda_i+\\lambda_j)``,
used as an independent cross-check of the SLD route.
"""
function _qfim_sigma_formula(ss::SpectralSolver, dρ::AbstractVector)
    d = size(ss.λ, 1)
    p = length(dρ)
    σeig = [ss.vec' * Matrix(Hermitian((dρ[a] + dρ[a]') / 2)) * ss.vec for a = 1:p]
    F = zeros(p, p)
    for a = 1:p, b = 1:p
        s = 0.0 + 0.0im
        @inbounds for j = 1:d, i = 1:d
            den = ss.λ[i] + ss.λ[j]
            if den > ss.τ
                s += 2 * σeig[a][i, j] * σeig[b][j, i] / den
            end
        end
        F[a, b] = real(s)
    end
    return Matrix(Hermitian((F + F') / 2))
end

"""
    _qfim_sld_solve(ρ, dρ; spec)

SLD-based QFIM with diagnostics.  Returns `(F, Ls, diag)`.
"""
function _qfim_sld_solve(
    ρ::AbstractMatrix{<:Complex},
    dρ::AbstractVector{<:AbstractMatrix{<:Complex}};
    spec::PrecisionSpec = PrecisionSpec(Float64),
)
    p = length(dρ)
    ss = _spectral_solver(ρ; spec = spec)
    sols = [_sld_solve(ρ, dρ[a]; spec = spec) for a = 1:p]
    Ls = [s.L for s in sols]
    F = zeros(p, p)
    for a = 1:p, b = 1:p
        F[a, b] = real(tr(0.5 * ss.ρh * (Ls[a] * Ls[b] + Ls[b] * Ls[a])))
    end
    F = Matrix(Hermitian((F + F') / 2))
    Fσ = _qfim_sigma_formula(ss, dρ)
    gap = opnorm(F - Fσ) / max(opnorm(Fσ), floatmin(Float64))
    trunc = _qfim_truncation_bound(ss, dρ)
    # input-data propagation (first order, both the σ and the ρ channel)
    input_prop = 0.0
    for a = 1:p
        for b = 1:p
            δσ = spec.input_error * opnorm(dρ[b])
            δρ = spec.input_error * opnorm(ρ)
            bound = 2 * opnorm(Ls[a]) * δσ
            bound +=
                δρ *
                (opnorm(Ls[a]) * opnorm(Ls[b]) + 2 * sqrt(abs(F[a, a])) * opnorm(Ls[b]) +
                 2 * sqrt(abs(F[b, b])) * opnorm(Ls[a]))
            input_prop = max(input_prop, bound)
        end
    end
    flags = _untracked() do
        f = Symbol[]
        all(:nonfinite_solution in s.diag.flags for s in sols) && push!(f, :nonfinite_solution)
        any(:rank_change_suspected in s.diag.flags for s in sols) &&
            push!(f, :rank_change_suspected)
        trunc > spec.tol * max(opnorm(F), 1.0) && push!(f, :truncation_nonnegligible)
        unique(f)
    end
    layer = all(s.diag.precision_layer === :compensated for s in sols) ? :compensated : :float64
    diag = NumericalDiagnostics(
        :QFIM;
        precision_layer = layer,
        spectral_floor = ss.τ,
        numerical_rank = sols[1].diag.numerical_rank,
        backward_error = maximum(s.diag.backward_error for s in sols),
        forward_error_bound = maximum(s.diag.forward_error_bound for s in sols),
        input_error_propagation = input_prop,
        truncation_bias_bound = trunc,
        condition_number = sols[1].diag.condition_number,
        cross_validation_gap = gap,
        dropped_fisher_mass = trunc,
        flags = flags,
        bottleneck = classify_bottleneck(;
            forward_error_bound = maximum(s.diag.forward_error_bound for s in sols),
            input_error_propagation = input_prop,
            truncation_bias_bound = trunc,
        ),
    )
    return F, Ls, diag
end

"""
    CFIMSolution

Result of the guarded CFIM evaluation.
"""
struct CFIMSolution
    F::Union{Float64,Matrix{Float64}}
    diag::NumericalDiagnostics
end

"""
    _cfim_solve(ρ, dρ, M; spec, upper_bound)

Guarded classical Fisher information.

Probabilities below the relative floor are removed (class C); the
returned value is then a lower bound of the true CFIM and the removed
probability mass is reported.  When `upper_bound` is set the QFIM is
computed as the information-theoretic upper bound and the gap is booked
in the diagnostics.
"""
function _cfim_solve(
    ρ::AbstractMatrix{<:Complex},
    dρ::AbstractVector{<:AbstractMatrix{<:Complex}},
    M;
    spec::PrecisionSpec = PrecisionSpec(Float64),
    upper_bound::Bool = false,
)
    p = length(dρ)
    m = length(M)
    prob = [real(tr(ρ * Π)) for Π in M]
    dprob = [[real(tr(dρ[a] * Π)) for a = 1:p] for Π in M]
    pmax = maximum(prob)
    τp = probability_floor(spec, pmax)
    kept = [py > τp for py in prob]
    nkept = count(kept)
    deleted_count = m - nkept
    deleted_mass = sum((max(py, 0.0) for (py, k) in zip(prob, kept) if !k); init = 0.0)
    terms = [(dprob[y] * dprob[y]') ./ prob[y] for y = 1:m if kept[y]]
    F = isempty(terms) ? zeros(p, p) : sum(terms)
    F = Matrix(Hermitian((F + F') / 2))
    flags = Symbol[]
    ub = NaN
    gap = NaN
    if deleted_count > 0
        _push_flag!(flags, :resolution_floor)
        if upper_bound
            Fq, _, dq = _untracked(() -> _qfim_sld_solve(ρ, dρ; spec = spec))
            if :rank_change_suspected in dq.flags || :nonfinite_solution in dq.flags
                _push_flag!(flags, :rank_change_suspected)
                ub = Inf
                gap = Inf
            else
                ub = opnorm(Fq)
                gap = ub - opnorm(F)
            end
        end
    end
    dpmax = maximum(maximum(abs, dp) for dp in dprob)
    input_prop =
        2 * (spec.input_error * dpmax) * max(dpmax, floatmin(Float64)) /
        max(pmax, floatmin(Float64)) +
        spec.input_error * opnorm(F) / max(pmax, floatmin(Float64))
    diag = NumericalDiagnostics(
        :CFIM;
        precision_layer = :float64,
        spectral_floor = τp,
        numerical_rank = nkept,
        truncation_bias_bound = deleted_mass,
        input_error_propagation = input_prop,
        upper_bound = ub,
        deleted_mass = deleted_mass,
        flags = flags,
        bottleneck = deleted_count > 0 ? :truncation : (input_prop > 0 ? :input_data : :none),
    )
    return CFIMSolution(F, diag)
end

############## logarrithmic derivative ###############
@doc raw"""

	SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=nothing,
	    tol=1e-10, input_error=0.0, guard=nothing, refine=true,
	    diagnostics=false) where {T<:Complex}

Calculate the symmetric logarithmic derivatives (SLDs). The SLD operator ``L_a`` is defined
as ``\partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho)``, where ``\rho`` is the parameterized density matrix.
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the SLD operator. Options are `"original"` (default) and `"eigen"`.
- `eps`: Deprecated relative spectral-floor factor. With the default `nothing` the relative precision policy is used; when supplied, `eps` is a relative floor, not machine epsilon.
- `tol`, `input_error`, `guard`, `refine`: Relative precision policy controls.
- `diagnostics`: If `true`, also return one [`NumericalDiagnostics`](@ref) record per parameter.

Returns a vector of SLD matrices, or `(Ls, diags)` when `diagnostics=true`.
"""
function SLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    rep = "original",
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    results = [
        SLD(
            ρ,
            x;
            rep = rep,
            eps = eps,
            tol = tol,
            input_error = input_error,
            guard = guard,
            refine = refine,
            diagnostics = diagnostics,
        ) for x in dρ
    ]
    if diagnostics
        return [r[1] for r in results], [r[2] for r in results]
    end
    return results
end

@doc raw"""

	SLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=nothing, tol=1e-10,
	    input_error=0.0, guard=nothing, refine=true, diagnostics=false) where {T<:Complex}

Calculate the SLD for a single parameter.

The spectral floor is *relative* to the largest eigenvalue. `eps` is a
deprecated relative spectral-floor factor: with the default `nothing` the
relative precision policy is used, and a supplied value is a relative floor,
not machine epsilon. With `diagnostics=true` the function returns
`(L, diag)` where `diag` is a [`NumericalDiagnostics`](@ref) record.
"""
function SLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    rep = "original",
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    if rep != "original" && rep != "eigen"
        throw(ArgumentError("The rep should be chosen in {'original', 'eigen'}."))
    end
    spec = precision_spec(
        eps;
        input_error = input_error,
        tol = tol,
        guard = guard,
        refine = refine,
        work_type = eltype(real(ρ)),
    )
    sol = _sld_solve(ρ, dρ; spec = spec)
    L = if rep == "original"
        sol.L
    else
        Matrix(Hermitian((sol.L_eig + sol.L_eig') / 2))
    end
    return diagnostics ? (L, sol.diag) : L
end

@doc raw"""
    ChainRulesCore.rrule(::typeof(SLD), ρ::Matrix{T}, dρ::Matrix{T}; eps=nothing)

Custom reverse-mode automatic differentiation (AD) rule for the SLD operator.

This rule enables backpropagation through `SLD(ρ, dρ)` in AD frameworks
(Zygote, etc.). The forward pass computes ``L = \mathrm{SLD}(\rho, \partial\rho)``.
The pullback maps the cotangent ``\bar{L}`` to cotangents w.r.t. ``\rho`` and
``\partial\rho``.

# Mathematical Definition

The pullback is the exact adjoint of the *truncated* (masked) forward map at
the current mask.  Let ``G`` solve ``(\rho G + G\rho)/2 = \bar{L}/2`` with the
same spectral floor, ``X = L + D`` the untruncated solution (``D`` the dropped
blocks, kept in the forward diagnostics), ``m`` the kept-channel mask, and
hats denote the eigenbasis of ``\rho``.  Then

```math
\bar{\rho} = -(G X + X G) + Q Z Q^\dagger,\qquad
\bar{\partial\rho} = 2G,
```

where the projector-rotation term ``Z`` accounts for the mask following the
eigenbasis of ``\rho``:

```math
Z_{ki} = \frac{\bigl([\hat{L}, \hat{\bar{L}}] + [m\circ\hat{\bar{L}}, \hat{X}]\bigr)_{ki}}
              {\lambda_k - \lambda_i}\quad (k\neq i),
\qquad Z_{kk}=0 .
```

The mask itself is locally constant; at a mask flip (a denominator crossing
the spectral floor) or at a rank-change point the two-sided derivative does
not exist and the pullback warns.

# Arguments

- `::typeof(SLD)`: The function signature for dispatch.
- `ρ::Matrix{T}`: Density matrix.
- `dρ::Matrix{T}`: Derivative of the density matrix.
- `eps::Float64=nothing`: legacy relative spectral floor (deprecated).

# Returns

- `Tuple{L, pullback}`: The SLD ``L`` and a pullback closure.

# See Also

- [`SLD`](@ref): Forward SLD computation.
raw"""
function ChainRulesCore.rrule(
    ::typeof(SLD),
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
) where {T<:Complex}
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, refine = refine)
    sol = _sld_solve(ρ, dρ; spec = spec)
    L = sol.L
    D = sol.L_dropped
    ss = sol.ss
    # Two-sided differentiability fails only when the kernel block itself is
    # fed (denominator at the rounding floor with non-vanishing right-hand
    # side) or the solution is non-finite; the f_drop-based rank-change
    # suspicion stays smooth and is reported by the trunc_dominant branch.
    dρh = (dρ + dρ') / 2
    σe = ss.vec' * dρh * ss.vec
    ker_floor = 10 * size(ρ, 1) * spec.u * ss.λmax
    km = 0.0
    @inbounds for j in axes(ss.λ, 1), i in axes(ss.λ, 1)
        if ss.λ[i] + ss.λ[j] <= ker_floor && abs(σe[i, j]) > km
            km = abs(σe[i, j])
        end
    end
    ill_posed =
        km > spec.tol * max(1.0, opnorm(dρh)) ||
        (:nonfinite_solution in sol.diag.flags)
    trunc_dominant =
        sol.diag.dropped_fisher_mass > 0.1 * max(abs(real(tr(ρ * L * L))), 1.0) ||
        (:rank_change_suspected in sol.diag.flags)
    # A channel flips across the floor when an input-scale perturbation moves
    # its denominator past τ; the fixed-mask adjoint is then only one-sided.
    margin = max(spec.input_error, size(ρ, 1) * spec.u) * ss.λmax
    mask_unstable = false
    @inbounds for j in eachindex(ss.λ), i in eachindex(ss.λ)
        den = ss.λ[i] + ss.λ[j]
        if den > 0 && abs(den - ss.τ) <= margin
            mask_unstable = true
            break
        end
    end
    has_dropped = any(!iszero, D)
    # The projector-rotation term divides by eigenvalue gaps; below the
    # degeneracy floor the quotient leaves the working-precision floor
    # (eigenvector noise ~u/gap amplified by another 1/gap), so the
    # contribution is suppressed pair-wise and reported once.
    deg = max(sqrt(spec.u), spec.input_error) * ss.λmax
    function SLD_pullback(L̄)
        if ill_posed
            @warn "SLD pullback: the forward solve detected a rank-change/nonfinite " *
                  "case (kernel block with non-vanishing right-hand side), where the " *
                  "two-sided derivative does not exist. Returning the fixed-mask " *
                  "adjoint of the truncated map; check `diag.flags` before using " *
                  "this gradient." maxlog = 1
        elseif trunc_dominant || mask_unstable
            @warn "SLD pullback: the gradient is the exact adjoint of the truncated " *
                  "(masked) map at the current mask, but the forward value is " *
                  "truncation-dominated (dropped Fisher mass above the data scale; " *
                  "a rank-change pattern is suspected) or a channel is within the " *
                  "input-error scale of flipping across the spectral floor; " *
                  "interpret this gradient accordingly." maxlog = 1
        end
        Lb = Matrix{ComplexF64}(L̄)
        Ḡ = SLD(
            Array(ρ),
            Lb / 2;
            eps = eps,
            tol = tol,
            input_error = input_error,
            guard = guard,
            refine = refine,
        )
        X = L + D
        ρ̄ = -Ḡ * X - X * Ḡ
        if has_dropped
            X̂ = ss.vec' * X * ss.vec
            L̂ = ss.vec' * L * ss.vec
            Ŵ = ss.vec' * ((Lb + Lb') / 2) * ss.vec
            m = [ss.λ[i] + ss.λ[j] > ss.τ ? 1.0 : 0.0 for i in axes(ss.λ, 1), j in axes(ss.λ, 1)]
            mŴ = m .* Ŵ
            A = L̂ * Ŵ - Ŵ * L̂ + mŴ * X̂ - X̂ * mŴ
            Λd = ss.λ .- ss.λ'
            supp = zero(ComplexF64)
            Zm = zeros(ComplexF64, size(A))
            @inbounds for k in axes(A, 1), i in axes(A, 2)
                if k != i && abs(Λd[k, i]) > deg
                    Zm[k, i] = A[k, i] / Λd[k, i]
                elseif k != i && abs(A[k, i]) > abs(supp)
                    supp = A[k, i]
                end
            end
            ρ̄ += ss.vec * Zm * ss.vec'
            if !ill_posed && abs(supp) > spec.tol * max(opnorm(ρ̄), 1.0)
                @warn "SLD pullback: the spectrum has near-degenerate eigenvalues " *
                      "(gap below sqrt(u)·λmax); the projector-rotation part of the " *
                      "truncated adjoint is suppressed on those pairs, so the gradient " *
                      "may miss a bounded contribution. Use a higher-precision backend " *
                      "for the exact adjoint in this regime." maxlog = 1
            end
        end
        return ChainRulesCore.NoTangent(), ρ̄, 2 * Ḡ
    end
    return L, SLD_pullback
end

@doc raw"""
    SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}; eps=nothing) where {T<:Complex}

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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.

# Returns

- `Matrix{ComplexF64}`: The SLD operator ``L`` reshaped from Liouville vector.

# See Also

- [`SLD`](@ref): Standard SLD via eigenbasis decomposition.
- [`SLD_qr`](@ref): Liouville-space SLD via QR decomposition.
"""
function SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}; eps = nothing) where {T<:Complex}
    rtol = eps === nothing ? default_guard(Float64) : max(Float64(eps), default_guard(Float64))
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), rtol = rtol) * vec(∂ρ_∂x) |> vec2mat
end

"""
    SLD_liouville(ρ::Vector{T}, ∂ρ_∂x::Vector{T}; eps=nothing) where {T<:Complex}

Vector-input convenience wrapper that reshapes vectorized density matrices
to matrix form and calls [`SLD_liouville`](@ref).
"""
function SLD_liouville(ρ::Vector{T}, ∂ρ_∂x::Vector{T}; eps = nothing) where {T<:Complex}
    SLD_liouville(ρ |> vec2mat, ∂ρ_∂x |> vec2mat; eps = eps)
end

raw"""
    SLD_liouville(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}; eps=nothing) where {T<:Complex}

Multi-parameter convenience wrapper that broadcasts the single-parameter
[`SLD_liouville`](@ref) over each derivative matrix.
raw"""
function SLD_liouville(
    ρ::Matrix{T},
    ∂ρ_∂x::Vector{Matrix{T}};
    eps = nothing,
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

    RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, diagnostics=false) where {T<:Complex}

Calculate the right logarithmic derivatives (RLDs). The RLD operator is defined as
``\partial_{a}\rho=\rho \mathcal{R}_a``, where ``\rho`` is the parameterized density matrix.
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the RLD operator. Options are `"original"` (default) and `"eigen"`.
- `eps`: Deprecated relative spectral-floor factor. With the default `nothing` the relative precision policy is used; when supplied, `eps` is a relative floor, not machine epsilon.
- `tol`, `input_error`, `guard`: Relative precision policy controls.
- `diagnostics`: If `true`, also return one [`NumericalDiagnostics`](@ref) record per parameter.

Returns a vector of RLD matrices, or `(Rs, diags)` when `diagnostics=true`.
"""
function RLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    rep = "original",
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {T<:Complex}
    results = [
        RLD(
            ρ,
            x;
            rep = rep,
            eps = eps,
            tol = tol,
            input_error = input_error,
            guard = guard,
            diagnostics = diagnostics,
        ) for x in dρ
    ]
    if diagnostics
        return [r[1] for r in results], [r[2] for r in results]
    end
    return results
end

@doc raw"""

	RLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=nothing, tol=1e-10,
	    input_error=0.0, guard=nothing, diagnostics=false) where {T<:Complex}

Calculate the RLD for a single parameter.

`eps` is a deprecated relative spectral-floor factor: with the default
`nothing` the relative precision policy is used, and a supplied value is a
relative floor, not machine epsilon. The support condition
``(I-\rho\rho^{+})\sigma=0`` is checked with a relative tolerance; when it is
violated the equation has no solution and an `ArgumentError` is thrown.
With `diagnostics=true` the function returns `(R, diag)`.
"""
function RLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    rep = "original",
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {T<:Complex}
    if rep != "original" && rep != "eigen"
        throw(ArgumentError("The rep should be chosen in {'original', 'eigen'}."))
    end
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, work_type = eltype(real(ρ)))
    sol = _rld_solve(ρ, dρ; spec = spec, label = "the parameter")
    R = sol.R
    if rep == "eigen"
        ss = _spectral_solver(ρ; spec = spec)
        R = ss.vec' * R * ss.vec
    end
    return diagnostics ? (R, sol.diag) : R
end


@doc raw"""

    LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; rep="original", eps=nothing, tol=1e-10,
        input_error=0.0, guard=nothing, diagnostics=false) where {T<:Complex}

Calculate the left logarithmic derivatives (LLDs). The LLD operator is defined as ``\partial_{a}\rho=\mathcal{L}_a\rho``,
where ``\mathcal{L}_a = \mathcal{R}_a^\dagger`` and ``\mathcal{R}_a`` is the RLD operator.
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `rep`: Representation of the LLD operator. Options are `"original"` (default) and `"eigen"`.
- `eps`: Deprecated relative spectral-floor factor. With the default `nothing` the relative precision policy is used; when supplied, `eps` is a relative floor, not machine epsilon.
- `tol`, `input_error`, `guard`: Relative precision policy controls.
- `diagnostics`: If `true`, also return one [`NumericalDiagnostics`](@ref) record per parameter.

Returns a vector of LLD matrices, or `(Ls, diags)` when `diagnostics=true`.
"""
function LLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    rep = "original",
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {T<:Complex}
    results = [
        LLD(
            ρ,
            x;
            rep = rep,
            eps = eps,
            tol = tol,
            input_error = input_error,
            guard = guard,
            diagnostics = diagnostics,
        ) for x in dρ
    ]
    if diagnostics
        return [r[1] for r in results], [r[2] for r in results]
    end
    return results
end

@doc raw"""
    LLD(ρ::Matrix{T}, dρ::Matrix{T}; rep="original", eps=nothing, tol=1e-10,
        input_error=0.0, guard=nothing, diagnostics=false) where {T<:Complex}

Calculate the LLD for a single parameter.

`eps` is a deprecated relative spectral-floor factor: with the default
`nothing` the relative precision policy is used, and a supplied value is a
relative floor, not machine epsilon. With `diagnostics=true` the function
returns `(L, diag)`.
"""
function LLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    rep = "original",
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {T<:Complex}
    if diagnostics
        R, diag = RLD(
            ρ,
            dρ;
            rep = rep,
            eps = eps,
            tol = tol,
            input_error = input_error,
            guard = guard,
            diagnostics = true,
        )
        return R', diag
    end
    return RLD(ρ, dρ; rep = rep, eps = eps, tol = tol, input_error = input_error, guard = guard)'
end


#========================================================#
####################### calculate QFI ####################
@doc raw"""
    QFIM_SLD(ρ::Matrix{T}, dρ::Matrix{T}; eps=nothing) where {T<:Complex}

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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.

# Returns

- `Float64`: The real part of ``\mathrm{Tr}(\rho L_x^2)``.

# See Also

- [`SLD`](@ref): Symmetric logarithmic derivative operator.
- [`QFIM_SLD`](@ref) (multi-parameter): Multi-parameter SLD-based QFIM.
raw"""
function QFIM_SLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    spec = precision_spec(
        eps;
        input_error = input_error,
        tol = tol,
        guard = guard,
        refine = refine,
        work_type = eltype(real(ρ)),
    )
    if diagnostics
        F, diag = _untracked() do
            sol = _sld_solve(ρ, dρ; spec = spec)
            Fv = real(tr(ρ * sol.L * sol.L))
            fdrop = sol.diag.dropped_fisher_mass
            fl = [f for f in sol.diag.flags if f !== :truncation_nonnegligible]
            if fdrop > spec.tol * max(abs(Fv), 1.0)
                push!(fl, :truncation_nonnegligible)
            end
            d = NumericalDiagnostics(
                :QFIM;
                precision_layer = sol.diag.precision_layer,
                spectral_floor = sol.diag.spectral_floor,
                numerical_rank = sol.diag.numerical_rank,
                backward_error = sol.diag.backward_error,
                forward_error_bound = sol.diag.forward_error_bound,
                input_error_propagation = sol.diag.input_error_propagation,
                truncation_bias_bound = fdrop,
                condition_number = sol.diag.condition_number,
                dropped_fisher_mass = fdrop,
                flags = fl,
                bottleneck = classify_bottleneck(;
                    forward_error_bound = sol.diag.forward_error_bound,
                    input_error_propagation = sol.diag.input_error_propagation,
                    truncation_bias_bound = fdrop,
                ),
            )
            (Fv, d)
        end
        return F, diag
    end
    L = SLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, refine = refine)
    F = real(tr(ρ * L * L))
    return F
end

@doc raw"""
    QFIM_RLD(ρ::Matrix{T}, dρ::Matrix{T}; eps=nothing) where {T<:Complex}

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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.

# Returns

- `Float64`: The real part of ``\mathrm{Tr}(\rho\mathcal{R}_x\mathcal{R}_x^\dagger)``.

# See Also

- [`RLD`](@ref): Right logarithmic derivative operator.
- [`QFIM_RLD`](@ref) (multi-parameter): Multi-parameter RLD-based QFIM.
- [`QFIM_LLD`](@ref): LLD-based QFI (equivalent to RLD-based QFI).
raw"""
function QFIM_RLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, refine = refine, work_type = eltype(real(ρ)))
    sol = _rld_solve(ρ, dρ; spec = spec, label = "the parameter")
    F = real(tr(ρ * sol.R * sol.R'))
    return diagnostics ? (F, sol.diag) : F
end

@doc raw"""
    QFIM_LLD(ρ::Matrix{T}, dρ::Matrix{T}; eps=nothing) where {T<:Complex}

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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.

# Returns

- `Float64`: The real part of ``\mathrm{Tr}(\rho\mathcal{L}_x^\dagger\mathcal{L}_x)``.

# See Also

- [`LLD`](@ref): Left logarithmic derivative operator.
- [`QFIM_LLD`](@ref) (multi-parameter): Multi-parameter LLD-based QFIM.
- [`QFIM_RLD`](@ref): RLD-based QFI (equivalent to LLD-based QFI).
raw"""
function QFIM_LLD(
    ρ::Matrix{T},
    dρ::Matrix{T};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, refine = refine, work_type = eltype(real(ρ)))
    sol = _rld_solve(ρ, dρ; spec = spec, label = "the parameter")
    F = real(tr(ρ * sol.R * sol.R'))
    return diagnostics ? (F, sol.diag) : F
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
    QFIM_SLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps=nothing) where {T<:Complex}

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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.

# Returns

- `Matrix{Float64}`: The ``p\times p`` QFIM with entries ``F_{ab}``.

# See Also

- [`SLD`](@ref): Symmetric logarithmic derivative operator.
- [`QFIM_SLD`](@ref) (single-parameter): Single-parameter SLD-based QFI.
raw"""
function QFIM_SLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    spec = precision_spec(
        eps;
        input_error = input_error,
        tol = tol,
        guard = guard,
        refine = refine,
        work_type = eltype(real(ρ)),
    )
    if diagnostics
        F, _, diag = _untracked(() -> _qfim_sld_solve(ρ, dρ; spec = spec))
        return F, diag
    end
    p_num = length(dρ)
    LD_tp = [SLD(ρ, x; eps = eps, tol = tol, input_error = input_error, guard = guard, refine = refine) for x in dρ]
    return [real(tr(0.5 * ρ * (LD_tp[i] * LD_tp[j] + LD_tp[j] * LD_tp[i]))) for i = 1:p_num, j = 1:p_num]
end

@doc raw"""
    QFIM_RLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps=nothing) where {T<:Complex}

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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.

# Returns

- `Matrix{ComplexF64}`: The ``p\times p`` RLD QFIM (complex-Hermitian).

# See Also

- [`RLD`](@ref): Right logarithmic derivative operator.
- [`QFIM_RLD`](@ref) (single-parameter): Single-parameter RLD-based QFI.
- [`QFIM_LLD`](@ref): LLD-based QFIM (``F_{ab}^{\mathrm{LLD}}=\mathrm{Tr}(\rho\mathcal{L}_a^\dagger\mathcal{L}_b)=\mathrm{Tr}(\rho\mathcal{R}_a\mathcal{R}_b^\dagger)``).
raw"""
function QFIM_RLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    p_num = length(dρ)
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, refine = refine, work_type = eltype(real(ρ)))
    Rs = [_rld_solve(ρ, dρ[i]; spec = spec, label = "parameter $i").R for i = 1:p_num]
    F = [tr(ρ * Rs[i] * Rs[j]') for i = 1:p_num, j = 1:p_num]
    diag = NumericalDiagnostics(
        :QFIM;
        branch = :RLD,
        spectral_floor = spectral_floor(spec, maximum(abs, eigvals(Hermitian((ρ + ρ') / 2)))),
    )
    return diagnostics ? (F, diag) : F
end

@doc raw"""
    QFIM_LLD(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; eps=nothing) where {T<:Complex}

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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.

# Returns

- `Matrix{ComplexF64}`: The ``p\times p`` LLD QFIM (complex-Hermitian).

# See Also

- [`LLD`](@ref): Left logarithmic derivative operator.
- [`QFIM_LLD`](@ref) (single-parameter): Single-parameter LLD-based QFI.
- [`QFIM_RLD`](@ref): RLD-based QFIM (identical to LLD-based QFIM).
raw"""
function QFIM_LLD(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    p_num = length(dρ)
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, refine = refine, work_type = eltype(real(ρ)))
    Rs = [_rld_solve(ρ, dρ[i]; spec = spec, label = "parameter $i").R for i = 1:p_num]
    F = [tr(ρ * Rs[i] * Rs[j]') for i = 1:p_num, j = 1:p_num]
    diag = NumericalDiagnostics(
        :QFIM;
        branch = :LLD,
        spectral_floor = spectral_floor(spec, maximum(abs, eigvals(Hermitian((ρ + ρ') / 2)))),
    )
    return diagnostics ? (F, diag) : F
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

	CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}, M; eps=nothing) where {T<:Complex}

Calculate the classical Fisher information matrix (CFIM). 
- `ρ`: Density matrix.
- `dρ`: Derivatives of the density matrix with respect to the unknown parameters to be estimated. For example, drho[1] is the derivative vector with respect to the first parameter.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.
"""
function CFIM(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}},
    M;
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {T<:Complex}
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, work_type = eltype(real(ρ)))
    sol = _cfim_solve(ρ, dρ, M; spec = spec, upper_bound = diagnostics)
    return diagnostics ? (sol.F, sol.diag) : sol.F
end

"""

	CFIM(ρ::Matrix{T}, dρ::Matrix{T}, M; eps=nothing) where {T<:Complex}

When applied to the case of single parameter. Calculate the classical Fisher information (CFI). 
"""
function CFIM(
    ρ::Matrix{T},
    dρ::Matrix{T},
    M;
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {T<:Complex}
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, work_type = eltype(real(ρ)))
    sol = _cfim_solve(ρ, [dρ], M; spec = spec, upper_bound = diagnostics)
    F = sol.F[1, 1]
    return diagnostics ? (F, sol.diag) : F
end

"""

	CFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; M=nothing, eps=nothing) where {T<:Complex}

When the set of POVM is not given. Calculate the CFIM with SIC-POVM. The SIC-POVM is generated from the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).
"""
function CFIM(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    M = nothing,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {T<:Complex}
    M = M === nothing ? SIC(size(ρ)[1]) : M
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, work_type = eltype(real(ρ)))
    sol = _cfim_solve(ρ, dρ, M; spec = spec, upper_bound = diagnostics)
    return diagnostics ? (sol.F, sol.diag) : sol.F
end

"""

	CFIM(ρ::Matrix{T}, dρ::Matrix{T}; eps=nothing) where {T<:Complex}

When applied to the case of single parameter and the set of POVM is not given. Calculate the CFI with SIC-POVM. 
"""
function CFIM(
    ρ::Matrix{T},
    dρ::Matrix{T};
    M = nothing,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {T<:Complex}
    M = M === nothing ? SIC(size(ρ)[1]) : M
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, work_type = eltype(real(ρ)))
    sol = _cfim_solve(ρ, [dρ], M; spec = spec, upper_bound = diagnostics)
    F = sol.F[1, 1]
    return diagnostics ? (F, sol.diag) : F
end

@doc raw"""
    CFIM(scheme::Scheme; full_trajectory=false, LDtype=:SLD, exportLD=false, eps=nothing)

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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.
- `tol`, `input_error`, `guard`, `diagnostics`: relative precision policy and
  diagnostic record, as in the matrix-level [`CFIM`](@ref).

# Returns

- `Matrix{Float64}` or `Vector{Matrix{Float64}}`: The CFIM, a 1×1 matrix for a
  single parameter (`F[1,1]` is the scalar CFI). If `full_trajectory=true`,
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
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
)
    M = meas_data(scheme)
    if full_trajectory
        rho, drho = expm(scheme)
        return [
            CFIM(
                r,
                dr,
                M;
                eps = eps,
                tol = tol,
                input_error = input_error,
                guard = guard,
                diagnostics = diagnostics,
            ) for (r, dr) in zip(rho, drho)
        ]
    else
        rho, drho = evolve(scheme)
        return CFIM(
            rho,
            drho,
            M;
            eps = eps,
            tol = tol,
            input_error = input_error,
            guard = guard,
            diagnostics = diagnostics,
        )
    end
end

## QFI with exportLD
"""

    QFIM(ρ::Matrix{T}, dρ::Matrix{T}; LDtype=:SLD, exportLD::Bool=false,
         eps=nothing, tol=1e-10, input_error=0.0, guard=nothing, refine=true,
         diagnostics=false) where {T<:Complex}

Calculate the quantum Fisher information matrix (QFIM) for a single parameter.
- `ρ`: Density matrix.
- `dρ`: Derivative of the density matrix with respect to the unknown parameter.
- `LDtype`: Type of QFIM: `:SLD` (default), `:RLD`, or `:LLD`.
- `exportLD`: If `true`, also return the logarithmic derivatives.
- `eps`: Deprecated relative spectral-floor factor. With the default `nothing` the relative precision policy is used; when supplied, `eps` is a relative floor, not machine epsilon.
- `tol`, `input_error`, `guard`: Relative precision policy controls.
- `refine`: Whether to refine the SLD solution; only used for `LDtype=:SLD`.
- `diagnostics`: If `true`, also return a [`NumericalDiagnostics`](@ref) record.

Returns the scalar QFI `F`, or `(F, LD)` when `exportLD=true`, or `(F, diag)`
when `diagnostics=true`, or `(F, LD, diag)` when both are `true`.
"""
function QFIM(
    ρ::Matrix{T},
    dρ::Matrix{T};
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    if LDtype == :SLD
        res = QFIM_SLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, refine = refine, diagnostics = diagnostics)
    elseif LDtype == :RLD
        res = QFIM_RLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, diagnostics = diagnostics)
    elseif LDtype == :LLD
        res = QFIM_LLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, diagnostics = diagnostics)
    else
        throw(ArgumentError("LDtype must be :SLD, :RLD, or :LLD"))
    end
    F = diagnostics ? res[1] : res
    diag = diagnostics ? res[2] : nothing
    if exportLD
        if LDtype == :SLD
            LD = SLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, refine = refine)
        elseif LDtype == :RLD
            LD = RLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard)
        else
            LD = LLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard)
        end
        return diagnostics ? (F, LD, diag) : (F, LD)
    end
    return diagnostics ? (F, diag) : F
end

"""

    QFIM(ρ::Matrix{T}, dρ::Vector{Matrix{T}}; LDtype=:SLD, exportLD::Bool=false,
         eps=nothing, tol=1e-10, input_error=0.0, guard=nothing, refine=true,
         diagnostics=false) where {T<:Complex}

Calculate the quantum Fisher information matrix (QFIM) for multiple parameters.
`dρ[i]` is the derivative with respect to parameter `i`.

`LDtype` is `:SLD` (default), `:RLD`, or `:LLD`. `eps` is a deprecated relative
spectral-floor factor: with the default `nothing` the relative precision policy
is used; when supplied, `eps` is a relative floor, not machine epsilon. `refine`
applies only to the SLD branch. Returns an `n × n` matrix, or `(F, LD)` when
`exportLD=true`, or `(F, diag)` when `diagnostics=true`, or `(F, LD, diag)` when
both are `true`.
"""
function QFIM(
    ρ::Matrix{T},
    dρ::Vector{Matrix{T}};
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
) where {T<:Complex}
    if LDtype == :SLD
        res = QFIM_SLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, refine = refine, diagnostics = diagnostics)
    elseif LDtype == :RLD
        res = QFIM_RLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, diagnostics = diagnostics)
    elseif LDtype == :LLD
        res = QFIM_LLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, diagnostics = diagnostics)
    else
        throw(ArgumentError("The LDtype should be chosen in {'SLD', 'RLD', 'LLD'}."))
    end
    F = diagnostics ? res[1] : res
    diag = diagnostics ? res[2] : nothing
    if exportLD
        if LDtype == :SLD
            LD = SLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard, refine = refine)
        elseif LDtype == :RLD
            LD = RLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard)
        else
            LD = LLD(ρ, dρ; eps = eps, tol = tol, input_error = input_error, guard = guard)
        end
        return diagnostics ? (F, LD, diag) : (F, LD)
    end
    return diagnostics ? (F, diag) : F
end

@doc raw"""
    QFIM(scheme::Scheme; full_trajectory=false, LDtype=:SLD, exportLD=false,
         eps=nothing, tol=1e-10, input_error=0.0, guard=nothing, refine=true,
         diagnostics=false)

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
- `eps`: Deprecated relative spectral-floor factor. With the default `nothing`
  the relative precision policy is used; when supplied, `eps` is a relative
  floor, not machine epsilon.
- `tol`, `input_error`, `guard`: Relative precision policy controls.
- `refine`: Whether to refine the SLD solution; only used for `LDtype=:SLD`.
- `diagnostics`: If `true`, also return a [`NumericalDiagnostics`](@ref) record.

# Returns

- `Matrix{Float64}` or `Vector{Matrix{Float64}}`: The QFIM; `full_trajectory=true`
  returns one matrix per time step. If `exportLD=true`, returns `(F, LD)`; if
  `diagnostics=true`, returns `(F, diag)`; if both are `true`, returns
  `(F, LD, diag)`.

# See Also

- [`QFIM`](@ref): Direct matrix QFIM computation.
- [`CFIM`](@ref): Classical Fisher information from a Scheme.
"""
function QFIM(
    scheme::Scheme;
    full_trajectory = false,
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    if full_trajectory
        rho, drho = expm(scheme)
        return [
            QFIM(
                r,
                dr;
                LDtype = LDtype,
                exportLD = exportLD,
                eps = eps,
                tol = tol,
                input_error = input_error,
                guard = guard,
                refine = refine,
                diagnostics = diagnostics,
            ) for (r, dr) in zip(rho, drho)
        ]
    else
        rho, drho = evolve(scheme)
        return QFIM(
            rho,
            drho;
            LDtype = LDtype,
            exportLD = exportLD,
            eps = eps,
            tol = tol,
            input_error = input_error,
            guard = guard,
            refine = refine,
            diagnostics = diagnostics,
        )
    end
end



"""

    QFIM_Kraus(ρ0::AbstractMatrix, K::AbstractVector, dK::AbstractVector;
               LDtype=:SLD, exportLD::Bool=false, eps=nothing, tol=1e-10,
               input_error=0.0, guard=nothing, refine=true, diagnostics=false)

Calculate the quantum Fisher information with Kraus operators.
- `ρ0`: Initial density matrix.
- `K`: Kraus operators.
- `dK`: Derivatives of the Kraus operators with respect to the unknown parameters. `dK[i][j]` is the derivative of `K[j]` with respect to parameter `i`.
- `LDtype`: Type of QFIM: `:SLD` (default), `:RLD`, or `:LLD`.
- `exportLD`: If `true`, also return the logarithmic derivatives.
- `eps`: Deprecated relative spectral-floor factor. With the default `nothing` the relative precision policy is used; when supplied, `eps` is a relative floor, not machine epsilon.
- `tol`, `input_error`, `guard`: Relative precision policy controls.
- `refine`: Whether to refine the SLD solution; only used for `LDtype=:SLD`.
- `diagnostics`: If `true`, also return a [`NumericalDiagnostics`](@ref) record.

Returns a scalar for one parameter or an `n × n` matrix for several, optionally
followed by the logarithmic derivatives and/or diagnostics.
"""
function QFIM_Kraus(
    ρ0::AbstractMatrix,
    K::AbstractVector,
    dK::AbstractVector;
    LDtype = :SLD,
    exportLD::Bool = false,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    para_num = length(dK[1])
    dK = [[dK[i][j] for i in eachindex(K)] for j = 1:para_num]
    ρ = [K * ρ0 * K' for K in K] |> sum
    dρ = [[dK * ρ0 * K' + K * ρ0 * dK' for (K, dK) in zip(K, dK)] |> sum for dK in dK]
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, refine = refine, work_type = eltype(real(ρ)))
    ρh, trace_correction, hermiticity_correction = hermitize_and_normalize(ρ; spec = spec)
    dρh = [Matrix(Hermitian((x + x') / 2)) for x in dρ]
    effective_input = max(input_error, trace_correction, hermiticity_correction)
    result = QFIM(
        ρh,
        dρh;
        LDtype = LDtype,
        exportLD = exportLD,
        eps = nothing,
        tol = tol,
        input_error = effective_input,
        guard = guard,
        refine = refine,
        diagnostics = diagnostics,
    )
    if diagnostics
        F = result[1]
        value = para_num == 1 ? F[1, 1] : F
        return (value, result[2:end]...)
    elseif exportLD
        F, LD = result
        return (para_num == 1 ? F[1, 1] : F), LD
    else
        F = result
        return para_num == 1 ? F[1, 1] : F
    end
end

"""

	QFIM_Bloch(r, dr; eps=nothing, tol=1e-10, input_error=0.0, guard=nothing,
	            diagnostics=false)

Calculate the SLD-based quantum Fisher information in Bloch representation.
- `r`: Parameterized Bloch vector.
- `dr`: Derivative(s) of the Bloch vector with respect to the unknown parameters. For example, dr[1] is the derivative vector with respect to the first parameter.
- `eps`: Deprecated relative spectral-floor factor. With the default `nothing` the relative precision policy is used; when supplied, `eps` is a relative floor, not machine epsilon.
- `tol`, `input_error`, `guard`: Relative precision policy controls.
- `diagnostics`: If `true`, also return a [`NumericalDiagnostics`](@ref) record.

Returns a scalar for one parameter or an `n × n` matrix for several.
"""
function QFIM_Bloch(
    r,
    dr;
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
)
    para_num = length(dr)
    QFIM_res = zeros(para_num, para_num)

    dim = Int(sqrt(length(r) + 1))
    Lambda = suN_generator(dim)
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, work_type = eltype(r))
    flags = Symbol[]
    branch = :mixed
    gap = -Inf
    floor_used = max(spec.guard, spec.input_error)
    rank = para_num
    if dim == 2
        ρ = Matrix(Hermitian((Matrix(I, 2, 2) + sqrt(1.0) * r' * Lambda) / 2))
        ε = max(2 * (1 - real(tr(ρ * ρ))), 0.0)
        τ = floor_used
        if ε > τ
            for para_i = 1:para_num
                for para_j = para_i:para_num
                    mixed = real(
                        dr[para_i]' * dr[para_j] +
                        (r' * dr[para_i]) * (r' * dr[para_j]) / ε,
                    )
                    pure = real(dr[para_i]' * dr[para_j])
                    QFIM_res[para_i, para_j] = mixed
                    QFIM_res[para_j, para_i] = mixed
                    gap = max(gap, abs(mixed - pure))
                end
            end
            if ε < 100 * τ
                _push_flag!(flags, :rank_change_suspected)
            end
        else
            branch = :pure
            _push_flag!(flags, :pure_limit)
            _push_flag!(flags, :rank_change_suspected)
            for para_i = 1:para_num
                for para_j = para_i:para_num
                    pure = real(dr[para_i]' * dr[para_j])
                    QFIM_res[para_i, para_j] = pure
                    QFIM_res[para_j, para_i] = pure
                    if ε > 0
                        mixed = real(
                            dr[para_i]' * dr[para_j] +
                            (r' * dr[para_i]) * (r' * dr[para_j]) / ε,
                        )
                        gap = max(gap, abs(mixed - pure))
                    end
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

        mat_tp = Hermitian(G * dim / (2 * (dim - 1)) - r * r')
        e = eigen(mat_tp)
        vals = real.(e.values)
        smax = maximum(abs, vals)
        τ = spectral_floor(spec, smax)
        floor_used = τ
        kept = findall(>(τ), abs.(vals))
        rank = length(kept)
        if rank < length(vals)
            _push_flag!(flags, :rank_change_suspected)
        end
        mat_inv = zeros(ComplexF64, size(mat_tp))
        for k in kept
            v = e.vectors[:, k]
            mat_inv .+= (v * v') ./ vals[k]
        end
        mat_inv = Matrix(Hermitian((mat_inv + mat_inv') / 2))

        for para_m = 1:para_num
            for para_n = para_m:para_num
                QFIM_res[para_m, para_n] = real(dr[para_n]' * mat_inv * dr[para_m])
                QFIM_res[para_n, para_m] = QFIM_res[para_m, para_n]
            end
        end
    end
    diag = NumericalDiagnostics(
        :QFIM;
        branch = branch,
        spectral_floor = floor_used,
        numerical_rank = rank,
        cross_validation_gap = isfinite(gap) ? gap : NaN,
        flags = flags,
    )
    value = para_num == 1 ? QFIM_res[1, 1] : QFIM_res
    return diagnostics ? (value, diag) : value
end

"""

    FIM(p::Vector{R}, dp::Vector{R}; eps=nothing, tol=1e-10,
        input_error=0.0, guard=nothing, diagnostics=false) where {R<:Real}

Calculate the classical Fisher information for a single parameter from a
probability distribution `p` and its derivative `dp`.
- `eps`: Deprecated relative probability-floor factor. With the default `nothing` the relative precision policy is used; when supplied, `eps` is a relative floor, not machine epsilon.
- `tol`, `input_error`, `guard`: Relative precision policy controls.
- `diagnostics`: If `true`, also return a [`NumericalDiagnostics`](@ref) record.

Returns a `Float64`.
"""
function FIM(
    p::Vector{R},
    dp::Vector{R};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {R<:Real}
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, work_type = eltype(p))
    pmax = maximum(p)
    τp = probability_floor(spec, pmax)
    kept = [pi > τp for pi in p]
    nkept = count(kept)
    deleted_mass = sum((max(pi, 0.0) for (pi, k) in zip(p, kept) if !k); init = 0.0)
    F = sum(((dp[i] * dp[i]) / p[i] for i in eachindex(p) if kept[i]); init = 0.0)
    diag = NumericalDiagnostics(
        :CFIM;
        spectral_floor = τp,
        numerical_rank = nkept,
        truncation_bias_bound = deleted_mass,
        deleted_mass = deleted_mass,
        flags = deleted_mass > 0 ? [:resolution_floor] : Symbol[],
        bottleneck = deleted_mass > 0 ? :truncation : :none,
    )
    return diagnostics ? (real(F), diag) : real(F)
end

"""

    FIM(p::Vector{R}, dp::Vector{Vector{R}}; eps=nothing, tol=1e-10,
        input_error=0.0, guard=nothing, diagnostics=false) where {R<:Real}

Calculate the classical Fisher information matrix for several parameters.
- `p`: The probability distribution.
- `dp`: Derivatives of the probability distribution with respect to the unknown parameters. For example, `dp[1]` is the derivative vector for the first parameter.
- `eps`: Deprecated relative probability-floor factor. With the default `nothing` the relative precision policy is used; when supplied, `eps` is a relative floor, not machine epsilon.
- `tol`, `input_error`, `guard`: Relative precision policy controls.
- `diagnostics`: If `true`, also return a [`NumericalDiagnostics`](@ref) record.

Returns an `n × n` `Matrix{Float64}`.
"""
function FIM(
    p::Vector{R},
    dp::Vector{Vector{R}};
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    diagnostics = false,
) where {R<:Real}
    spec = precision_spec(eps; input_error = input_error, tol = tol, guard = guard, work_type = eltype(p))
    pmax = maximum(p)
    τp = probability_floor(spec, pmax)
    para_num = length(dp[1])
    kept = [pj > τp for pj in p]
    nkept = count(kept)
    deleted_mass = sum((max(pj, 0.0) for (pj, k) in zip(p, kept) if !k); init = 0.0)
    FIM_res = reduce(
        +,
        (
            real.([dp[pj][a] * dp[pj][b] / p[pj] for a = 1:para_num, b = 1:para_num]) for
            pj in eachindex(p) if kept[pj]
        );
        init = zeros(para_num, para_num),
    )
    diag = NumericalDiagnostics(
        :CFIM;
        spectral_floor = τp,
        numerical_rank = nkept,
        truncation_bias_bound = deleted_mass,
        deleted_mass = deleted_mass,
        flags = deleted_mass > 0 ? [:resolution_floor] : Symbol[],
        bottleneck = deleted_mass > 0 ? :truncation : :none,
    )
    if para_num == 1
        value = FIM_res[1, 1]
    else
        value = FIM_res
    end
    return diagnostics ? (value, diag) : value
end

@doc raw"""
    FI_Expt(y1, y2, dx; ftype=:norm)

Estimate the classical Fisher information from two experimental data sets
separated by the known parameter drift `dx`.

- `y1`: Experimental data obtained at the truth value.
- `y2`: Experimental data obtained at `x + dx`.
- `dx`: Known small drift of the parameter (must be nonzero).
- `ftype`: Distribution fitted to the data; one of `:norm` (default),
  `:gamma`, `:rayleigh`, or `:poisson`.

Returns a `Float64` CFI estimate. An unsupported `ftype` prints a message and
returns `0.0`.
"""
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
- `eps`: legacy keyword (deprecated); when `nothing`, the new relative precision policy is used.
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
