# =====================================================================
# Precision and numerical-diagnostics policy.
#
# This file defines the five precision quantities used across the
# package -- machine precision `u`, input-data error `eta`, relative
# spectral floor `tau`, iterative stopping threshold `tau_stop`, and
# the user target accuracy `tol` -- together with the compensated
# arithmetic used to evaluate residual matrices and the diagnostic
# record returned by the numerical kernels.
#
# All thresholds defined here are *relative* (scaled by the largest
# eigenvalue/probability/singular value of the object they guard).
# =====================================================================

"""
    default_guard(::Type{T})

Default relative spectral floor, in units of the largest eigenvalue
(or probability/singular value), for floating-point type `T`.

The default is chosen from the numerical evidence for the
eigen-decomposition backward error of a `d`-dimensional Hermitian
matrix.  `Float64` uses ``10^{-14}`` and `Float32` uses ``5\\times
10^{-7}``; other types fall back to `64 * eps(T)`.
"""
default_guard(::Type{Float64}) = 1.0e-14
default_guard(::Type{Float32}) = 5.0f-7
default_guard(::Type{T}) where {T<:AbstractFloat} = 64 * eps(T)
default_guard(::Type{Complex{T}}) where {T<:AbstractFloat} = default_guard(T)
default_guard(::Type{T}) where {T} = default_guard(Float64)

"""
    PrecisionSpec(::Type{T}=Float64; input_error=0.0, tol=1e-10, guard=nothing, refine=true)

Central precision policy for a numerical evaluation.

# Fields

- `u`: unit roundoff of the working floating-point type.
- `input_error`: relative error of the input data (``\\eta``); it can
  never be smaller than `u` in the reported error budget.
- `tol`: target relative accuracy of the delivered quantity.
- `guard`: relative spectral floor (in units of the largest eigenvalue,
  probability or singular value).
- `refine`: whether one step of residual refinement is attempted.

# Examples

```julia
spec = PrecisionSpec(Float64; input_error = 1e-12, tol = 1e-10)
```
"""
struct PrecisionSpec
    u::Float64
    input_error::Float64
    tol::Float64
    guard::Float64
    refine::Bool
end

function PrecisionSpec(
    ::Type{T} = Float64;
    input_error::Real = 0.0,
    tol::Real = 1.0e-10,
    guard = nothing,
    refine::Bool = true,
) where {T}
    RT = T <: Complex ? real(T) : T
    u = Float64(eps(RT))
    g = guard === nothing ? Float64(default_guard(RT)) : Float64(guard)
    return PrecisionSpec(u, Float64(input_error), Float64(tol), g, refine)
end

"""
    precision_spec(eps; input_error=0.0, tol=1e-10, guard=nothing, refine=true)

Build a [`PrecisionSpec`](@ref) from the legacy `eps` keyword.

When `eps === nothing` the standard relative policy is used.  A
non-`nothing` value triggers a deprecation warning and is interpreted
as a *relative* spectral-floor factor (the historical absolute
thresholds are no longer supported); the effective guard is never
smaller than `default_guard(Float64)`.
"""
function precision_spec(
    eps;
    input_error::Real = 0.0,
    tol::Real = 1.0e-10,
    guard = nothing,
    refine::Bool = true,
    work_type::Type = Float64,
)
    if eps === nothing
        return PrecisionSpec(
            work_type;
            input_error = input_error,
            tol = tol,
            guard = guard,
            refine = refine,
        )
    else
        Base.depwarn(
            "the `eps` keyword is deprecated; it is now interpreted as a relative " *
            "spectral-floor factor (relative to the largest eigenvalue or probability). " *
            "Use `guard` (relative floor) or `tol` (target accuracy) instead.",
            :precision_spec,
        )
        RT = work_type <: Complex ? real(work_type) : work_type
        g = max(Float64(eps), Float64(default_guard(RT)))
        return PrecisionSpec(
            work_type;
            input_error = input_error,
            tol = tol,
            guard = g,
            refine = refine,
        )
    end
end

"""
    NumericalDiagnostics

Diagnostic record returned by the numerical kernels when
`diagnostics = true`.  All error quantities are expressed in the same
units as the value they accompany (absolute error estimates).

# Fields

- `quantity`: symbol of the computed quantity, e.g. `:SLD`, `:QFIM`, `:CFIM`.
- `precision_layer`: `:float64`, `:compensated` or `:highprec`.
- `spectral_floor`: the relative floor actually used.
- `numerical_rank`: number of eigenvalue pairs (probabilities, singular
  values) retained.
- `backward_error`: relative residual ``\\|R\\| / (2\\|\\rho\\|\\|L\\| + 2\\|\\sigma\\|)``.
- `forward_error_bound`: ``\\|R\\| / \\mathrm{sep}``.
- `input_error_propagation`: first-order propagation of the input-data error.
- `truncation_bias_bound`: norm of the dropped contribution.
- `condition_number`: condition number relevant for the quantity.
- `branch`: branch used (`:generic`, `:pure`, `:mixed`, ...).
- `cross_validation_gap`: relative difference between two independent routes.
- `upper_bound`: an upper bound of the quantity when available (e.g. QFIM for CFIM).
- `deleted_mass`: probability/eigenvalue mass discarded by the floor.
- `dropped_fisher_mass`: for SLD/QFIM diagnostics, the Fisher-space mass
  removed by the floor (the sum of 2*abs(sigma_ij)^2/(lambda_i+lambda_j)
  over the dropped pairs); this is the quantity that decides whether
  truncation matters for QFIM, whereas the operator-norm bound
  `truncation_bias_bound` is not sufficient for that decision.
- `flags`: warning flags, see below.
- `bottleneck`: the dominant error channel (`:rounding`, `:input_data`,
  `:truncation`, `:refinement_floor` or `:none`).

# Flags

- `:resolution_floor`: some terms were removed because they fell below
  the numerical resolution; the returned value is a lower bound.
- `:truncation_nonnegligible`: the truncation bias may be comparable to
  or larger than the target accuracy.
- `:rank_change_suspected`: the state or parameter family is close to a
  rank-change point where the quantity may diverge.
- `:input_error_dominant`: the input-data error dominates the budget.
- `:stagnated`: residual refinement stopped before reaching the target.
- `:hermiticity_corrected`: a non-Hermitian input was symmeterized.
- `:trace_corrected`: a non-unit-trace input was rescaled.
- `:pure_limit`: a pure-state limit had to be used.
- `:objective_nullspace`: the weight matrix has a non-negligible
  component on the numerical nullspace of ``F``.
"""
struct NumericalDiagnostics
    quantity::Symbol
    precision_layer::Symbol
    spectral_floor::Float64
    numerical_rank::Int
    backward_error::Float64
    forward_error_bound::Float64
    input_error_propagation::Float64
    truncation_bias_bound::Float64
    condition_number::Float64
    branch::Symbol
    cross_validation_gap::Float64
    upper_bound::Float64
    deleted_mass::Float64
    dropped_fisher_mass::Float64
    flags::Vector{Symbol}
    bottleneck::Symbol
end

function NumericalDiagnostics(
    quantity::Symbol;
    precision_layer::Symbol = :float64,
    spectral_floor::Real = 0.0,
    numerical_rank::Integer = 0,
    backward_error::Real = NaN,
    forward_error_bound::Real = NaN,
    input_error_propagation::Real = NaN,
    truncation_bias_bound::Real = 0.0,
    condition_number::Real = NaN,
    branch::Symbol = :generic,
    cross_validation_gap::Real = NaN,
    upper_bound::Real = NaN,
    deleted_mass::Real = 0.0,
    dropped_fisher_mass::Real = NaN,
    flags = Symbol[],
    bottleneck::Symbol = :none,
)
    flags_vec = _untracked(() -> collect(Symbol, flags))
    return NumericalDiagnostics(
        quantity,
        precision_layer,
        Float64(spectral_floor),
        Int(numerical_rank),
        Float64(backward_error),
        Float64(forward_error_bound),
        Float64(input_error_propagation),
        Float64(truncation_bias_bound),
        Float64(condition_number),
        branch,
        Float64(cross_validation_gap),
        Float64(upper_bound),
        Float64(deleted_mass),
        Float64(dropped_fisher_mass),
        flags_vec,
        bottleneck,
    )
end

# ---------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------

"""
    spectral_floor(spec::PrecisionSpec, scale::Real)

Relative spectral floor in absolute units: `max(guard, input_error) * scale`,
where `scale` is the largest eigenvalue (probability, singular value) of
the object being guarded.
"""
spectral_floor(spec::PrecisionSpec, scale::Real) =
    max(spec.guard, spec.input_error) * Float64(scale)

"""
    probability_floor(spec::PrecisionSpec, scale::Real)

Relative floor used when discarding small probabilities.
"""
probability_floor(spec::PrecisionSpec, scale::Real) =
    max(spec.guard, spec.input_error) * Float64(scale)

"""
    stop_threshold(spec::PrecisionSpec; kappa=1.0, residual_precision=nothing)

Relative residual threshold for an iterative solve, consistent with the
refinement floor ``c u / \\kappa + u_r``.
"""
function stop_threshold(
    spec::PrecisionSpec;
    kappa::Real = 1.0,
    residual_precision = nothing,
)
    ur = residual_precision === nothing ? spec.u : Float64(residual_precision)
    return min(1.0, spec.u / max(Float64(kappa), 1.0) + ur)
end

"""
    classify_bottleneck(; backward_error, forward_error_bound, input_error_propagation, truncation_bias_bound)

Return the dominant error channel.  Residual (backward) error is mapped
to the forward bound through `forward_error_bound` when available.
"""
function classify_bottleneck(;
    backward_error::Real = NaN,
    forward_error_bound::Real = NaN,
    input_error_propagation::Real = NaN,
    truncation_bias_bound::Real = NaN,
)
    best = :none
    bestv = -Inf
    for (k, v) in (
        :rounding => Float64(forward_error_bound),
        :input_data => Float64(input_error_propagation),
        :truncation => Float64(truncation_bias_bound),
    )
        if isfinite(v) && v > bestv
            best = k
            bestv = v
        end
    end
    return best
end

# ---------------------------------------------------------------------
# Compensated (double-double) arithmetic used for residuals.
# ---------------------------------------------------------------------

@inline function two_sum(a::Real, b::Real)
    a64 = Float64(a)
    b64 = Float64(b)
    s = a64 + b64
    bb = s - a64
    err = (a64 - (s - bb)) + (b64 - bb)
    return s, err
end

@inline function two_prod(a::Real, b::Real)
    a64 = Float64(a)
    b64 = Float64(b)
    p = a64 * b64
    err = fma(a64, b64, -p)
    return p, err
end

@inline function dd_add(hi::Real, lo::Real, term_hi::Real, term_lo::Real)
    s, e = two_sum(hi, term_hi)
    lo = Float64(lo) + (e + Float64(term_lo))
    s2, e2 = two_sum(s, lo)
    return s2, e2
end

"""
    _untracked(f)

Execute `f` outside the automatic-differentiation graph.  Used for the
diagnostic bookkeeping (flags, counters), which must not interfere with
the differentiable numerical path.
"""
@inline _untracked(f) = Zygote.ignore_derivatives(f)

@inline function _push_flag!(flags::Vector{Symbol}, flag::Symbol)
    Zygote.ignore_derivatives() do
        push!(flags, flag)
        nothing
    end
    return flags
end

"""
    compensated_dot(x, y)

Dot product with compensated (double-double) accumulation.  Returns the
pair `(hi, lo)` such that `hi + lo ≈ sum(x .* y)` with relative accuracy
close to `u^2` per element.
"""
function compensated_dot(x::AbstractVector{<:Complex}, y::AbstractVector{<:Complex})
    hr = 0.0
    lr = 0.0
    hi = 0.0
    li = 0.0
    @inbounds for k in eachindex(x, y)
        xk = x[k]
        yk = y[k]
        xr = real(xk)
        xi = imag(xk)
        yr = real(yk)
        yi = imag(yk)
        p1, e1 = two_prod(xr, yr)
        p2, e2 = two_prod(xi, yi)
        p3, e3 = two_prod(xr, yi)
        p4, e4 = two_prod(xi, yr)
        hr, lr = dd_add(hr, lr, p1, e1)
        hr, lr = dd_add(hr, lr, -p2, -e2)
        hi, li = dd_add(hi, li, p3, e3)
        hi, li = dd_add(hi, li, p4, e4)
    end
    return complex(hr, hi), complex(lr, li)
end

function compensated_dot(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    h = 0.0
    l = 0.0
    @inbounds for k in eachindex(x, y)
        p, e = two_prod(Float64(x[k]), Float64(y[k]))
        h, l = dd_add(h, l, p, e)
    end
    return h, l
end

"""
    compensated_mul(A, B)

Matrix product with compensated accumulation.  Returns `(C, E)` where
`C + E` is the product to nearly twice the working precision.
"""
function compensated_mul(A::AbstractMatrix{<:Complex}, B::AbstractMatrix{<:Complex})
    m = size(A, 1)
    n = size(B, 2)
    C = Matrix{ComplexF64}(undef, m, n)
    E = Matrix{ComplexF64}(undef, m, n)
    @inbounds for j = 1:n
        for i = 1:m
            h, l = compensated_dot(view(A, i, :), view(B, :, j))
            C[i, j] = h
            E[i, j] = l
        end
    end
    return C, E
end

"""
    lyapunov_residual(ρ, X, rhs)

Compensated evaluation of the residual ``R = \\mathrm{rhs} - \\rho X - X\\rho``
used by the SLD and Bayes-`Λ` equations.
"""
function lyapunov_residual(
    ρ::AbstractMatrix{<:Complex},
    X::AbstractMatrix{<:Complex},
    rhs::AbstractMatrix{<:Complex},
)
    P1, E1 = compensated_mul(ρ, X)
    P2, E2 = compensated_mul(X, ρ)
    n = size(ρ, 1)
    R = Matrix{ComplexF64}(undef, n, n)
    @inbounds for j = 1:n
        for i = 1:n
            hr = real(rhs[i, j])
            lr = 0.0
            hi = imag(rhs[i, j])
            li = 0.0
            hr, lr = dd_add(hr, lr, -real(P1[i, j]), -real(E1[i, j]))
            hi, li = dd_add(hi, li, -imag(P1[i, j]), -imag(E1[i, j]))
            hr, lr = dd_add(hr, lr, -real(P2[i, j]), -real(E2[i, j]))
            hi, li = dd_add(hi, li, -imag(P2[i, j]), -imag(E2[i, j]))
            R[i, j] = complex(hr + lr, hi + li)
        end
    end
    return R
end

# ---------------------------------------------------------------------
# Guarded inversion (class D: inversion of a computed result matrix)
# ---------------------------------------------------------------------

"""
    guarded_pinv(F; spec=PrecisionSpec(Float64), W=nothing, label="F")

Moore-Penrose pseudo-inverse of the Hermitian matrix `F` with a
relative cutoff derived from `spec`.  When the real symmetric weight
matrix `W` is supplied, the projection of `W` onto the numerical
nullspace of `F` is checked and an `ArgumentError` is thrown when it is
non-negligible (the objective ``\\mathrm{Tr}(WF^{+})`` diverges in that
case).

Returns `(Fpinv, nrank, smin, smax, null_projection)`.
"""
function guarded_pinv(
    F::AbstractMatrix;
    spec::PrecisionSpec = PrecisionSpec(Float64),
    W = nothing,
    label::AbstractString = "F",
)
    size(F, 1) == size(F, 2) || throw(ArgumentError("$label must be square."))
    Fh = Hermitian((F + F') / 2)
    e = eigen(Fh)
    vals = real.(e.values)
    smax = maximum(abs, vals)
    τ = spectral_floor(spec, smax)
    kept = findall(>(τ), abs.(vals))
    nrank = length(kept)
    if nrank == 0
        n = size(F, 1)
        return zeros(ComplexF64, n, n), 0, 0.0, smax, 0.0
    end
    smin = minimum(abs.(vals[kept]))
    # truncated spectral inverse (functional form, AD friendly)
    inv_vals = [(abs(v) > τ ? inv(v) : zero(v)) for v in vals]
    Finv = Hermitian((e.vectors * Diagonal(inv_vals) * e.vectors' + (e.vectors * Diagonal(inv_vals) * e.vectors')') / 2)
    null_projection = 0.0
    if W !== nothing
        Wm = Matrix(W)
        for (k, v) in enumerate(eachcol(e.vectors))
            if abs(vals[k]) <= τ
                wv = (v' * Wm * v)[1]
                null_projection += abs(wv)
            end
        end
        threshold = spec.tol * max(opnorm(Wm), 1.0)
        if null_projection > threshold
            throw(
                ArgumentError(
                    "objective Tr(W * pinv($label)) diverges: W has a non-negligible " *
                    "component on the numerical nullspace of $label " *
                    "(projection ≈ $(null_projection), threshold = $(threshold); " *
                    "κ($label) ≈ $(smax / smin), rank = $nrank).",
                ),
            )
        end
    end
    return Matrix(Finv), nrank, smin, smax, null_projection
end

"""
    guarded_objective(F, W; spec=PrecisionSpec(Float64))

Evaluate ``\\mathrm{Tr}(W F^{+})`` with the guarded pseudo-inverse.
Returns `(value, diag)`.
"""
function guarded_objective(F::AbstractMatrix, W; spec::PrecisionSpec = PrecisionSpec(Float64))
    Fpinv, nrank, smin, smax, null_projection =
        guarded_pinv(F; spec = spec, W = W, label = "F")
    value = real(tr(Matrix(W) * Fpinv))
    κ = smin > 0 ? smax / smin : Inf
    fl = Symbol[]
    null_projection > 0 && _push_flag!(fl, :objective_nullspace)
    input_prop = isfinite(κ) ? κ^2 * max(spec.input_error, spec.u) : Inf
    diag = NumericalDiagnostics(
        :objective;
        precision_layer = spec.refine ? :compensated : :float64,
        spectral_floor = spectral_floor(spec, smax),
        numerical_rank = nrank,
        condition_number = κ,
        input_error_propagation = input_prop,
        flags = fl,
        bottleneck = input_prop > 0 ? :input_data : :none,
    )
    return value, diag
end

# ---------------------------------------------------------------------
# Misc small helpers
# ---------------------------------------------------------------------

"""
    hermitize_and_normalize(ρ; spec)

Return `(ρh, trace_correction, hermiticity_correction)` where `ρh` is
the Hermitian part of `ρ` rescaled to unit trace when the deviation is
above the floor.  The corrections are reported as norms so that they
can be booked as input-data error.
"""
function hermitize_and_normalize(ρ::AbstractMatrix; spec::PrecisionSpec = PrecisionSpec(Float64))
    h_err = opnorm(ρ - ρ')
    ρh = Hermitian((ρ + ρ') / 2) |> Matrix
    tr_val = real(tr(ρh))
    t_err = abs(tr_val - 1.0)
    if t_err > max(spec.guard, spec.u) * max(abs(tr_val), 1.0) && tr_val > 0
        ρh = ρh ./ tr_val
    end
    return ρh, t_err, h_err
end
