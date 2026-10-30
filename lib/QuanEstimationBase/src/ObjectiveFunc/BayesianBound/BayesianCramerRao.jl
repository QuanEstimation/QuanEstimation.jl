########## Bayesian version of CFIM ##########
@doc raw"""
    simpson(x::AbstractVector, y::AbstractVector)

Numerical integration using the composite Simpson's rule.

For ``n`` points with uniform spacing ``h = (x_n-x_1)/(n-1)``:

```math
\int_{x_1}^{x_n} f(x)\,\mathrm{d}x \approx
\frac{h}{3}\Bigl(f_1+f_n + 4\sum_{\text{odd }i} f_i + 2\sum_{\text{even }i} f_i\Bigr).
```

If ``n`` is even, a trapezoidal correction is applied to the last subinterval.

# Arguments

- `x::AbstractVector`: Abscissa points (must be uniformly spaced).
- `y::AbstractVector`: Function values ``f(x_i)`` at the abscissa points.

# Returns

- `Float64`: The approximate integral.

# See Also

- `Trapz.trapz`: Trapezoidal integration (used for multi-dimensional integrals).
raw"""
function simpson(x::AbstractVector, y::AbstractVector)
    n = length(x)
    if n % 2 == 0
        h = (x[end] - x[1]) / (n - 1)
        s = y[1] + y[end]
        s += 4 * sum(y[2:2:n-2])
        s += 2 * sum(y[3:2:n-3])
        s = h * s / 3
        s += 0.5 * h * (y[end-1] + y[end])
        return s
    end
    h = (x[end] - x[1]) / (n - 1)
    s = y[1] + y[end]
    s += 4 * sum(y[2:2:n-1])
    s += 2 * sum(y[3:2:n-2])
    return h * s / 3
end

"""
    _pinv_diagnostics(spec, n, nrank, smin, smax, null_projection)

Diagnostics record for a guarded pseudo-inverse produced by
[`guarded_pinv`](@ref): numerical rank, condition number and the
rank/input-data flags.
"""
function _pinv_diagnostics(
    spec::PrecisionSpec,
    n::Integer,
    nrank::Integer,
    smin::Real,
    smax::Real,
    null_projection::Real,
)
    κ = smin > 0 ? smax / smin : Inf
    flags = Symbol[]
    nrank < n && push!(flags, :rank_change_suspected)
    null_projection > 0 && push!(flags, :objective_nullspace)
    return NumericalDiagnostics(
        :objective;
        precision_layer = spec.refine ? :compensated : :float64,
        spectral_floor = spectral_floor(spec, smax),
        numerical_rank = nrank,
        condition_number = κ,
        input_error_propagation = isfinite(κ) ? κ^2 * max(spec.input_error, spec.u) : Inf,
        flags = flags,
        bottleneck = :input_data,
    )
end

@doc raw"""

    BCFIM(x::AbstractVector, p, rho, drho; M=nothing, eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, refine=true,
        diagnostics=false)

Calculate the Bayesian classical Fisher information (BCFI) or Bayesian
classical Fisher information matrix (BCFIM)

```math
\mathcal{I}_{\mathrm{Bayes}}=\int p(\mathbf{x})\,\mathcal{I}\,\mathrm{d}\mathbf{x},
```

where ``\mathcal{I}`` is the CFIM and ``p(\mathbf{x})`` is the prior
distribution. The integral is evaluated with Simpson's rule for one parameter
and the trapezoidal rule for multiple parameters.

# Arguments

- `x`: The regimes of the parameters for the integral, given as a vector of
  grid vectors (one per parameter). For a single parameter, pass a one-element
  vector of grid vectors, e.g. `[xgrid]`.
- `p`: The prior distribution sampled on the grid, with shape
  `length.(x)`.
- `rho`: Parameterized density matrices, with one entry per grid point.
- `drho`: Derivatives of `rho` with respect to the unknown parameters, with
  one entry per grid point. Each entry is a vector with one derivative matrix
  per parameter.
- `M`: A set of positive operator-valued measures (POVMs). The default
  (`nothing`) uses a rank-one symmetric informationally complete POVM
  (SIC-POVM).
- `eps`: Legacy relative floor factor. When `nothing`, the relative precision
  policy is used; when given, it is deprecated and interpreted by the core.
- `tol`: Target relative accuracy of the inner CFIM evaluations.
- `input_error`: Relative error of the input data.
- `guard`: Relative spectral floor of the inner CFIM evaluations.
- `refine`: Reserved for API uniformity; the inner CFIM evaluations do not use
  residual refinement.
- `diagnostics`: When `true`, also return diagnostics for all grid points.

# Returns

- For one parameter (`length(x) == 1`), a `Float64` containing the integrated
  BCFI.
- For multiple parameters, a `length(x) × length(x)` matrix containing the
  integrated BCFIM.
- If `diagnostics = true`, returns `(F, diag)`, where `diag` is the
  [`NumericalDiagnostics`](@ref) record obtained by merging the diagnostics of
  all grid points.
raw"""
function BCFIM(
    x::AbstractVector,
    p,
    rho,
    drho;
    M = nothing,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    para_num = length(x)
    if para_num == 1
        #### singleparameter scenario ####
        p_num = length(p)
        if isnothing(M)
            M = SIC(size(rho[1])[1])
        end
        F_tp = zeros(p_num)
        diags = NumericalDiagnostics[]
        for i = 1:p_num
            if diagnostics
                F_tp[i], diag = CFIM(
                    rho[i],
                    drho[i][1],
                    M;
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    diagnostics = true,
                )
                push!(diags, diag)
            else
                F_tp[i] = CFIM(
                    rho[i],
                    drho[i][1],
                    M;
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                )
            end
        end
        F = 0.0
        arr = [p[i] * F_tp[i] for i = 1:p_num]
        F = simpson(x[1], arr)
        return diagnostics ? (F, _merge_diagnostics(:BCFIM, diags)) : F
    else
        #### multiparameter scenario #### 
        if isnothing(M)
            M = SIC(size(vec(rho)[1])[1])
        end

        xnum = length(x)
        trapzm(x, integrands, slice_dim) = [
            trapz(tuple(x...), I) for
            I in [reshape(hcat(integrands...)[i, :], length.(x)...) for i = 1:slice_dim]
        ]
        diags = NumericalDiagnostics[]
        Fs = Vector{Vector{Float64}}()
        for (pt, rho_tp, drho_tp) in zip(p, rho, drho)
            if diagnostics
                F_tp, diag = CFIM(
                    rho_tp,
                    drho_tp,
                    M;
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    diagnostics = true,
                )
                push!(Fs, pt * F_tp |> vec)
                push!(diags, diag)
            else
                push!(
                    Fs,
                    pt * CFIM(
                        rho_tp,
                        drho_tp,
                        M;
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                    ) |> vec,
                )
            end
        end
        F = trapzm(x, Fs, xnum^2) |> I -> reshape(I, xnum, xnum)
        return diagnostics ? (F, _merge_diagnostics(:BCFIM, diags)) : F
    end
end

########## Bayesian version of QFIM ##########
@doc raw"""

    BQFIM(x::AbstractVector, p, rho, drho; LDtype=:SLD, eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, refine=true,
        diagnostics=false)

Calculate the Bayesian quantum Fisher information (BQFI) or Bayesian quantum
Fisher information matrix (BQFIM)

```math
\mathcal{F}_{\mathrm{Bayes}}=\int p(\mathbf{x})\,\mathcal{F}\,\mathrm{d}\mathbf{x},
```

where ``\mathcal{F}`` is the QFIM of the selected type and ``p(\mathbf{x})``
is the prior distribution. The integral is evaluated with Simpson's rule for
one parameter and the trapezoidal rule for multiple parameters.

# Arguments

- `x`: The regimes of the parameters for the integral, given as a vector of
  grid vectors (one per parameter). For a single parameter, pass a one-element
  vector of grid vectors, e.g. `[xgrid]`.
- `p`: The prior distribution sampled on the grid, with shape
  `length.(x)`.
- `rho`: Parameterized density matrices, with one entry per grid point.
- `drho`: Derivatives of `rho` with respect to the unknown parameters, with
  one entry per grid point. Each entry is a vector with one derivative matrix
  per parameter.
- `LDtype`: Type of QFI (QFIM): `:SLD` (default), `:RLD`, or `:LLD`.
- `eps`: Legacy relative floor factor. When `nothing`, the relative precision
  policy is used; when given, it is deprecated and interpreted by the core.
- `tol`: Target relative accuracy of the inner QFIM evaluations.
- `input_error`: Relative error of the input data.
- `guard`: Relative spectral floor of the inner QFIM evaluations.
- `refine`: Whether one residual-refinement step is attempted in the inner
  SLD solves.
- `diagnostics`: When `true`, also return diagnostics for all grid points.

# Returns

- For one parameter (`length(x) == 1`), a `Float64` containing the integrated
  BQFI.
- For multiple parameters, a `length(x) × length(x)` matrix containing the
  integrated BQFIM.
- If `diagnostics = true`, returns `(F, diag)`, where `diag` is the
  [`NumericalDiagnostics`](@ref) record obtained by merging the diagnostics of
  all grid points.
raw"""
function BQFIM(
    x::AbstractVector,
    p,
    rho,
    drho;
    LDtype = :SLD,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    para_num = length(x)
    if para_num == 1
        #### singleparameter scenario ####
        p_num = length(p)
        F_tp = zeros(p_num)
        diags = NumericalDiagnostics[]
        for i = 1:p_num
            if diagnostics
                F_tp[i], diag = QFIM(
                    rho[i],
                    drho[i][1];
                    LDtype = LDtype,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    refine = refine,
                    diagnostics = true,
                )
                push!(diags, diag)
            else
                F_tp[i] = QFIM(
                    rho[i],
                    drho[i][1];
                    LDtype = LDtype,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    refine = refine,
                )
            end
        end
        F = 0.0
        arr = [p[i] * F_tp[i] for i = 1:p_num]
        F = simpson(x[1], arr)
        return diagnostics ? (F, _merge_diagnostics(:BQFIM, diags)) : F
    else
        #### multiparameter scenario #### 
        xnum = length(x)
        trapzm(x, integrands, slice_dim) = [
            trapz(tuple(x...), I) for
            I in [reshape(hcat(integrands...)[i, :], length.(x)...) for i = 1:slice_dim]
        ]
        diags = NumericalDiagnostics[]
        Fs = Vector{Vector{ComplexF64}}()
        for (pt, rho_tp, drho_tp) in zip(p, rho, drho)
            if diagnostics
                F_tp, diag = QFIM(
                    rho_tp,
                    drho_tp;
                    LDtype = LDtype,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    refine = refine,
                    diagnostics = true,
                )
                push!(Fs, pt * F_tp |> vec)
                push!(diags, diag)
            else
                push!(
                    Fs,
                    pt * QFIM(
                        rho_tp,
                        drho_tp;
                        LDtype = LDtype,
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        refine = refine,
                    ) |> vec,
                )
            end
        end
        F = trapzm(x, Fs, xnum^2) |> I -> reshape(I, xnum, xnum)
        return diagnostics ? (F, _merge_diagnostics(:BQFIM, diags)) : F
    end
end

########## Bayesian quantum Cramer-Rao bound ##########
@doc raw"""

    BQCRB(x::AbstractVector, p, dp, rho, drho; b=nothing, db=nothing,
        LDtype=:SLD, btype=1, eps=nothing, tol=1e-10, input_error=0.0,
        guard=nothing, refine=true, diagnostics=false)

Calculation of the Bayesian quantum Cramer-Rao bound (BQCRB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `b`: Vector of biases of the form ``\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}``.
- `db`: Derivatives of b on the unknown parameters to be estimated, It should be expressed as ``\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}``.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
- `btype`: Types of the BCRB. Options are 1, 2 and 3.
- `eps`: Legacy relative floor factor; when `nothing`, the relative precision policy is used; when given, it is deprecated and interpreted by the core.
- `tol`: Target relative accuracy of the inner QFIM evaluations.
- `input_error`: Relative error of the input data.
- `guard`: Relative spectral floor of the inner QFIM evaluations.
- `refine`: Whether one residual-refinement step is attempted in the inner SLD solves.
- `diagnostics`: When `true`, return `(value, diag)` where `diag` merges the
  diagnostics of the inner QFIM evaluations and the guarded pseudo-inverses
  into one [`NumericalDiagnostics`](@ref) record.
"""
function BQCRB(
    x::AbstractVector,
    p,
    dp,
    rho,
    drho;
    b = nothing,
    db = nothing,
    LDtype = :SLD,
    btype = 1,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    spec = precision_spec(
        eps;
        input_error = input_error,
        tol = tol,
        guard = guard,
        refine = refine,
    )

    if isnothing(b)
        b = [zero(x) for x in x]
        db = [zero(x) for x in x]
    end
    if isnothing(db)
        db = [zero(x) for x in x]
    end

    if x isa Vector{<:Number} || length(x) == 1
        if length(x) == 1
            x = x[1]
        end

        #### singleparameter scenario ####
        p_num = length(p)

        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i = 1:p_num]
        end
        if typeof(b[1]) == Vector{Float64} || typeof(b[1]) == Vector{Int64}
            b = b[1]
        end
        if typeof(db[1]) == Vector{Float64} || typeof(db[1]) == Vector{Int64}
            db = db[1]
        end
        F_tp = zeros(p_num)
        diags = NumericalDiagnostics[]
        for i = 1:p_num
            if diagnostics
                f, diag = QFIM(
                    rho[i],
                    drho[i];
                    LDtype = LDtype,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    refine = refine,
                    diagnostics = true,
                )
                F_tp[i] = f
                push!(diags, diag)
            else
                f = QFIM(
                    rho[i],
                    drho[i];
                    LDtype = LDtype,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    refine = refine,
                )
                F_tp[i] = f
            end
        end
        F = 0.0
        if btype == 1
            arr = [p[i] * ((1 + db[i])^2 / F_tp[i] + b[i]^2) for i = 1:p_num]
            F = simpson(x, arr)
        elseif btype == 2
            arr = [p[i] * F_tp[i] for i = 1:p_num]
            F1 = simpson(x, arr)
            arr2 = [p[j] * (1 + db[j]) for j = 1:p_num]
            B = simpson(x, arr2)
            arr3 = [p[k] * b[k]^2 for k = 1:p_num]
            bb = simpson(x, arr3)
            F = B^2 / F1 + bb
        elseif btype == 3
            I_tp = [real(dp[i] * dp[i] / p[i]^2) for i = 1:p_num]
            arr = [
                p[j] * (dp[j] * b[j] / p[j] + (1 + db[j]))^2 / (I_tp[j] + F_tp[j]) for
                j = 1:p_num
            ]
            F = simpson(x, arr)
        else
            println("NameError: btype should be choosen in {1, 2, 3}.")
        end
        return diagnostics ? (F, _merge_diagnostics(:BQCRB, diags)) : F
    else
        #### multiparameter scenario ####
        xnum = length(x)
        bs = Iterators.product(b...)
        dbs = Iterators.product(db...)
        trapzm(x, integrands, slice_dim) = [
            trapz(tuple(x...), I) for
            I in [reshape(hcat(integrands...)[i, :], length.(x)...) for i = 1:slice_dim]
        ]
        diags = NumericalDiagnostics[]

        if btype == 1
            integrands = Vector{Vector{ComplexF64}}()
            for (pt, rho_tp, drho_tp, b_tp, db_tp) in zip(p, rho, drho, bs, dbs)
                bv = [b_tp...]
                dbv = [db_tp...]
                if diagnostics
                    Fq, dq = QFIM(
                        rho_tp,
                        drho_tp;
                        LDtype = LDtype,
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        refine = refine,
                        diagnostics = true,
                    )
                    push!(diags, dq)
                    Fpinv, nrank, smin, smax, nullp =
                        guarded_pinv(Fq; spec = spec, label = "QFIM")
                    push!(
                        diags,
                        _pinv_diagnostics(spec, size(Fq, 1), nrank, smin, smax, nullp),
                    )
                else
                    Fq = QFIM(
                        rho_tp,
                        drho_tp;
                        LDtype = LDtype,
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        refine = refine,
                    )
                    Fpinv = guarded_pinv(Fq; spec = spec, label = "QFIM")[1]
                end
                push!(
                    integrands,
                    (pt * diagm(1 .+ dbv) * Fpinv * diagm(1 .+ dbv) + bv * bv') |> vec,
                )
            end
            I = trapzm(x, integrands, xnum^2) |> I -> reshape(I, xnum, xnum)
        elseif btype == 2
            Bs = [p * (1 .+ [db...]) for (p, db) in zip(p, dbs)]
            B = trapzm(x, Bs, xnum) |> diagm
            Fs = Vector{Vector{ComplexF64}}()
            for (pt, rho_tp, drho_tp) in zip(p, rho, drho)
                if diagnostics
                    Fq, dq = QFIM(
                        rho_tp,
                        drho_tp;
                        LDtype = LDtype,
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        refine = refine,
                        diagnostics = true,
                    )
                    push!(Fs, pt * Fq |> vec)
                    push!(diags, dq)
                else
                    push!(
                        Fs,
                        pt * QFIM(
                            rho_tp,
                            drho_tp;
                            LDtype = LDtype,
                            eps = eps,
                            tol = tol,
                            input_error = input_error,
                            guard = guard,
                            refine = refine,
                        ) |> vec,
                    )
                end
            end
            F = trapzm(x, Fs, xnum^2) |> I -> reshape(I, xnum, xnum)
            bbts = [p * [b...] * [b...]' |> vec for (p, b) in zip(p, bs)]
            if diagnostics
                Fpinv, nrank, smin, smax, nullp = guarded_pinv(F; spec = spec, label = "F")
                push!(diags, _pinv_diagnostics(spec, size(F, 1), nrank, smin, smax, nullp))
            else
                Fpinv = guarded_pinv(F; spec = spec, label = "F")[1]
            end
            I = B * Fpinv * B + (trapzm(x, bbts, xnum^2) |> I -> reshape(I, xnum, xnum))
        elseif btype == 3
            Ip(p, dp) = dp * dp' / p^2
            G = [G_mat(p, dp, [b...], [db...]) for (p, dp, b, db) in zip(p, dp, bs, dbs)]
            integrands = Vector{Vector{ComplexF64}}()
            for (pt, dp_tp, rho_tp, drho_tp, G_tp) in zip(p, dp, rho, drho, G)
                if diagnostics
                    Fq, dq = QFIM(
                        rho_tp,
                        drho_tp;
                        LDtype = LDtype,
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        refine = refine,
                        diagnostics = true,
                    )
                    push!(diags, dq)
                    Msum = Ip(pt, dp_tp) + Fq
                    Fpinv, nrank, smin, smax, nullp =
                        guarded_pinv(Msum; spec = spec, label = "Ip + QFIM")
                    push!(
                        diags,
                        _pinv_diagnostics(spec, size(Msum, 1), nrank, smin, smax, nullp),
                    )
                else
                    Fq = QFIM(
                        rho_tp,
                        drho_tp;
                        LDtype = LDtype,
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        refine = refine,
                    )
                    Fpinv =
                        guarded_pinv(Ip(pt, dp_tp) + Fq; spec = spec, label = "Ip + QFIM")[1]
                end
                push!(integrands, (pt * G_tp * Fpinv * G_tp') |> vec)
            end
            I = trapzm(x, integrands, xnum^2) |> I -> reshape(I, xnum, xnum)
        else
            println("NameError: btype should be choosen in {1, 2, 3}.")
        end
        return diagnostics ? (I, _merge_diagnostics(:BQCRB, diags)) : I
    end
end

"""
    BQCRB(scheme::Scheme; b=nothing, db=nothing, btype=1, eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, refine=true,
        diagnostics=false)

Compute the Bayesian quantum Cramér-Rao bound (BQCRB) from a Scheme.

Evolves the state over the parameter region encoded in the scheme and calls
[`BQCRB`](@ref) with the prior distribution and biases.

`eps` is a legacy keyword; when `nothing`, the new relative precision policy is used; when given, it is deprecated and interpreted by the core.

# See Also

- [`BQCRB`](@ref): Direct BQCRB computation.
raw"""
function BQCRB(
    scheme::Scheme;
    b = nothing,
    db = nothing,
    btype = 1,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    rho, drho = evolve_parameter_region(scheme)
    (; x, p, dp) = getfield(scheme, :EstimationStrategy)
    return BQCRB(
        x,
        p,
        dp,
        rho,
        drho;
        b = b,
        db = db,
        btype = btype,
        eps = eps,
        tol = tol,
        input_error = input_error,
        guard = guard,
        refine = refine,
        diagnostics = diagnostics,
    )
end


@doc raw"""

    BCRB(x::AbstractVector, p, dp, rho, drho; M=nothing, b=nothing,
        db=nothing, btype=1, eps=nothing, tol=1e-10, input_error=0.0,
        guard=nothing, refine=true, diagnostics=false)

Calculation of the Bayesian Cramer-Rao bound (BCRB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `b`: Vector of biases of the form ``\textbf{b}=(b(x_0),b(x_1),\dots)^{\mathrm{T}}``.
- `db`: Derivatives of b on the unknown parameters to be estimated, It should be expressed as ``\textbf{b}'=(\partial_0 b(x_0),\partial_1 b(x_1),\dots)^{\mathrm{T}}``.
- `btype`: Types of the BCRB. Options are 1, 2 and 3.
- `eps`: Legacy relative floor factor; when `nothing`, the relative precision policy is used; when given, it is deprecated and interpreted by the core.
- `tol`: Target relative accuracy of the inner CFIM evaluations.
- `input_error`: Relative error of the input data.
- `guard`: Relative spectral floor of the inner CFIM evaluations.
- `refine`: Reserved for API uniformity; the inner CFIM evaluations do not use
  residual refinement.
- `diagnostics`: When `true`, return `(value, diag)` where `diag` merges the
  diagnostics of the inner CFIM evaluations and the guarded pseudo-inverses
  into one [`NumericalDiagnostics`](@ref) record.
"""
function BCRB(
    x::AbstractVector,
    p,
    dp,
    rho,
    drho;
    M = nothing,
    b = nothing,
    db = nothing,
    btype = 1,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    spec = precision_spec(
        eps;
        input_error = input_error,
        tol = tol,
        guard = guard,
        refine = refine,
    )
    if isnothing(b)
        b = [zero(x) for x in x]
        db = [zero(x) for x in x]
    end
    if isnothing(db)
        db = [zero(x) for x in x]
    end

    if x isa Vector{<:Number} || length(x) == 1
        if length(x) == 1
            x = x[1]
        end
        #### singleparameter scenario ####
        p_num = length(p)
        if isnothing(M)
            M = SIC(size(rho[1])[1])
        end
        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i = 1:p_num]
        end
        if typeof(b[1]) == Vector{Float64} || typeof(b[1]) == Vector{Int64}
            b = b[1]
        end
        if typeof(db[1]) == Vector{Float64} || typeof(db[1]) == Vector{Int64}
            db = db[1]
        end
        F_tp = zeros(p_num)
        diags = NumericalDiagnostics[]

        for i = 1:p_num
            if diagnostics
                f, diag = CFIM(
                    rho[i],
                    drho[i];
                    M = M,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    diagnostics = true,
                )
                F_tp[i] = f
                push!(diags, diag)
            else
                f = CFIM(
                    rho[i],
                    drho[i];
                    M = M,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                )
                F_tp[i] = f
            end
        end
        F = 0.0
        if btype == 1
            arr = [p[i] * ((1 + db[i])^2 / F_tp[i] + b[i]^2) for i = 1:p_num]
            F = simpson(x, arr)
        elseif btype == 2
            arr = [p[i] * F_tp[i] for i = 1:p_num]
            F1 = simpson(x, arr)
            arr2 = [p[j] * (1 + db[j]) for j = 1:p_num]
            B = simpson(x, arr2)
            arr3 = [p[k] * b[k]^2 for k = 1:p_num]
            bb = simpson(x, arr3)
            F = B^2 / F1 + bb
        elseif btype == 3
            I_tp = [real(dp[i] * dp[i] / p[i]^2) for i = 1:p_num]
            arr = [
                p[j] * (dp[j] * b[j] / p[j] + (1 + db[j]))^2 / (I_tp[j] + F_tp[j]) for
                j = 1:p_num
            ]
            F = simpson(x, arr)
        else
            println("NameError: btype should be choosen in {1, 2, 3}")
        end
        return diagnostics ? (F, _merge_diagnostics(:BCRB, diags)) : F
    else
        #### multiparameter scenario #### 
        if isnothing(M)
            M = SIC(size(vec(rho)[1])[1])
        end

        xnum = length(x)
        bs = Iterators.product(b...)
        dbs = Iterators.product(db...)
        trapzm(x, integrands, slice_dim) = [
            trapz(tuple(x...), I) for
            I in [reshape(hcat(integrands...)[i, :], length.(x)...) for i = 1:slice_dim]
        ]
        diags = NumericalDiagnostics[]

        if btype == 1
            integrands = Vector{Vector{ComplexF64}}()
            for (pt, rho_tp, drho_tp, b_tp, db_tp) in zip(p, rho, drho, bs, dbs)
                bv = [b_tp...]
                dbv = [db_tp...]
                if diagnostics
                    Fq, dq = CFIM(
                        rho_tp,
                        drho_tp,
                        M;
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        diagnostics = true,
                    )
                    push!(diags, dq)
                    Fpinv, nrank, smin, smax, nullp =
                        guarded_pinv(Fq; spec = spec, label = "CFIM")
                    push!(
                        diags,
                        _pinv_diagnostics(spec, size(Fq, 1), nrank, smin, smax, nullp),
                    )
                else
                    Fq = CFIM(
                        rho_tp,
                        drho_tp,
                        M;
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                    )
                    Fpinv = guarded_pinv(Fq; spec = spec, label = "CFIM")[1]
                end
                push!(
                    integrands,
                    (pt * diagm(1 .+ dbv) * Fpinv * diagm(1 .+ dbv) + bv * bv') |> vec,
                )
            end
            I = trapzm(x, integrands, xnum^2) |> I -> reshape(I, xnum, xnum)
        elseif btype == 2
            Bs = [p * (1 .+ [db...]) for (p, db) in zip(p, dbs)]
            B = trapzm(x, Bs, xnum) |> diagm
            Fs = Vector{Vector{Float64}}()
            for (pt, rho_tp, drho_tp) in zip(p, rho, drho)
                if diagnostics
                    Fq, dq = CFIM(
                        rho_tp,
                        drho_tp,
                        M;
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        diagnostics = true,
                    )
                    push!(Fs, pt * Fq |> vec)
                    push!(diags, dq)
                else
                    push!(
                        Fs,
                        pt * CFIM(
                            rho_tp,
                            drho_tp,
                            M;
                            eps = eps,
                            tol = tol,
                            input_error = input_error,
                            guard = guard,
                        ) |> vec,
                    )
                end
            end
            F = trapzm(x, Fs, xnum^2) |> I -> reshape(I, xnum, xnum)
            bbts = [p * [b...] * [b...]' |> vec for (p, b) in zip(p, bs)]
            if diagnostics
                Fpinv, nrank, smin, smax, nullp = guarded_pinv(F; spec = spec, label = "F")
                push!(diags, _pinv_diagnostics(spec, size(F, 1), nrank, smin, smax, nullp))
            else
                Fpinv = guarded_pinv(F; spec = spec, label = "F")[1]
            end
            I = B * Fpinv * B + (trapzm(x, bbts, xnum^2) |> I -> reshape(I, xnum, xnum))
        elseif btype == 3
            Ip(p, dp) = dp * dp' / p^2
            G = [G_mat(p, dp, [b...], [db...]) for (p, dp, b, db) in zip(p, dp, bs, dbs)]
            integrands = Vector{Vector{ComplexF64}}()
            for (pt, dp_tp, rho_tp, drho_tp, G_tp) in zip(p, dp, rho, drho, G)
                if diagnostics
                    Fq, dq = CFIM(
                        rho_tp,
                        drho_tp,
                        M;
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        diagnostics = true,
                    )
                    push!(diags, dq)
                    Msum = Ip(pt, dp_tp) + Fq
                    Fpinv, nrank, smin, smax, nullp =
                        guarded_pinv(Msum; spec = spec, label = "Ip + CFIM")
                    push!(
                        diags,
                        _pinv_diagnostics(spec, size(Msum, 1), nrank, smin, smax, nullp),
                    )
                else
                    Fq = CFIM(
                        rho_tp,
                        drho_tp,
                        M;
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                    )
                    Fpinv =
                        guarded_pinv(Ip(pt, dp_tp) + Fq; spec = spec, label = "Ip + CFIM")[1]
                end
                push!(integrands, (pt * G_tp * Fpinv * G_tp') |> vec)
            end
            I = trapzm(x, integrands, xnum^2) |> I -> reshape(I, xnum, xnum)
        else
            println("NameError: btype should be choosen in {1, 2, 3}.")
        end
        return diagnostics ? (I, _merge_diagnostics(:BCRB, diags)) : I
    end
end

raw"""
    BCRB(scheme::Scheme; b=nothing, db=nothing, btype=1, eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, refine=true,
        diagnostics=false)

Compute the Bayesian Cramér-Rao bound (BCRB) from a Scheme.

Evolves the state over the parameter region and calls [`BCRB`](@ref) with
the measurement from the scheme.

`eps` is a legacy keyword; when `nothing`, the new relative precision policy is used; when given, it is deprecated and interpreted by the core.

# See Also

- [`BCRB`](@ref): Direct BCRB computation.
raw"""
function BCRB(
    scheme::Scheme;
    b = nothing,
    db = nothing,
    btype = 1,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    rho, drho = evolve_parameter_region(scheme)
    (; x, p, dp) = getfield(scheme, :EstimationStrategy)
    return BCRB(
        x,
        p,
        dp,
        rho,
        drho;
        M = meas_data(scheme),
        b = b,
        db = db,
        btype = btype,
        eps = eps,
        tol = tol,
        input_error = input_error,
        guard = guard,
        refine = refine,
        diagnostics = diagnostics,
    )
end


@doc raw"""
    G_mat(p, dp, b, db)

Construct the ``G`` matrix for the BCRB type-3 formulation.

The matrix ``G`` combines the prior derivative and bias information:

```math
G_{ab}(x) = \frac{1}{p(x)}\frac{\partial p}{\partial x_b}\,b_a(x) +
\underbrace{(1+\partial_b b_a(x))\,\delta_{ab}}_{\text{diagonal only}}.
```

# Arguments

- `p`: Prior distribution value at the point.
- `dp`: Vector of prior derivatives ``\partial_a p``.
- `b`: Vector of biases ``b_a``.
- `db`: Vector of bias derivatives ``\partial_a b_a``.

# Returns

- `Matrix{Float64}`: The ``G`` matrix at the given parameter point.

# See Also

- [`BQCRB`](@ref), [`BCRB`](@ref): Bound functions using ``G``.
"""
function G_mat(p, dp, b, db)
    para_num = length(db)
    G_tp = zeros(para_num, para_num)
    for i = 1:para_num
        for j = 1:para_num
            if i == j
                G_tp[i, j] = dp[j] * b[i] / p + (1 + db[i])
            else
                G_tp[i, j] = dp[j] * b[i] / p
            end
        end
    end
    return G_tp
end

"""

    QVTB(x::AbstractVector, p, dp, rho, drho; LDtype=:SLD, eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, refine=true,
        diagnostics=false)

Calculation of the Bayesian version of Cramer-Rao bound in troduced by Van Trees (VTB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are "SLD" (default), "RLD" and "LLD".
- `eps`: Legacy relative floor factor; when `nothing`, the relative precision policy is used; when given, it is deprecated and interpreted by the core.
- `tol`: Target relative accuracy of the inner QFIM evaluations.
- `input_error`: Relative error of the input data.
- `guard`: Relative spectral floor of the inner QFIM evaluations.
- `refine`: Whether one residual-refinement step is attempted in the inner SLD solves.
- `diagnostics`: When `true`, return `(value, diag)` where `diag` merges the
  diagnostics of the inner QFIM evaluations and of the guarded inversion of
  ``I_{\\mathrm{prior}} + F`` into one [`NumericalDiagnostics`](@ref) record.
"""
function QVTB(
    x::AbstractVector,
    p,
    dp,
    rho,
    drho;
    LDtype = :SLD,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    spec = precision_spec(
        eps;
        input_error = input_error,
        tol = tol,
        guard = guard,
        refine = refine,
    )
    if x isa Vector{<:Number} || length(x) == 1
        if length(x) == 1
            x = x[1]
        end
        #### singleparameter scenario ####
        p_num = length(p)
        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i = 1:p_num]
        end
        if typeof(dp[1]) == Vector{Float64}
            dp = [dp[i][1] for i = 1:p_num]
        end
        F_tp = zeros(p_num)
        diags = NumericalDiagnostics[]
        for m = 1:p_num
            if diagnostics
                F_tp[m], diag = QFIM(
                    rho[m],
                    drho[m];
                    LDtype = LDtype,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    refine = refine,
                    diagnostics = true,
                )
                push!(diags, diag)
            else
                F_tp[m] = QFIM(
                    rho[m],
                    drho[m];
                    LDtype = LDtype,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    refine = refine,
                )
            end
        end

        arr1 = [real(dp[i] * dp[i] / p[i]) for i = 1:p_num]
        I = simpson(x, arr1)
        arr2 = [real(F_tp[j] * p[j]) for j = 1:p_num]
        F = simpson(x, arr2)
        I = 1.0 / (I + F)
        return diagnostics ? (I, _merge_diagnostics(:QVTB, diags)) : I
    else
        #### multiparameter scenario ####
        xnum = length(x)
        trapzm(x, integrands, slice_dim) = [
            trapz(tuple(x...), I) for
            I in [reshape(hcat(integrands...)[i, :], length.(x)...) for i = 1:slice_dim]
        ]
        Ip(p, dp) = dp * dp' / p^2

        Iprs = [p * Ip(p, dp) |> vec for (p, dp) in zip(p, dp)]
        Ipr = trapzm(x, Iprs, xnum^2) |> I -> reshape(I, xnum, xnum)
        diags = NumericalDiagnostics[]
        Fs = Vector{Vector{ComplexF64}}()
        for (pt, rho_tp, drho_tp) in zip(p, rho, drho)
            if diagnostics
                F_tp, diag = QFIM(
                    rho_tp,
                    drho_tp;
                    LDtype = LDtype,
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    refine = refine,
                    diagnostics = true,
                )
                push!(Fs, pt * F_tp |> vec)
                push!(diags, diag)
            else
                push!(
                    Fs,
                    pt * QFIM(
                        rho_tp,
                        drho_tp;
                        LDtype = LDtype,
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                        refine = refine,
                    ) |> vec,
                )
            end
        end
        F = trapzm(x, Fs, xnum^2) |> I -> reshape(I, xnum, xnum)
        if diagnostics
            I, nrank, smin, smax, nullp =
                guarded_pinv(Ipr + F; spec = spec, label = "Ipr + F")
            push!(
                diags,
                _pinv_diagnostics(spec, size(Ipr + F, 1), nrank, smin, smax, nullp),
            )
            return I, _merge_diagnostics(:QVTB, diags)
        end
        I = guarded_pinv(Ipr + F; spec = spec, label = "Ipr + F")[1]
        return I
    end
end

"""
    QVTB(scheme::Scheme; LDtype=:SLD, eps=nothing, tol=1e-10,
        input_error=0.0, guard=nothing, refine=true, diagnostics=false)

Compute the quantum Van Trees bound from a Scheme.

Evolves the state over the parameter region and calls [`QVTB`](@ref).

`eps` is a legacy keyword; when `nothing`, the new relative precision policy is used; when given, it is deprecated and interpreted by the core.

# See Also

- [`QVTB`](@ref): Direct quantum Van Trees bound.
"""
function QVTB(
    scheme::Scheme;
    LDtype = :SLD,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    rho, drho = evolve_parameter_region(scheme)
    (; x, p, dp) = getfield(scheme, :EstimationStrategy)
    return QVTB(
        x,
        p,
        dp,
        rho,
        drho;
        LDtype = LDtype,
        eps = eps,
        tol = tol,
        input_error = input_error,
        guard = guard,
        refine = refine,
        diagnostics = diagnostics,
    )
end


"""

    VTB(x::AbstractVector, p, dp, rho, drho; M=nothing, eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, refine=true,
        diagnostics=false)

Calculation of the Bayesian version of Cramer-Rao bound introduced by Van Trees (VTB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `dp`: Derivatives of the prior distribution with respect to the unknown parameters to be estimated. For example, dp[0] is the derivative vector on the first parameter.
- `rho`: Parameterized density matrix.
- `drho`: Derivatives of the parameterized density matrix (rho) with respect to the unknown parameters to be estimated.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `eps`: Legacy relative floor factor; when `nothing`, the relative precision policy is used; when given, it is deprecated and interpreted by the core.
- `tol`: Target relative accuracy of the inner CFIM evaluations.
- `input_error`: Relative error of the input data.
- `guard`: Relative spectral floor of the inner CFIM evaluations.
- `refine`: Reserved for API uniformity; the inner CFIM evaluations do not use
  residual refinement.
- `diagnostics`: When `true`, return `(value, diag)` where `diag` merges the
  diagnostics of the inner CFIM evaluations and of the guarded inversion of
  ``I_{\\mathrm{prior}} + I`` into one [`NumericalDiagnostics`](@ref) record.
"""
function VTB(
    x::AbstractVector,
    p,
    dp,
    rho,
    drho;
    M = nothing,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    spec = precision_spec(
        eps;
        input_error = input_error,
        tol = tol,
        guard = guard,
        refine = refine,
    )
    if x isa Vector{<:Number} || length(x) == 1
        if length(x) == 1
            x = x[1]
        end
        #### singleparameter scenario ####
        p_num = length(p)
        if isnothing(M)
            M = SIC(size(rho[1])[1])
        end
        if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
            drho = [drho[i][1] for i = 1:p_num]
        end
        if typeof(dp[1]) == Vector{Float64}
            dp = [dp[i][1] for i = 1:p_num]
        end
        F_tp = zeros(p_num)
        diags = NumericalDiagnostics[]
        for m = 1:p_num
            if diagnostics
                F_tp[m], diag = CFIM(
                    rho[m],
                    drho[m],
                    M;
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    diagnostics = true,
                )
                push!(diags, diag)
            else
                F_tp[m] = CFIM(
                    rho[m],
                    drho[m],
                    M;
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                )
            end
        end

        arr1 = [real(dp[i] * dp[i] / p[i]) for i = 1:p_num]
        I = simpson(x, arr1)
        arr2 = [real(F_tp[j] * p[j]) for j = 1:p_num]
        F = simpson(x, arr2)
        res = 1.0 / (I + F)
        return diagnostics ? (res, _merge_diagnostics(:VTB, diags)) : res
    else
        #### multiparameter scenario #### 
        if isnothing(M)
            M = SIC(size(vec(rho)[1])[1])
        end
        xnum = length(x)
        trapzm(x, integrands, slice_dim) = [
            trapz(tuple(x...), I) for
            I in [reshape(hcat(integrands...)[i, :], length.(x)...) for i = 1:slice_dim]
        ]
        Ip(p, dp) = dp * dp' / p^2

        Iprs = [p * Ip(p, dp) |> vec for (p, dp) in zip(p, dp)]
        Ipr = trapzm(x, Iprs, xnum^2) |> I -> reshape(I, xnum, xnum)
        diags = NumericalDiagnostics[]
        Fs = Vector{Vector{Float64}}()
        for (pt, rho_tp, drho_tp) in zip(p, rho, drho)
            if diagnostics
                F_tp, diag = CFIM(
                    rho_tp,
                    drho_tp,
                    M;
                    eps = eps,
                    tol = tol,
                    input_error = input_error,
                    guard = guard,
                    diagnostics = true,
                )
                push!(Fs, pt * F_tp |> vec)
                push!(diags, diag)
            else
                push!(
                    Fs,
                    pt * CFIM(
                        rho_tp,
                        drho_tp,
                        M;
                        eps = eps,
                        tol = tol,
                        input_error = input_error,
                        guard = guard,
                    ) |> vec,
                )
            end
        end
        F = trapzm(x, Fs, xnum^2) |> I -> reshape(I, xnum, xnum)
        if diagnostics
            I, nrank, smin, smax, nullp =
                guarded_pinv(Ipr + F; spec = spec, label = "Ipr + F")
            push!(
                diags,
                _pinv_diagnostics(spec, size(Ipr + F, 1), nrank, smin, smax, nullp),
            )
            return I, _merge_diagnostics(:VTB, diags)
        end
        I = guarded_pinv(Ipr + F; spec = spec, label = "Ipr + F")[1]
        return I
    end
end

"""
    VTB(scheme::Scheme; eps=nothing, tol=1e-10, input_error=0.0,
        guard=nothing, refine=true, diagnostics=false)

Compute the classical Van Trees bound from a Scheme.

Evolves the state over the parameter region and calls [`VTB`](@ref) with
the measurement from the scheme.

`eps` is a legacy keyword; when `nothing`, the new relative precision policy is used; when given, it is deprecated and interpreted by the core.

# See Also

- [`VTB`](@ref): Direct Van Trees bound.
raw"""
function VTB(
    scheme::Scheme;
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    rho, drho = evolve_parameter_region(scheme)
    (; x, p, dp) = getfield(scheme, :EstimationStrategy)
    return VTB(
        x,
        p,
        dp,
        rho,
        drho;
        M = meas_data(scheme),
        eps = eps,
        tol = tol,
        input_error = input_error,
        guard = guard,
        refine = refine,
        diagnostics = diagnostics,
    )
end

########## optimal biased bound ##########
@doc raw"""
    OBB_func(du, u, para, t)

ODE right-hand side for the optimal biased bound (OBB) differential equation.

Solves the Euler-Lagrange equation for the optimal bias function ``b(x)``:

```math
\frac{\mathrm{d}}{\mathrm{d}x}\begin{pmatrix}b\\ b'\end{pmatrix} =
\begin{pmatrix}
b' \\
-J(x)\,b' + F(x)\,b - J(x)
\end{pmatrix},
```

where ``F(x)`` is the QFI, ``J(x) = p'(x)/p(x) - F'(x)/F(x)``, and
``p(x)`` is the prior. This is used internally by [`OBB`](@ref) via
`DifferentialEquations.jl`.

# Arguments

- `du`: Output derivative vector ``[b', b'']``.
- `u`: Current state ``[b, b']``.
- `para`: Tuple ``(F, J, x)`` containing the QFI array, ``J`` array, and
  abscissa grid.
- `t`: Current abscissa value ``x``.

# See Also

- [`OBB`](@ref): Top-level OBB computation.
- [`boundary_condition`](@ref): Boundary-value problem condition.
raw"""
function OBB_func(du, u, para, t)
    F, J, x = para
    J_tp = interp1(x, J, t)
    F_tp = interp1(x, F, t)
    bias = u[1]
    dbias = u[2]
    du[1] = dbias
    du[2] = -J_tp * dbias + F_tp * bias - J_tp
end

@doc raw"""
    boundary_condition(residual, u, p, t)

Boundary condition for the OBB boundary-value problem (BVP).

Imposes ``b'(x_0) + 1 = 0`` at the left endpoint and
``b'(x_{\mathrm{end}}) + 1 = 0`` at the right endpoint.
These natural boundary conditions ensure the optimal bias satisfies
``b'(x_{\mathrm{boundary}}) = -1``.

Used by `DifferentialEquations.jl` via `BVProblem`.

# See Also

- [`OBB_func`](@ref): ODE right-hand side.
- [`OBB`](@ref): Top-level OBB computation.
"""
function boundary_condition(residual, u, p, t)
    residual[1] = u[1][2] + 1.0
    residual[2] = u[end][2] + 1.0
end

"""
    interp1(xspan, yspan, x)

Linear interpolation of a 1D function.

Evaluates ``f(x)`` using linear interpolation on the grid `(xspan, yspan)`.

# Arguments

- `xspan`: Abscissa grid points.
- `yspan`: Function values at the grid points.
- `x`: Evaluation point(s).

# Returns

- Interpolated value(s) at `x`.

# See Also

- [`OBB_func`](@ref): Uses this to evaluate ``F(x)`` and ``J(x)``.
"""
function interp1(xspan, yspan, x)
    idx = (x .>= xspan[1]) .& (x .<= xspan[end])
    intf = interpolate((xspan,), yspan, Gridded(Linear()))
    y = intf[x[idx]]
    return y
end

@doc raw"""

    OBB(x::AbstractVector, p, dp, rho, drho, d2rho; LDtype=:SLD,
        eps=nothing, tol=1e-10, input_error=0.0, guard=nothing,
        refine=true, diagnostics=false)

Calculate the optimal biased bound (OBB) for single-parameter Bayesian
estimation. The bound follows from the first type of the Bayesian quantum
Cramér-Rao bound and is obtained by solving the associated Euler-Lagrange
boundary-value problem for the optimal bias function.

# Arguments

- `x`: The regimes of the parameter for the integral, normally given as a
  one-element vector of grid vectors, e.g. `[xgrid]`. A bare grid vector is
  also accepted.
- `p`: The prior distribution sampled on the grid.
- `dp`: Derivatives of the prior distribution with respect to the unknown
  parameter, sampled on the grid.
- `rho`: Parameterized density matrices, with one entry per grid point.
- `drho`: First derivatives of `rho` with respect to the unknown parameter,
  with one entry per grid point.
- `d2rho`: Second derivatives of `rho` with respect to the unknown parameter,
  with one entry per grid point.
- `LDtype`: Type of QFI (QFIM): `:SLD` (default), `:RLD`, or `:LLD`.
- `eps`: Legacy relative floor factor. When `nothing`, the relative precision
  policy is used; when given, it is deprecated and interpreted by the core.
- `tol`: Target relative accuracy of the inner QFIM evaluations.
- `input_error`: Relative error of the input data.
- `guard`: Relative spectral floor of the inner QFIM evaluations.
- `refine`: Whether one residual-refinement step is attempted in the inner
  SLD solves.
- `diagnostics`: When `true`, also return diagnostics for all grid points.

# Returns

- A real-valued `Float64` containing the integrated optimal biased bound.
- If `diagnostics = true`, returns `(value, diag)`, where `diag` is the
  [`NumericalDiagnostics`](@ref) record obtained by merging the diagnostics of
  all grid points.
raw"""
function OBB(
    x::AbstractVector,
    p,
    dp,
    rho,
    drho,
    d2rho;
    LDtype = :SLD,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    p_num = length(p)

    if typeof(drho[1]) == Vector{Matrix{ComplexF64}}
        drho = [drho[i][1] for i = 1:p_num]
    end
    if typeof(d2rho[1]) == Vector{Matrix{ComplexF64}}
        d2rho = [d2rho[i][1] for i = 1:p_num]
    end
    if typeof(dp[1]) == Vector{Float64}
        dp = [dp[i][1] for i = 1:p_num]
    end
    if typeof(x[1]) != Float64 && typeof(x[1]) != Int64
        x = x[1]
    end

    delta = x[2] - x[1]
    F, J = zeros(p_num), zeros(p_num)
    diags = NumericalDiagnostics[]
    for m = 1:p_num
        if diagnostics
            f, LD, diag = QFIM(
                rho[m],
                drho[m];
                LDtype = LDtype,
                exportLD = true,
                eps = eps,
                tol = tol,
                input_error = input_error,
                guard = guard,
                refine = refine,
                diagnostics = true,
            )
            push!(diags, diag)
        else
            f, LD = QFIM(
                rho[m],
                drho[m];
                LDtype = LDtype,
                exportLD = true,
                eps = eps,
                tol = tol,
                input_error = input_error,
                guard = guard,
                refine = refine,
            )
        end
        dF = real(tr(2 * d2rho[m] * LD - LD * LD * drho[m]))
        J[m] = dp[m] / p[m] - dF / f
        F[m] = f
    end

    prob = BVProblem(OBB_func, boundary_condition, [0.0, 0.0], (x[1], x[end]), (F, J, x))
    sol = solve(prob, GeneralMIRK4(), dt = delta)

    bias = [sol.u[i][1] for i = 1:p_num]
    dbias = [sol.u[i][2] for i = 1:p_num]

    value = [p[i] * ((1 + dbias[i])^2 / F[i] + bias[i]^2) for i = 1:p_num]
    result = simpson(x, value)
    return diagnostics ? (result, _merge_diagnostics(:OBB, diags)) : result
end
