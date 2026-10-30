"""

    Bayes(x, p, rho, y; M=nothing, savefile=false)

Bayesian estimation. The prior distribution is updated via the posterior distribution obtained by the Bayes’ rule and the estimated value of parameters obtained via the maximum a posteriori probability (MAP).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho`: Parameterized density matrix.
- `y`: The experimental results obtained in practice.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `savefile`: Whether or not to save all the posterior distributions. 
"""
function Bayes(x, p, rho, y; M = nothing, estimator = "mean", savefile = false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        #### singleparameter senario ####
        if isnothing(M)
            M = SIC(size(rho[1])[1])
        end
        if savefile == false
            x_out = []
            if estimator == "mean"
                for mi = 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho .* [M[res_exp]]))
                    py = trapz(x[1], pyx .* p)
                    p_update = pyx .* p / py
                    p = p_update
                    append!(x_out, trapz(x[1], p .* x[1]))
                end
            elseif estimator == "MAP"
                for mi = 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho .* [M[res_exp]]))
                    py = trapz(x[1], pyx .* p)
                    p_update = pyx .* p / py
                    p = p_update

                    indx = findmax(p)[2]
                    append!(x_out, x[1][indx])
                end
            else
                throw(ArgumentError(
                    "The input is not a valid value for estimator, supported values are 'mean' and 'MAP'.",
                ))
            end

            jldopen("bayes.dat", "w") do f
                f["p"] = [p]
                f["x"] = x_out
            end

            # Save results to CSV
            # df = DataFrame(p = [p], x = x_out)
            # CSV.write("bayes.csv", df)

            return p, x_out[end]
        else
            p_out, x_out = [], []
            if estimator == "mean"
                for mi = 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho .* [M[res_exp]]))
                    py = trapz(x[1], pyx .* p)
                    p_update = pyx .* p / py
                    p = p_update

                    append!(p_out, [p])
                    append!(x_out, trapz(x[1], p .* x[1]))
                end
            elseif estimator == "MAP"
                for mi = 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho .* [M[res_exp]]))
                    py = trapz(x[1], pyx .* p)
                    p_update = pyx .* p / py
                    p = p_update

                    indx = findmax(p)[2]
                    append!(p_out, [p])
                    append!(x_out, x[1][indx])
                end
            else
                throw(ArgumentError(
                    "The input is not a valid value for estimator, supported values are 'mean' and 'MAP'.",
                ))
            end

            jldopen("bayes.dat", "w") do f
                f["p"] = [p_out]
                f["x"] = x_out
            end

            # Save results to CSV
            # df = DataFrame(p = [p_out], x = x_out)
            # CSV.write("bayes.csv", df)

            return p, x_out[end]
        end
    else
        #### multiparameter senario ####
        if isnothing(M)
            M = SIC(size(vec(rho)[1])[1])
        end
        if savefile == false
            x_out = []
            if estimator == "mean"
                for mi = 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho .* [M[res_exp]]))
                    arr = p .* pyx
                    py = trapz(tuple(x...), arr)

                    p_update = p .* pyx / py
                    p = p_update
                    append!(x_out, [integ(x, p)])
                end
            elseif estimator == "MAP"
                for mi = 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho .* [M[res_exp]]))
                    arr = p .* pyx
                    py = trapz(tuple(x...), arr)

                    p_update = p .* pyx / py
                    p = p_update

                    indx = findmax(p)[2]
                    append!(x_out, [[x[i][indx[i]] for i = 1:para_num]])
                end
            else
                throw(ArgumentError(
                    "The input is not a valid value for estimator, supported values are 'mean' and 'MAP'.",
                ))
            end

            jldopen("bayes.dat", "w") do f
                f["p"] = [p]
                f["x"] = x_out
            end

            # Save results to CSV
            # df = DataFrame(p = [p], x = x_out)
            # CSV.write("bayes.csv", df)

            return p, x_out[end]
        else
            p_out, x_out = [], []
            if estimator == "mean"
                for mi = 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho .* [M[res_exp]]))
                    arr = p .* pyx
                    py = trapz(tuple(x...), arr)
                    p_update = p .* pyx / py
                    p = p_update

                    append!(p_out, [p])
                    append!(x_out, [integ(x, p)])
                end
            elseif estimator == "MAP"
                for mi = 1:max_episode
                    res_exp = y[mi] |> Int
                    pyx = real.(tr.(rho .* [M[res_exp]]))
                    arr = p .* pyx
                    py = trapz(tuple(x...), arr)
                    p_update = p .* pyx / py
                    p = p_update

                    indx = findmax(p)[2]
                    append!(p_out, [p])
                    append!(x_out, [[x[i][indx[i]] for i = 1:para_num]])
                end
            else
                throw(ArgumentError(
                    "The input is not a valid value for estimator, supported values are 'mean' and 'MAP'.",
                ))
            end

            jldopen("bayes.dat", "w") do f
                f["p"] = p_out
                f["x"] = x_out
            end

            # Save results to CSV
            # df = DataFrame(p = p_out, x = x_out)
            # CSV.write("bayes.csv", df)

            return p, x_out[end]
        end
    end
end

"""

    MLE(x, rho, y; M=nothing, savefile=false)

Bayesian estimation. The estimated value of parameters obtained via the maximum likelihood estimation (MLE).
- `x`: The regimes of the parameters for the integral.
- `rho`: Parameterized density matrix.
- `y`: The experimental results obtained in practice.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `savefile`: Whether or not to save all the posterior distributions. 
raw"""
function MLE(x, rho, y; M = nothing, savefile = false)
    y = y .+ 1
    para_num = length(x)
    max_episode = length(y)
    if para_num == 1
        if isnothing(M)
            M = SIC(size(rho[1])[1])
        end
        if savefile == false
            x_out = []
            L_out = zeros(length(x[1]))
            for mi = 1:max_episode
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho .* [M[res_exp]]))
                p_safe = clamp.(p_tp, eps(Float64), Inf)
                L_out .+= log.(p_safe)
                indx = findmax(L_out)[2]
                append!(x_out, x[1][indx])
            end

            jldopen("MLE.dat", "w") do f
                f["L"] = [L_out]
                f["x"] = x_out
            end

            # Save results to CSV
            # df = DataFrame(L = [L_out], x = x_out)
            # CSV.write("MLE.csv", df)

            return L_out, x_out[end]
        else
            L_out, x_out = [], []
            L_tp = zeros(length(x[1]))
            for mi = 1:max_episode
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho .* [M[res_exp]]))
                p_safe = clamp.(p_tp, eps(Float64), Inf)
                L_tp .+= log.(p_safe)
                indx = findmax(L_tp)[2]
                append!(L_out, [L_tp])
                append!(x_out, x[1][indx])
            end

            jldopen("MLE.dat", "w") do f
                f["L"] = L_out
                f["x"] = x_out
            end

            # Save results to CSV
            # df = DataFrame(L = L_out, x = x_out)
            # CSV.write("MLE.csv", df)

            return L_tp, x_out[end]
        end
    else
        #### multiparameter senario ####
        p_shape = []
        for i = 1:para_num
            append!(p_shape, length(x[i]))
        end

        if isnothing(M)
            M = SIC(size(vec(rho)[1])[1])
        end

        if savefile == false
            x_out = []
            L_out = zeros(p_shape...)
            for mi = 1:max_episode
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho .* [M[res_exp]]))
                p_safe = clamp.(p_tp, eps(Float64), Inf)
                L_out .+= log.(p_safe)
                indx = findmax(L_out)[2]
                append!(x_out, [[x[i][indx[i]] for i = 1:para_num]])
            end
            jldopen("MLE.dat", "w") do f
                f["L"] = [L_out]
                f["x"] = x_out
            end
            
            # Save results to CSV
            # df = DataFrame(L = [L_out], x = x_out)
            # CSV.write("MLE.csv", df)

            return L_out, x_out[end]
        else
            L_out, x_out = [], []
            L_tp = zeros(p_shape...)
            for mi = 1:max_episode
                res_exp = y[mi] |> Int
                p_tp = real.(tr.(rho .* [M[res_exp]]))
                p_safe = clamp.(p_tp, eps(Float64), Inf)
                L_tp .+= log.(p_safe)
                indx = findmax(L_tp)[2]
                append!(L_out, [L_tp])
                append!(x_out, [[x[i][indx[i]] for i = 1:para_num]])
            end

            jldopen("MLE.dat", "w") do f
                f["L"] = L_out
                f["x"] = x_out
            end

            # Save results to CSV
            # df = DataFrame(L = L_out, x = x_out)
            # CSV.write("MLE.csv", df)

            return L_tp, x_out[end]
        end
    end
end

@doc raw"""
    integ(x, p)

Compute the posterior mean estimate for a multi-parameter Bayesian estimation
by marginalising the posterior distribution.

For ``d`` parameters with grids ``x_1,\dots,x_d``, the mean estimate for
parameter ``i`` is

```math
\hat{x}_i = \int x_i\,p(\mathbf{x})\,\prod_{k\neq i}\mathrm{d}x_k.
```

The function sequentially integrates out all dimensions except ``i`` using
`Trapz.trapz`, then computes ``\int x_i\,p_{\mathrm{marg}}(x_i)\,\mathrm{d}x_i``.

# Arguments

- `x::Vector{AbstractVector}`: Grid vectors for each parameter.
- `p::AbstractArray`: ``d``-dimensional posterior distribution array.

# Returns

- `Vector{Float64}`: Vector of mean estimates ``(\hat{x}_1,\dots,\hat{x}_d)``.

# See Also

- `Trapz.trapz`: Trapezoidal integration.
- [`Bayes`](@ref): Bayesian estimation using this for multi-parameter mean.
"""
function integ(x, p)
    para_num = length(x)
    mean = [0.0 for i = 1:para_num]
    for i = 1:para_num
        p_tp = p
        if i == para_num
            for si = 1:para_num-1
                p_tp = trapz(x[si], p_tp, Val(1))
            end

        elseif i == 1
            for si = para_num:2
                p_tp = trapz(x[si], p_tp)
            end
        else
            p_tp = trapz(x[end], p_tp)
            for si = 1:para_num-1
                p_tp = trapz(x[si], p_tp, Val(1))
            end
        end
        mean[i] = trapz(x[i], x[i] .* p_tp)
    end
    mean
end

"""
    BayesCost(x, p, xest, rho, M; W=nothing, eps=nothing)

Calculate the average Bayesian cost for a quadratic cost function,
``\\bar{C}=\\int p(\\mathbf{x})\\sum_y\\mathrm{Tr}(\\rho\\Pi_y)
(\\mathbf{x}-\\hat{\\mathbf{x}})^{\\mathrm{T}}W(\\mathbf{x}-\\hat{\\mathbf{x}})
\\,\\mathrm{d}\\mathbf{x}``.

# Arguments

- `x`: The regimes of the parameters for the integral, given as a vector of
  grid vectors (one per parameter).
- `p`: The prior distribution sampled on the grid, with shape `length.(x)`.
- `xest`: The estimators. `xest[j]` is the estimate associated with the
  `j`-th POVM element; each entry is a vector with one component per
  parameter (or a one-element vector for a single parameter).
- `rho`: Parameterized density matrices, with one entry per grid point.
- `M`: A set of POVMs; pass `nothing` to use a rank-one SIC-POVM.
- `W`: Weight matrix. For multiple parameters the default (`nothing`) uses
  the identity matrix; it is ignored in the single-parameter case.
- `eps`: Legacy relative spectral-floor factor. When `nothing`, the relative
  precision policy is used; when given, it is deprecated and interpreted by
  the core.

# Returns

- A real-valued `Float64` containing the average Bayesian cost.
"""
function BayesCost(x, p, xest, rho, M; W = nothing, eps = nothing)
    para_num = length(x)
    trapzm(x, integrands, slice_dim) = [
        trapz(tuple(x...), I) for
        I in [reshape(hcat(integrands...)[i, :], length.(x)...) for i = 1:slice_dim]
    ]

    if para_num == 1
        # single-parameter scenario
        if isnothing(M)
            M = SIC(size(rho[1])[1])
        end

        p_num = length(x[1])
        value = [
            p[i] * sum([
                tr(rho[i] * M[mi]) * (x[1][i] - xest[mi][1])^2 for mi in eachindex(M)
            ]) for i = 1:p_num
        ]
        return real(trapz(x[1], value))
    else
        # multi-parameter scenario
        if isnothing(W)
            W = Matrix(I, para_num, para_num)
        end

        if isnothing(M)
            M = SIC(size(vec(rho)[1])[1])
        end

        x_list = Iterators.product(x...)
        p_num = length(x_list)
        xCx = [
            sum([
                tr(rho_i * M[mi]) * ([xi...] - xest[mi])' * W * ([xi...] - xest[mi]) for
                mi in eachindex(M)
            ]) for (xi, rho_i) in zip(x_list, rho)
        ]
        xCx = reshape(xCx, size(p))

        value = p .* xCx
        for si in reverse(1:para_num)
            value = trapz(x[si], value)
        end
        return real(value)
    end
end

"""
    _second_difference_error(x, y)

Practical estimate of the trapezoidal-rule truncation bias on a
near-uniform grid `x`:

```math
\\Delta_{\\mathrm{int}} \\approx \\frac{h^2}{12}
\\sum_i |y_{i-1} - 2y_i + y_{i+1}|,
\\qquad h = \\frac{x_n - x_1}{n - 1}.
```

Returns `0.0` when fewer than three samples are available.
"""
function _second_difference_error(x::AbstractVector, y::AbstractVector)
    n = length(y)
    n < 3 && return 0.0
    h = (x[end] - x[1]) / (n - 1)
    s = 0.0
    @inbounds for i = 2:n-1
        s += abs(y[i-1] - 2 * y[i] + y[i+1])
    end
    return h^2 / 12 * s
end

"""
    _second_difference_error_nd(x, A)

Sum of [`_second_difference_error`](@ref) over the axes of the array `A`,
whose shape is `length.(x)`.
"""
function _second_difference_error_nd(x, A::AbstractArray)
    err = 0.0
    for d = 1:ndims(A)
        n = size(A, d)
        n < 3 && continue
        h = (x[d][end] - x[d][1]) / (n - 1)
        err += h^2 / 12 * sum(abs, diff(diff(A, dims = d), dims = d))
    end
    return err
end

"""
    _integration_error_matrix(x, integrands, ncomp)

Maximum second-difference integration error over the `ncomp` components of
a `trapzm`-style integrand collection.
"""
function _integration_error_matrix(x, integrands, ncomp::Integer)
    mat = hcat(integrands...)
    return maximum(
        _second_difference_error_nd(x, reshape(mat[i, :], length.(x)...)) for i = 1:ncomp
    )
end

"""
    _integration_diagnostics(spec, sld_diag; quantity, truncation_bias, deleted_mass, scale)

Combine the kernel diagnostics of the ``\\Lambda`` solve with the numerical
integration error estimate of a Bayesian bound.

The integration bias is reported in `truncation_bias_bound` (the channel it
belongs to), the eigenvalue mass removed by the relative spectral floor in
`deleted_mass`, and the dominant channel is re-classified with
[`classify_bottleneck`](@ref).
"""
function _integration_diagnostics(
    spec::PrecisionSpec,
    sld_diag::NumericalDiagnostics;
    quantity::Symbol = :BCB,
    truncation_bias::Real = 0.0,
    deleted_mass::Real = 0.0,
    scale::Real = 1.0,
)
    flags = copy(sld_diag.flags)
    if truncation_bias > spec.tol * max(Float64(scale), 1.0)
        push!(flags, :truncation_nonnegligible)
    end
    unique!(flags)
    return NumericalDiagnostics(
        quantity;
        precision_layer = sld_diag.precision_layer,
        spectral_floor = sld_diag.spectral_floor,
        numerical_rank = sld_diag.numerical_rank,
        backward_error = sld_diag.backward_error,
        forward_error_bound = sld_diag.forward_error_bound,
        input_error_propagation = sld_diag.input_error_propagation,
        truncation_bias_bound = truncation_bias,
        condition_number = sld_diag.condition_number,
        branch = sld_diag.branch,
        cross_validation_gap = sld_diag.cross_validation_gap,
        upper_bound = sld_diag.upper_bound,
        deleted_mass = deleted_mass,
        flags = flags,
        bottleneck = classify_bottleneck(
            ;
            forward_error_bound = sld_diag.forward_error_bound,
            input_error_propagation = sld_diag.input_error_propagation,
            truncation_bias_bound = truncation_bias,
        ),
    )
end

"""
    _merge_diagnostics(quantity, diags)

Merge a collection of [`NumericalDiagnostics`](@ref) records into one
record for `quantity`: worst-case error channels, union of the flags and
the joint bottleneck.
"""
function _merge_diagnostics(quantity::Symbol, diags::AbstractVector{<:NumericalDiagnostics})
    isempty(diags) && return NumericalDiagnostics(quantity)
    flags = Symbol[]
    for d in diags
        append!(flags, d.flags)
    end
    unique!(flags)
    layer = all(d.precision_layer === :compensated for d in diags) ? :compensated : :float64
    fwd = maximum(d.forward_error_bound for d in diags)
    inp = maximum(d.input_error_propagation for d in diags)
    trc = maximum(d.truncation_bias_bound for d in diags)
    return NumericalDiagnostics(
        quantity;
        precision_layer = layer,
        spectral_floor = maximum(d.spectral_floor for d in diags),
        numerical_rank = minimum(d.numerical_rank for d in diags),
        backward_error = maximum(d.backward_error for d in diags),
        forward_error_bound = fwd,
        input_error_propagation = inp,
        truncation_bias_bound = trc,
        condition_number = maximum(d.condition_number for d in diags),
        branch = diags[1].branch,
        cross_validation_gap = maximum(d.cross_validation_gap for d in diags),
        upper_bound = maximum(d.upper_bound for d in diags),
        deleted_mass = sum(d.deleted_mass for d in diags),
        flags = flags,
        bottleneck = classify_bottleneck(
            ;
            forward_error_bound = fwd,
            input_error_propagation = inp,
            truncation_bias_bound = trc,
        ),
    )
end

@doc raw"""

    BCB(x, p, rho; W=nothing, eps=nothing, tol=1e-10, input_error=0.0,
        guard=nothing, refine=true, diagnostics=false)

Calculate the minimum Bayesian cost bound for a quadratic cost function.
- `x`: The regimes of the parameters for the integral, given as a vector of
  grid vectors (one per parameter).
- `p`: The prior distribution sampled on the grid, with shape `length.(x)`.
- `rho`: Parameterized density matrices, with one entry per grid point.
- `W`: Weight matrix. For multiple parameters the default (`nothing`) uses
  the identity matrix; it is ignored in the single-parameter case.
- `eps`: Legacy relative floor factor; when `nothing`, the relative precision
  policy is used; when given, it is deprecated and interpreted by the core.
- `tol`: Target relative accuracy of the bound.
- `input_error`: Relative error of the input data.
- `guard`: Relative spectral floor (in units of the largest eigenvalue).
- `refine`: Whether one residual-refinement step is attempted in the
  ``\Lambda`` solve.
- `diagnostics`: When `true`, return `(value, diag)` where `diag` is a
  [`NumericalDiagnostics`](@ref) record. Its `truncation_bias_bound` collects
  the trapezoidal-rule estimate ``h^2/12\sum|\Delta^2 f|`` over every
  integrand, its `deleted_mass` is the eigenvalue mass removed by the
  relative spectral floor of the ``\Lambda`` solve, and its `bottleneck`
  reports the dominant channel among rounding, input-data and truncation
  errors. The numerical formula of the bound is unchanged.

# Returns

- A real-valued bound: a `Float64` in the single-parameter case and the
  integrated minimum cost bound in the multiple-parameter case.
- If `diagnostics = true`, returns `(value, diag)`.
raw"""
function BCB(
    x,
    p,
    rho;
    W = nothing,
    eps = nothing,
    tol = 1e-10,
    input_error = 0.0,
    guard = nothing,
    refine = true,
    diagnostics = false,
)
    para_num = length(x)
    trapzm(x, integrands, slice_dim) = [
        trapz(tuple(x...), I) for
        I in [reshape(hcat(integrands...)[i, :], length.(x)...) for i = 1:slice_dim]
    ]
    spec = precision_spec(
        eps;
        input_error = input_error,
        tol = tol,
        guard = guard,
        refine = refine,
    )

    if para_num == 1
        # single-parameter scenario
        dim = size(rho[1])[1]
        p_num = length(x[1])
        value = [p[i] * x[1][i]^2 for i = 1:p_num]
        delta2_x = trapz(x[1], value)
        err_delta = diagnostics ? _second_difference_error(x[1], value) : 0.0
        err_rho = 0.0
        err_pri = 0.0
        rho_avg = zeros(ComplexF64, dim, dim)
        rho_pri = zeros(ComplexF64, dim, dim)

        for di = 1:dim
            for dj = 1:dim
                rho_avg_arr = [p[m] * rho[m][di, dj] for m = 1:p_num]
                rho_pri_arr = [p[n] * x[1][n] * rho[n][di, dj] for n = 1:p_num]
                rho_avg[di, dj] = trapz(x[1], rho_avg_arr)
                rho_pri[di, dj] = trapz(x[1], rho_pri_arr)
                if diagnostics
                    err_rho = max(err_rho, _second_difference_error(x[1], rho_avg_arr))
                    err_pri = max(err_pri, _second_difference_error(x[1], rho_pri_arr))
                end
            end
        end

        if diagnostics
            Lambda, sld_diag = Lambda_avg(
                rho_avg,
                rho_pri;
                eps = eps,
                tol = tol,
                input_error = input_error,
                guard = guard,
                refine = refine,
                diagnostics = true,
            )
        else
            Lambda = Lambda_avg(
                rho_avg,
                rho_pri;
                eps = eps,
                tol = tol,
                input_error = input_error,
                guard = guard,
                refine = refine,
            )
        end
        minBC = delta2_x - real(tr(rho_avg * Lambda * Lambda))
        diagnostics || return minBC

        ss = _spectral_solver(rho_avg; spec = spec)
        deleted_mass = sum(abs(λ) for λ in ss.λ if abs(λ) <= ss.τ; init = 0.0)
        δΛ = (2 * err_pri + 2 * opnorm(Lambda) * err_rho) / ss.sep
        truncation_bias =
            err_delta +
            (2 * opnorm(Lambda) * opnorm(rho_avg) * δΛ + opnorm(Lambda)^2 * err_rho) +
            sld_diag.truncation_bias_bound
        diag = _integration_diagnostics(
            spec,
            sld_diag;
            quantity = :BCB,
            truncation_bias = truncation_bias,
            deleted_mass = deleted_mass,
            scale = max(abs(delta2_x), abs(minBC), 1.0),
        )
        return minBC, diag
    else
        # multi-parameter scenario
        if isnothing(W)
            W = Matrix(I, para_num, para_num)
        end
        x_list = Iterators.product(x...)
        p_num = length(x_list)
        xCx = [[xi...]' * W * [xi...] for xi in x_list]
        xCx = reshape(xCx, size(p))

        delta2_x = p .* xCx
        err_delta = diagnostics ? _second_difference_error_nd(x, delta2_x) : 0.0
        for si in reverse(1:para_num)
            delta2_x = trapz(x[si], delta2_x)
        end

        term_tp = [pp * rho_i |> vec for (pp, rho_i) in zip(p, rho)]
        err_rho = diagnostics ? _integration_error_matrix(x, term_tp, para_num^2) : 0.0
        rho_avg = trapzm(x, term_tp, para_num^2) |> I -> reshape(I, para_num, para_num)

        x_re = Vector{Float64}[]
        for xj in x_list
            append!(x_re, [[_ for _ in xj]])
        end

        pri_integrands = [
            term_tp .* reshape([x_re[i][para_i] for i = 1:p_num], size(p)) for
            para_i = 1:para_num
        ]
        err_pri = diagnostics ? maximum(
            _integration_error_matrix(x, integrands, para_num^2) for integrands in pri_integrands
        ) : 0.0
        rho_pri = [
            trapzm(x, integrands, para_num^2) |> I -> reshape(I, para_num, para_num) for
            integrands in pri_integrands
        ]

        if diagnostics
            Lambda, sld_diag = Lambda_avg(
                rho_avg,
                rho_pri;
                eps = eps,
                tol = tol,
                input_error = input_error,
                guard = guard,
                refine = refine,
                diagnostics = true,
            )
        else
            Lambda = Lambda_avg(
                rho_avg,
                rho_pri;
                eps = eps,
                tol = tol,
                input_error = input_error,
                guard = guard,
                refine = refine,
            )
        end

        Mat = zeros(ComplexF64, para_num, para_num)
        for para_m = 1:para_num
            for para_n = 1:para_num
                Mat += W[para_m, para_n] * (Lambda[para_m] * Lambda[para_n])
            end
        end

        minBC = delta2_x - real(tr(rho_avg * Mat))
        diagnostics || return minBC

        ss = _spectral_solver(rho_avg; spec = spec)
        deleted_mass = sum(abs(λ) for λ in ss.λ if abs(λ) <= ss.τ; init = 0.0)
        Λmax = maximum(opnorm, Lambda)
        Wsum = sum(abs, W)
        δΛ = (2 * err_pri + 2 * Λmax * err_rho) / ss.sep
        truncation_bias =
            err_delta +
            Wsum * (2 * Λmax * opnorm(rho_avg) * δΛ + Λmax^2 * err_rho) +
            sld_diag.truncation_bias_bound
        diag = _integration_diagnostics(
            spec,
            sld_diag;
            quantity = :BCB,
            truncation_bias = truncation_bias,
            deleted_mass = deleted_mass,
            scale = max(abs(delta2_x), abs(minBC), 1.0),
        )
        return minBC, diag
    end
end


@doc raw"""
    Lambda_avg(rho_avg::Matrix{T}, rho_pri::Matrix{T}; eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, refine=true,
        diagnostics=false) where {T<:Complex}

Compute the average cost operator ``\Lambda`` for the minimum Bayesian cost
bound (BCB).

Solves the Lyapunov-type equation

```math
\frac{1}{2}\bigl(\bar{\rho}\Lambda + \Lambda\bar{\rho}\bigr) = \bar{\rho}_{\mathrm{pri}},
```

where ``\bar{\rho} = \int p(x)\rho(x)\,\mathrm{d}x`` is the prior-averaged
density matrix and ``\bar{\rho}_{\mathrm{pri}} = \int p(x)\,x\,\rho(x)\,\mathrm{d}x``
is the prior-weighted average. This is analogous to the SLD equation with
``\bar{\rho}_{\mathrm{pri}}`` playing the role of ``\partial\rho``.

The equation is solved by the guarded SLD kernel [`_sld_solve`](@ref) with a
relative spectral floor; eigenblocks whose eigenvalue sum falls below the
floor are dropped (minimum-norm convention) and reported in the diagnostics.

# Arguments

- `rho_avg::Matrix{T}`: Prior-averaged density matrix ``\bar{\rho}``.
- `rho_pri::Matrix{T}`: Prior-weighted density matrix ``\bar{\rho}_{\mathrm{pri}}``.
- `eps=nothing`: Legacy keyword; when `nothing`, the new relative precision policy is used; when given, it is deprecated and interpreted as a relative spectral floor.
- `tol=1e-10`: Target relative accuracy.
- `input_error=0.0`: Relative error of the input data.
- `guard=nothing`: Relative spectral floor (in units of the largest eigenvalue).
- `refine=true`: Whether one residual-refinement step is attempted.
- `diagnostics=false`: When `true`, return `(Λ, diag)` with `diag` a [`NumericalDiagnostics`](@ref) record.

# Returns

- `Matrix{ComplexF64}`: The average cost operator ``\Lambda`` (or the tuple
  `(Λ, diag)` when `diagnostics=true`).

# See Also

- [`Lambda_avg`](@ref): Multi-parameter dispatch.
- [`BCB`](@ref): Bayesian cost bound using ``\Lambda``.
raw"""
function Lambda_avg(
    rho_avg::Matrix{T},
    rho_pri::Matrix{T};
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
    )
    sol = _sld_solve(rho_avg, rho_pri; spec = spec)
    return diagnostics ? (sol.L, sol.diag) : sol.L
end

@doc raw"""
    Lambda_avg(rho_avg::Matrix{T}, rho_pri::Vector{Matrix{T}}; eps=nothing,
        tol=1e-10, input_error=0.0, guard=nothing, refine=true,
        diagnostics=false) where {T<:Complex}

Multi-parameter version of [`Lambda_avg`](@ref). Computes one ``\Lambda_a``
operator per parameter by solving the corresponding guarded SLD equation for
each ``\bar{\rho}_{\mathrm{pri}}^{(a)}`` matrix.

# Arguments

- `rho_avg::Matrix{T}`: Prior-averaged density matrix ``\bar{\rho}``.
- `rho_pri::Vector{Matrix{T}}`: Vector of prior-weighted matrices
  ``\bar{\rho}_{\mathrm{pri}}^{(a)}``, one per parameter.
- `eps=nothing`: Legacy keyword; when `nothing`, the new relative precision policy is used; when given, it is deprecated and interpreted as a relative spectral floor.
- `tol=1e-10`: Target relative accuracy.
- `input_error=0.0`: Relative error of the input data.
- `guard=nothing`: Relative spectral floor (in units of the largest eigenvalue).
- `refine=true`: Whether one residual-refinement step is attempted.
- `diagnostics=false`: When `true`, return `(Λ, diag)` where the diagnostics of
  all parameters are merged into a single [`NumericalDiagnostics`](@ref) record.

# Returns

- `Vector{Matrix{ComplexF64}}`: Vector of ``\Lambda_a`` operators (or the
  tuple `(Λ, diag)` when `diagnostics=true`).

# See Also

- [`Lambda_avg`](@ref): Single-parameter dispatch.
- [`SLD`](@ref): The underlying SLD solver used for each parameter.
- [`BCB`](@ref): Bayesian cost bound.
raw"""
function Lambda_avg(
    rho_avg::Matrix{T},
    rho_pri::Vector{Matrix{T}};
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
    )
    sols = [_sld_solve(rho_avg, x; spec = spec) for x in rho_pri]
    Lambda = [sol.L for sol in sols]
    diagnostics || return Lambda
    return Lambda, _merge_diagnostics(:Lambda, [sol.diag for sol in sols])
end
