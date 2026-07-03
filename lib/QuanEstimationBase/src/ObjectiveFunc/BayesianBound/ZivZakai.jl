@doc raw"""
    trace_norm(X::AbstractMatrix{<:Number})

Compute the trace norm (Schatten 1-norm) of a matrix:

```math
\|X\|_1 = \mathrm{Tr}\,|X| = \sum_i \sigma_i(X),
```

where ``\sigma_i(X)`` are the singular values of ``X``.

# Arguments

- `X::AbstractMatrix{<:Number}`: Input matrix.

# Returns

- `Float64`: The trace norm ``\|X\|_1``.

# See Also

- [`trace_norm`](@ref): Two-argument form for the distance between two states.
- [`helstrom_bound`](@ref): Helstrom bound using trace norm.
raw"""
trace_norm(X::AbstractMatrix{<:Number}) = norm(X |> svdvals, 1)

@doc raw"""
    trace_norm(ρ::AbstractMatrix{<:Number}, σ::AbstractMatrix{<:Number})

Compute the trace distance between two density matrices:

```math
D_{\mathrm{tr}}(\rho, \sigma) = \frac{1}{2}\|\rho - \sigma\|_1.
```

Note: This function returns ``\|\rho-\sigma\|_1`` (NOT halved).

# Arguments

- `ρ::AbstractMatrix{<:Number}`: First density matrix.
- `σ::AbstractMatrix{<:Number}`: Second density matrix.

# Returns

- `Float64`: The trace distance ``\|\rho-\sigma\|_1``.

# See Also

- [`trace_norm`](@ref): Single-argument form.
- [`helstrom_bound`](@ref): Helstrom bound using trace norm.
"""
trace_norm(ρ::AbstractMatrix{<:Number}, σ::AbstractMatrix{<:Number}) = trace_norm(ρ - σ)

@doc raw"""
    fidelity(ρ::AbstractMatrix{<:Number}, σ::AbstractMatrix{<:Number})

Compute the Uhlmann fidelity between two density matrices:

```math
\mathcal{F}(\rho, \sigma) = \bigl(\mathrm{Tr}\,\sqrt{\sqrt{\rho}\,\sigma\sqrt{\rho}}\bigr)^2.
```

# Arguments

- `ρ::AbstractMatrix{<:Number}`: First density matrix.
- `σ::AbstractMatrix{<:Number}`: Second density matrix.

# Returns

- `Float64`: The fidelity ``\mathcal{F}(\rho,\sigma) \in [0,1]``.

# See Also

- [`fidelity`](@ref): Two-argument form for pure states.
- [`helstrom_bound`](@ref): Helstrom bound using fidelity for pure states.
raw"""
function fidelity(ρ::AbstractMatrix{<:Number}, σ::AbstractMatrix{<:Number})
    return (ρ |> sqrt) * σ * (ρ |> sqrt) |> sqrt |> tr |> real |> x -> x^2
end # fidelity for density matrixes

@doc raw"""
    fidelity(ψ::AbstractVector{<:Number}, ϕ::AbstractVector{<:Number})

Compute the fidelity between two pure states:

```math
\mathcal{F}(|\psi\rangle, |\phi\rangle) = |\langle\psi|\phi\rangle|^2.
```

# Arguments

- `ψ::AbstractVector{<:Number}`: First pure state ket.
- `ϕ::AbstractVector{<:Number}`: Second pure state ket.

# Returns

- `Float64`: The squared overlap ``|\langle\psi|\phi\rangle|^2``.

# See Also

- [`fidelity`](@ref): Two-argument form for density matrices.
"""
function fidelity(ψ::AbstractVector{<:Number}, ϕ::AbstractVector{<:Number})
    overlap = ψ'ϕ
    return overlap'overlap
end  # fidelity for pure states

@doc raw"""
    helstrom_bound(ρ::AbstractMatrix{<:Number}, σ::AbstractMatrix{<:Number}, ν=1, P0=0.5)
    helstrom_bound(ψ::AbstractVector{<:Number}, ϕ::AbstractVector{<:Number}, ν=1)

Compute the Helstrom bound for the minimum error probability in quantum
hypothesis testing between two states.

# Mathematical Definition

For density matrices ``\rho`` and ``\sigma`` with prior probabilities ``P_0``
and ``1-P_0``, the minimum error probability is lower-bounded by

```math
P_{\mathrm{err}} \ge \frac{1}{2}\bigl(1 - \|P_0\rho - (1-P_0)\sigma\|_1\bigr).
```

For ``\nu`` independent copies, the collective bound is

```math
P_{\mathrm{err}}^{(\nu)} \ge \frac{1}{2}\bigl(1 - \|P_0\rho^{\otimes\nu} - (1-P_0)\sigma^{\otimes\nu}\|_1\bigr).
```

For pure states, this simplifies via the fidelity:

```math
P_{\mathrm{err}}^{(\nu)} \ge \frac{1}{2}\bigl(1 - \sqrt{1-\mathcal{F}(|\psi\rangle,|\phi\rangle)^\nu}\bigr).
```

# Arguments

- `ρ, σ`: The two density matrices (or `ψ, ϕ` for pure states).
- `ν::Number=1`: Number of independent copies.
- `P0::Float64=0.5`: Prior probability for ``\rho``.

# Returns

- `Float64`: The Helstrom bound on the error probability.

# See Also

- [`trace_norm`](@ref): Trace norm used in the density-matrix formulation.
- [`fidelity`](@ref): Fidelity used in the pure-state formulation.
- [`QZZB`](@ref): Quantum Ziv-Zakai bound using the Helstrom bound.
"""
# Helstorm bound of error probability for the hypothesis testing problem 
function helstrom_bound(
    ρ::AbstractMatrix{<:Number},
    σ::AbstractMatrix{<:Number},
    ν = 1,
    P0 = 0.5,
)
    return (1 - trace_norm(P0 * ρ - (1 - P0) * σ)) / 2 |> real
end

raw"""
    helstrom_bound(ψ::AbstractVector{<:Number}, ϕ::AbstractVector{<:Number}, ν=1)

Pure-state version of the Helstrom bound. See the density-matrix
[`helstrom_bound`](@ref) for the mathematical definition.
raw"""
function helstrom_bound(ψ::AbstractVector{<:Number}, ϕ::AbstractVector{<:Number}, ν = 1)
    return (1 - sqrt(1 - fidelity(ψ, ϕ))^ν) / 2 |> real
end

@doc raw"""
    prior_uniform(W=1.0, μ=0.0)

Create a uniform prior distribution function over the interval
``[\mu-W/2,\;\mu+W/2]``.

Returns a closure ``p(x)`` that evaluates to ``1/W`` for ``|x-\mu| \le W/2``
and ``0`` otherwise.

# Arguments

- `W::Float64=1.0`: Width of the uniform distribution.
- `μ::Float64=0.0`: Center of the distribution.

# Returns

- `Function`: A function ``p(x)`` representing the uniform prior.

# See Also

- [`QZZB`](@ref): Quantum Ziv-Zakai bound using this prior.
"""
prior_uniform(W = 1.0, μ = 0.0) = x -> abs(x - μ) > abs(W / 2) ? 0 : 1 / W

"""

    QZZB(x::AbstractVector, p::AbstractVector, rho::AbstractVecOrMat; eps=GLOBAL_EPS)

Calculation of the quantum Ziv-Zakai bound (QZZB).
- `x`: The regimes of the parameters for the integral.
- `p`: The prior distribution.
- `rho`: Parameterized density matrix.
- `eps`: Machine epsilon.
"""
function QZZB(
    x::AbstractVector,
    p::AbstractVector,
    rho::AbstractVecOrMat;
    eps = GLOBAL_EPS,
    ν::Number = 1,
)
    if typeof(x[1]) == Vector{Float64} || typeof(x[1]) == Vector{Int64}
        x = x[1]
    end

    tau = x .- x[1]
    p_num = length(p)
    f_tau = zeros(p_num)
    for i = 1:p_num
        arr = [
            real(2 * minimum([p[j], p[j+i-1]]) * helstrom_bound(rho[j], rho[j+i-1], ν))
            for j = 1:p_num-i+1
        ]
        f_tp = trapz(x[1:p_num-i+1], arr)
        f_tau[i] = f_tp
    end
    arr2 = [tau[m] * maximum(f_tau[m:end]) for m = 1:p_num]
    I = trapz(tau, arr2)

    return 0.5 * I
end  # Quantum Ziv-Zakai bound for equally likely hypotheses with valley-filling

"""
    QZZB(scheme::AbstractScheme; ν::Number=1)

Compute the quantum Ziv-Zakai bound from an Estimation Scheme.

Evolves the state over the parameter region encoded in the scheme and calls
[`QZZB`](@ref) with the prior distribution.

# See Also

- [`QZZB`](@ref): Direct QZZB computation.
"""
function QZZB(scheme::AbstractScheme; ν::Number = 1)
    rho, _ = evolve_parameter_region(scheme)
    (; x, p) = getfield(scheme, :EstimationStrategy)
    return QZZB(x, p, rho, ν = ν)
end  # function QZZB
