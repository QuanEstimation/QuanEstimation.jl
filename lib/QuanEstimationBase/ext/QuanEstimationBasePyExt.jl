module QuanEstimationBasePyExt
using PythonCall
using QuanEstimationBase
using OrdinaryDiffEq: ODEProblem, solve, Tsit5
using LinearAlgebra
using Random

# Julia → Python: convert ComplexF64 matrices to numpy arrays directly.
# Without this rule, PythonCall wraps Matrix{ComplexF64} as AnyValue
# with a ._jl attribute, requiring manual np.array(obj._jl) in Python.
PythonCall.pyconvert(::Type{Py}, x::AbstractMatrix{ComplexF64}) =
    PythonCall.PyArray(x)

QuanEstimationBase.ControlOpt(ctrl::PyList, ctrl_bound::PyList, seed::Py) =
    QuanEstimationBase.ControlOpt(;
        ctrl = pyconvert(Vector{Vector{Float64}}, ctrl),
        ctrl_bound = pyconvert(Vector{Float64}, ctrl_bound),
        seed = pyconvert(Int, seed),
    )

# GeneralState: bridge for Python numpy arrays passed as probe
# (density matrix as 2D PyArray, pure state as 1D PyArray)
function QuanEstimationBase.GeneralState(arr::PyArray)
    if length(size(arr)) == 1
        m = pyconvert(Vector{ComplexF64}, arr)
    else
        m = pyconvert(Matrix{ComplexF64}, arr)
    end
    return QuanEstimationBase.GeneralState(m)
end

# Lindblad with control + decay (most common)
QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    tspan::PyArray,
    Hc::PyList,
    decay::PyList;
    ctrl::PyList = nothing,
    dyn_method::String = "Expm",
    kwargs...,
) = QuanEstimationBase.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix{ComplexF64}}, dH),
    pyconvert(Vector{Float64}, tspan),
    pyconvert(Vector{Matrix{ComplexF64}}, Hc),
    pyconvert(Vector, decay);
    ctrl = isnothing(ctrl) ? QuanEstimationBase.ZeroCTRL() : pyconvert(Vector{Vector{Float64}}, ctrl),
    dyn_method = Symbol(dyn_method),
    kwargs...,
)

# Lindblad with control + NO decay (4 positional)
QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    tspan::PyArray,
    Hc::PyList;
    ctrl::PyList = nothing,
    dyn_method::String = "Expm",
    kwargs...,
) = QuanEstimationBase.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix{ComplexF64}}, dH),
    pyconvert(Vector{Float64}, tspan),
    pyconvert(Vector{Matrix{ComplexF64}}, Hc);
    ctrl = isnothing(ctrl) ? QuanEstimationBase.ZeroCTRL() : pyconvert(Vector{Vector{Float64}}, ctrl),
    dyn_method = Symbol(dyn_method),
    kwargs...,
)

# Lindblad with optional decay + NO control (3 positional)
QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    tspan::PyArray;
    decay::Union{PyList,Nothing} = nothing,
    dyn_method::String = "Expm",
    kwargs...,
) = if isnothing(decay)
    QuanEstimationBase.Lindblad(
        pyconvert(Matrix{ComplexF64}, H0),
        pyconvert(Vector{Matrix{ComplexF64}}, dH),
        pyconvert(Vector{Float64}, tspan);
        dyn_method = Symbol(dyn_method),
        kwargs...,
    )
else
    QuanEstimationBase.Lindblad(
        pyconvert(Matrix{ComplexF64}, H0),
        pyconvert(Vector{Matrix{ComplexF64}}, dH),
        pyconvert(Vector{Float64}, tspan),
        pyconvert(Vector, decay);
        dyn_method = Symbol(dyn_method),
        kwargs...,
    )
end

# Kraus: now only takes (K, dK) — probe state passed via GeneralScheme
QuanEstimationBase.Kraus(K::PyList, dK::PyList; kwargs...) =
    QuanEstimationBase.Kraus(
        pyconvert(Vector{Matrix{ComplexF64}}, K),
        pyconvert(Vector{Vector{Matrix{ComplexF64}}}, dK);
        kwargs...,
    )

function QuanEstimationBase.expm_py(
    tspan::PyArray,
    ρ0::AbstractMatrix,
    H0::PyArray,
    dH::PyList,
    decay_opt::PyList,
    γ::PyList,
    Hc::PyList,
    ctrl::PyList,
)
    decay = [(pyconvert(Matrix{ComplexF64}, decay_opt[i]), pyconvert(Float64, γ[i])) for i in 1:length(γ)]
    return QuanEstimationBase.expm(
        pyconvert(Vector{Float64}, tspan),
        pyconvert(Matrix{ComplexF64}, ρ0),
        pyconvert(Matrix{ComplexF64}, H0),
        pyconvert(Vector{Matrix{ComplexF64}}, dH);
        decay = decay,
        Hc = pyconvert(Vector{Matrix{ComplexF64}}, Hc),
        ctrl = pyconvert(Vector{Vector{Float64}}, ctrl),
    )
end

function QuanEstimationBase.liouville_commu_py(A::Array{T}) where {T<:Complex}
    dim = size(A)[1]
    result = zeros(T, dim^2, dim^2)
    for i = 1:dim
        for j = 1:dim
            @inbounds for k = 1:dim
                ni = dim * (i - 1) + j
                nj = dim * (k - 1) + j
                nk = dim * (i - 1) + k

                result[ni, nj] = A[i, k]
                result[ni, nk] = -A[k, j]
                result[ni, ni] = A[i, i] - A[j, j]
            end
        end
    end
    result
end

function QuanEstimationBase.liouville_dissip_py(A::Array{T}) where {T<:Complex}
    dim = size(A)[1]
    result = zeros(T, dim^2, dim^2)
    for i = 1:dim
        for j = 1:dim
            ni = dim * (i - 1) + j
            for k = 1:dim
                @inbounds for l = 1:dim
                    nj = dim * (k - 1) + l
                    L_temp = A[i, k] * conj(A[j, l])
                    for p = 1:dim
                        L_temp -=
                            0.5 * float(k == i) * A[p, j] * conj(A[p, l]) +
                            0.5 * float(l == j) * A[p, k] * conj(A[p, i])
                    end
                    result[ni, nj] = L_temp
                end
            end
        end
    end
    result[findall(abs.(result) .< 1e-10)] .= 0.0
    result
end

# ode_py wrapper: converts Python args to Julia types, then dispatches
function QuanEstimationBase.ode_py(
    tspan::PyArray,
    ρ0::PyArray,
    H0::PyArray,
    dH::PyList,
    Γ::PyList,
    γ::PyList,
    Hc::PyList,
    ctrl::PyList,
)
    dH_jl = pyconvert(Vector{Matrix{ComplexF64}}, dH)
    tspan_jl = pyconvert(Vector{Float64}, tspan)
    ρ0_jl = pyconvert(Matrix{ComplexF64}, ρ0)
    H0_jl = pyconvert(Matrix{ComplexF64}, H0)
    Γ_jl = pyconvert(Vector{Matrix{ComplexF64}}, Γ)
    γ_jl = pyconvert(Vector{Float64}, γ)
    Hc_jl = pyconvert(Vector{Matrix{ComplexF64}}, Hc)
    ctrl_jl = pyconvert(Vector{Vector{Float64}}, ctrl)
    # Dispatch based on dH dimensionality
    if length(dH_jl) == 1 && dH_jl[1] isa AbstractMatrix
        # Single-parameter: delegate to multi-param with dH wrapped in a 1-element vector
        # so that ∂ρt returns Vector{Vector{Matrix}} (consistent with Python post-processing)
        return QuanEstimationBase.ode_py(tspan_jl, ρ0_jl, H0_jl, [dH_jl[1]], Γ_jl, γ_jl, Hc_jl, ctrl_jl)
    else
        # Multi-parameter: (tspan, ρ0, H0, dH::AbstractVector, Γ, γ, Hc, ctrl)
        return QuanEstimationBase.ode_py(tspan_jl, ρ0_jl, H0_jl, dH_jl, Γ_jl, γ_jl, Hc_jl, ctrl_jl)
    end
end

# Single-parameter ode_py with converted types
function QuanEstimationBase.ode_py(
    tspan,
    ρ0::AbstractMatrix,
    H0::AbstractVecOrMat,
    dH::AbstractMatrix,
    Hc::AbstractVector,
    Γ::AbstractVector,
    γ,
    ctrl::AbstractVector,
)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl =
        [repeat(ctrl[i]; inner=ctrl_interval) for i = 1:ctrl_num]
    push!.(ctrl, [0.0 for i = 1:ctrl_num])
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

    ∂ρt_func!(∂ρ, ctrl, t) =
        -im * (dH * ρt[t2Num(t)] - ρt[t2Num(t)] * dH) -
        im * (H(ctrl)[t2Num(t)] * ∂ρ - ∂ρ * H(ctrl)[t2Num(t)]) + (
            [
                γ[i] * (Γ[i] * ∂ρ * Γ[i]' - 0.5 * (Γ[i]' * Γ[i] * ∂ρ + ∂ρ * Γ[i]' * Γ[i]))
                for i in eachindex(Γ)
            ] |> sum
        )

    prob_∂ρ = ODEProblem(∂ρt_func!, ρ0 |> zero, (tspan[1], tspan[end]), ctrl)
    ∂ρt = solve(prob_∂ρ, Tsit5(), saveat = dt).u
    ρt, ∂ρt
end

function QuanEstimationBase.ode_py(
    tspan,
    ρ0::AbstractMatrix,
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Γ::AbstractVector,
    γ,
    Hc::AbstractVector,
    ctrl::AbstractVector,
)
    param_num = length(dH)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl =
        [repeat(ctrl[i]; inner=ctrl_interval) for i = 1:ctrl_num]
    push!.(ctrl, [0.0 for i = 1:ctrl_num])
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

# DE_deltaphiOpt: Python seed/target -> Julia RNG/Symbol
QuanEstimationBase.DE_deltaphiOpt(
    x::PyArray, p::PyArray, rho0::PyArray, comb::PyList,
    p_num, ini_population::PyList,
    c, cr, seed, max_episode, target::String, eps,
) = QuanEstimationBase.DE_deltaphiOpt(
    pyconvert(Vector{Float64}, x),
    pyconvert(Vector{Float64}, p),
    pyconvert(Matrix{ComplexF64}, rho0),
    pyconvert(Vector, comb),
    p_num,
    pyconvert(Vector, ini_population),
    c, cr,
    MersenneTwister(seed |> Int),
    max_episode,
    Symbol(target),
    eps,
)

# PSO_deltaphiOpt: Python seed/target -> Julia RNG/Symbol
QuanEstimationBase.PSO_deltaphiOpt(
    x::PyArray, p::PyArray, rho0::PyArray, comb::PyList,
    p_num, ini_particle::PyList,
    c0, c1, c2, seed, max_episode, target::String, eps,
) = QuanEstimationBase.PSO_deltaphiOpt(
    pyconvert(Vector{Float64}, x),
    pyconvert(Vector{Float64}, p),
    pyconvert(Matrix{ComplexF64}, rho0),
    pyconvert(Vector, comb),
    p_num,
    pyconvert(Vector, ini_particle),
    c0, c1, c2,
    MersenneTwister(seed |> Int),
    max_episode,
    Symbol(target),
    eps,
)

end # module QuanEstimationBasePyExt
