module QuanEstimationBasePyExt
using PythonCall
using QuanEstimationBase

QuanEstimationBase.ControlOpt(ctrl::PyList, ctrl_bound::PyList, seed::Py) =
    QuanEstimationBase.ControlOpt(
        pyconvert(Vector{Vector{Float64}}, ctrl),
        pyconvert(Vector{Float64}, ctrl_bound),
        seed,
    )

QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    Hc::PyList,
    ctrl::PyList,
    ρ0::PyArray,
    tspan::PyArray,
    decay_opt::PyList,
    γ::PyList;
    kwargs...,
) = QuanEstimationBase.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix}, dH),
    pyconvert(Vector{Matrix}, Hc),
    pyconvert(Vector{Vector{Float64}}, ctrl),
    pyconvert(Matrix{ComplexF64}, ρ0),
    pyconvert(Vector, tspan),
    pyconvert(Vector{Matrix{ComplexF64}}, decay_opt),
    pyconvert(Vector{Float64}, γ);
    kwargs...,
)

QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    Hc::PyList,
    ctrl::PyList,
    ψ0::PyList,
    tspan::PyArray,
    decay_opt::PyList,
    γ::PyList;
    kwargs...,
) = QuanEstimationBase.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix}, dH),
    pyconvert(Vector{Matrix}, Hc),
    pyconvert(Vector{Vector{Float64}}, ctrl),
    pyconvert(Vector{ComplexF64}, ψ0),
    pyconvert(Vector, tspan),
    pyconvert(Vector{Matrix{ComplexF64}}, decay_opt),
    pyconvert(Vector{Float64}, γ);
    kwargs...,
)

QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    ψ0::PyList,
    tspan::PyArray,
    decay_opt::PyList,
    γ::PyList;
    kwargs...,
) = QuanEstimationBase.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix}, dH),
    pyconvert(Vector{ComplexF64}, ψ0),
    pyconvert(Vector, tspan),
    pyconvert(Vector{Matrix{ComplexF64}}, decay_opt),
    pyconvert(Vector{Float64}, γ);
    kwargs...,
)

QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    ρ0::PyArray,
    tspan::PyArray,
    decay_opt::PyList,
    γ::PyList;
    kwargs...,
) = QuanEstimationBase.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix}, dH),
    pyconvert(Matrix{ComplexF64}, ρ0),
    pyconvert(Vector, tspan),
    pyconvert(Vector{Matrix{ComplexF64}}, decay_opt),
    pyconvert(Vector{Float64}, γ);
    kwargs...,
)

QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    ψ0::PyList,
    tspan::PyArray;
    kwargs...,
) = QuanEstimationBase.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix}, dH),
    pyconvert(Vector{ComplexF64}, ψ0),
    pyconvert(Vector{Float64}, tspan);
    kwargs...,
)

QuanEstimationBase.Lindblad(
    H0::PyArray,
    dH::PyList,
    ρ0::PyArray,
    tspan::PyArray;
    kwargs...,
) = QuanEstimationBase.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix}, dH),
    pyconvert(Matrix{ComplexF64}, ρ0),
    pyconvert(Vector{Float64}, tspan);
    kwargs...,
)

QuanEstimationBase.Kraus(ρ0::PyArray, K::PyList, dK::PyList; kwargs...) =
    QuanEstimationBase.Kraus(
        pyconvert(Matrix{ComplexF64}, ρ0),
        pyconvert(Vector{Matrix{ComplexF64}}, K),
        pyconvert(Vector{Vector{Matrix{ComplexF64}}}, dK);
        kwargs...,
    )

QuanEstimationBase.Kraus(ψ0::PyList, K::PyList, dK::PyList; kwargs...) =
    QuanEstimationBase.Kraus(
        pyconvert(Vector{ComplexF64}, ψ0),
        pyconvert(Vector{Matrix{ComplexF64}}, K),
        pyconvert(Vector{Vector{Matrix{ComplexF64}}}, dK);
        kwargs...,
    )

QuanEstimationBase.expm_py(
    tspan::PyArray,
    ρ0::AbstractMatrix,
    H0::PyArray,
    dH::PyList,
    decay_opt::PyList,
    γ::PyList,
    Hc::PyList,
    ctrl::PyList,
) = QuanEstimationBase.expm_py(
    pyconvert(Vector, tspan),
    pyconvert(Matrix{ComplexF64}, ρ0),
    pyconvert(Matrix{ComplexF64}, H0),
    pyconvert(Vector{Matrix{ComplexF64}}, dH),
    pyconvert(Vector{Matrix{ComplexF64}}, decay_opt),
    pyconvert(Vector{Float64}, γ),
    pyconvert(Vector{Matrix{ComplexF64}}, Hc),
    pyconvert(Vector{Vector{Float64}}, ctrl),
)

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
        [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec |> Array for i = 1:ctrl_num]
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
        [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec |> Array for i = 1:ctrl_num]
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

end
