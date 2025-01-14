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


end
