module QuanEstimationPyExt
using PythonCall
using QuanEstimation
np = pyimport("numpy")

QuanEstimation.ControlOpt(ctrl::PyList, ctrl_bound::PyList, seed::Py) = QuanEstimation.ControlOpt(pyconvert(Vector{Vector{Float64}}, ctrl), pyconvert(Vector{Float64}, ctrl_bound), seed)

QuanEstimation.Lindblad(
	H0::PyArray,
	dH::PyList,
	Hc::PyList,
	ctrl::PyList,
	ρ0::PyArray,
	tspan::PyArray,
	decay_opt::PyList,
	γ::PyList;
	kwargs...,
) = QuanEstimation.Lindblad(
	pyconvert(Matrix{ComplexF64}, H0),
	pyconvert(Vector{Matrix}, dH),
	pyconvert(Vector{Matrix}, Hc),
	pyconvert(Vector{Vector{Float64}}, ctrl),
	pyconvert(Matrix{ComplexF64}, ρ0),
	pyconvert(Vector, tspan),
	pyconvert(Vector{Matrix}, decay_opt),
	pyconvert(Vector, γ);
	kwargs...,
)

QuanEstimation.Lindblad(
	H0::PyArray,
	dH::PyList,
	Hc::PyList,
	ctrl::PyList,
	ψ0::PyList,
	tspan::PyArray,
	decay_opt::PyList,
	γ::PyList;
	kwargs...,
) = QuanEstimation.Lindblad(
	pyconvert(Matrix{ComplexF64}, H0),
	pyconvert(Vector{Matrix}, dH),
	pyconvert(Vector{Matrix}, Hc),
	pyconvert(Vector{Vector{Float64}}, ctrl),
	pyconvert(Vector{ComplexF64}, ψ0),
	pyconvert(Vector, tspan),
	pyconvert(Vector{Matrix}, decay_opt),
	pyconvert(Vector, γ);
	kwargs...,
)

QuanEstimation.Lindblad(
	H0::PyArray,
	dH::PyList,
	ψ0::PyList,
	tspan::PyArray,
	decay_opt::PyList,
	γ::PyList;
	kwargs...,
) = QuanEstimation.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
	pyconvert(Vector{Matrix}, dH),
	pyconvert(Vector{ComplexF64}, ψ0),
	pyconvert(Vector, tspan),
	pyconvert(Vector{Matrix}, decay_opt),
	pyconvert(Vector, γ);
	kwargs...,
)

QuanEstimation.Lindblad(
	H0::PyArray,
	dH::PyList,
	ρ0::PyArray,
	tspan::PyArray,
	decay_opt::PyList,
	γ::PyList;
	kwargs...,
) = QuanEstimation.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
	pyconvert(Vector{Matrix}, dH),
	pyconvert(Matrix{ComplexF64}, ρ0),
	pyconvert(Vector, tspan),
	pyconvert(Vector{Matrix}, decay_opt),
	pyconvert(Vector, γ);
	kwargs...,
)

QuanEstimation.Lindblad(
	H0::PyArray,
	dH::PyList,
	ψ0::PyList,
	tspan::PyArray;
	kwargs...,
) = QuanEstimation.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
	pyconvert(Vector{Matrix}, dH),
	pyconvert(Vector{ComplexF64}, ψ0),
	pyconvert(Vector, tspan);
	kwargs...,
)

QuanEstimation.Lindblad(
	H0::PyArray,
	dH::PyList,
	ρ0::PyArray,
	tspan::PyArray;
	kwargs...,
) = QuanEstimation.Lindblad(
    pyconvert(Matrix{ComplexF64}, H0),
	pyconvert(Vector{Matrix}, dH),
	pyconvert(Matrix{ComplexF64}, ρ0),
	pyconvert(Vector, tspan);
	kwargs...,
)

QuanEstimation.Kraus(
	ρ0::PyArray,
	K::PyList,
	dK::PyList;
	kwargs...,
) = QuanEstimation.Kraus(
	pyconvert(Matrix{ComplexF64}, ρ0),
	pyconvert(Vector{Matrix{ComplexF64}}, K),
	pyconvert(Vector{Vector{Matrix{ComplexF64}}}, dK);
	kwargs...,
)

QuanEstimation.Kraus(
	ψ0::PyList,
	K::PyList,
	dK::PyList;
	kwargs...,
) = QuanEstimation.Kraus(
	pyconvert(Vector{ComplexF64}, ψ0),
	pyconvert(Vector{Matrix{ComplexF64}}, K),
	pyconvert(Vector{Vector{Matrix{ComplexF64}}}, dK);
	kwargs...,
)

QuanEstimation.expm_py(
    tspan::PyArray,
    ρ0::AbstractMatrix,
    H0::PyArray,
    dH::PyList,
    decay_opt::PyList,
    γ::PyList,
    Hc::PyList,
    ctrl::PyList,
) = QuanEstimation.expm_py(
	pyconvert(Vector, tspan),
	pyconvert(Matrix{ComplexF64}, ρ0),
	pyconvert(Matrix{ComplexF64}, H0),
	pyconvert(Vector{Matrix{ComplexF64}}, dH),
	pyconvert(Vector{Matrix{ComplexF64}}, decay_opt),
	pyconvert(Vector, γ),
	pyconvert(Vector{Matrix{ComplexF64}}, Hc),
	pyconvert(Vector{Vector{Float64}}, ctrl),
)


end
