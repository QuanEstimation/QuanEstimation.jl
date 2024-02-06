module QuanEstimationPyExt
using PythonCall
using QuanEstimation
np = pyimport("numpy")
# QuanEstimation.Htot(H0::PyArray, Hc::PyList, ctrl) = QuanEstimation.Htot(Matrix(H0), [Matrix(hc) for hc in Hc], ctrl)
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
end
