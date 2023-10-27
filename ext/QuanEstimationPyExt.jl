module QuanEstimationPyExt
import PythonCall: PyArray
using QuanEstimation
QuanEstimation.Htot(H0::PyArray, Hc::PyList, ctrl) = QuanEstimation.Htot(Matrix(H0), [Matrix(hc) for hc in Hc], ctrl)
end