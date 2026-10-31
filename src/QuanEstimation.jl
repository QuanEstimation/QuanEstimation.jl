module QuanEstimation
using Reexport
@reexport using QuanEstimationBase
@reexport using NVMagnetometer

# Re-bake Base workloads after NVMagnetometer has installed its methods.
include("precompile_workload.jl")

end
