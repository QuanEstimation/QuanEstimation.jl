# QuanEstimation.jl

QuanEstimation.jl is an open-source Julia framework for scheme evaluation and design
in quantum parameter estimation. It can be used either as an independent package or
as the computational core of the
[Python-Julia package](https://github.com/QuanEstimation/QuanEstimation).

## Key Features

- Dynamics evolution (matrix exponential / ODE) with derivatives
- Kraus operator parameterization
- Quantum and classical Fisher information (QFI, QFIM, CFI, CFIM)
- Asymptotic bounds: SLD, RLD, LLD, HCRB, NHB
- Bayesian bounds: BCRB, BQCRB, VTB, QVTB, QZZB, OBB
- Bayesian estimation and cost bounds
- Control, state, measurement, and comprehensive optimization
- Adaptive measurement schemes
- Metrological resources: spin squeezing, minimum-time search

## Installation

```julia
using Pkg
Pkg.add("QuanEstimation")
```

If you need to install via a Julia mirror, see the
[CERNET Julia mirror](https://help.mirrors.cernet.edu.cn/julia/) for usage.

### Requirements

QuanEstimation.jl requires Julia >= 1.10. The following packages are
automatically installed as dependencies:

| Package | Version | Package | Version |
|---------|---------|---------|---------|
| LinearAlgebra | -- | BoundaryValueDiffEq | 2.7.2 |
| Zygote | 0.6.37 | SCS | 0.8.1 |
| Convex | 0.14.18 | Trapz | 2.0.3 |
| ReinforcementLearning | 0.10.0 | Interpolations | 0.13.5 |
| IntervalSets | 0.5.4 | SparseArrays | -- |
| Flux | 0.12.4 | DelimitedFiles | -- |
| StatsBase | 0.33.16 | Random | -- |
| Printf | -- | StableRNGs | -- |
| Distributions | -- | QuadGK | -- |
| DifferentialEquations | -- | | |

## Quick Start

```julia
using QuanEstimation

# Initial state
rho0 = 0.5 * ones(2, 2)

# Free Hamiltonian and derivatives
sz = [1. 0.0im; 0. -1.]
H0 = 0.5 * sz
dH = [0.5 * sz]

# Dynamics
tspan = range(0.0, 10.0, length=2500)
rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH)

# Quantum Fisher information
F = QuanEstimation.QFIM(rho[end], drho[end])
```

## Users Guide

- [Parameterization](guide/guide_dynamics.md) — define system dynamics
    (Hamiltonian, decay, Kraus operators)
- [Metrological Tools](guide/guide_bounds.md) — Fisher information,
    Cramér-Rao bounds, Bayesian estimation
- [Control Optimization](guide/guide_Copt.md) — GRAPE, PSO, DE algorithms
- [State Optimization](guide/guide_Sopt.md) — AD, PSO, DE, Nelder-Mead, RI
- [Measurement Optimization](guide/guide_Mopt.md) — projection, linear
    combination, rotation
- [Comprehensive Optimization](guide/guide_Compopt.md) — joint optimization
    of state, control, and measurement
- [Adaptive Schemes](guide/guide_adaptive.md) — online/offline adaptive
    measurement
- [Resources](guide/guide_resources.md) — spin squeezing, minimum time
- [Output Files](guide/output_files.md) — file formats and post-processing

## API Reference

- [General API](api/GeneralAPI.md)
- [Base API](api/BaseAPI.md)
- [NV Magnetometer API](api/NVMagnetometerAPI.md)
- [Examples](https://github.com/QuanEstimation/QuanEstimation.jl/tree/main/examples)

## Citing QuanEstimation.jl

If you use QuanEstimation.jl in your research, please cite:

[1] M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and J. Liu,
QuanEstimation: An open-source toolkit for quantum parameter estimation,
[Phys. Rev. Res. **4**, 043057 (2022).](https://doi.org/10.1103/PhysRevResearch.4.043057)

[2] H.-M. Yu and J. Liu, QuanEstimation.jl: An open-source Julia framework for
quantum parameter estimation,
[Fundam. Res. (2025).](https://doi.org/10.1016/j.fmre.2025.02.020)

## Contributing

We welcome contributions! Please see our
[GitHub repository](https://github.com/QuanEstimation/QuanEstimation.jl)
for contribution guidelines.
