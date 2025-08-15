# QuanEstimation.jl
![GitHub release (latest by date)](https://img.shields.io/github/v/release/QuanEstimation/QuanEstimation.jl?label=version)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![][action-img]][action-url]
[![][codecov-img]][codecov-url]
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FQuanEstimation&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/QuanEstimation)

[action-img]: https://github.com/QuanEstimation/QuanEstimation.jl/actions/workflows/CI_QuanEstimation.yml/badge.svg
[action-url]: https://github.com/QuanEstimation/QuanEstimation.jl/actions
[codecov-img]: https://codecov.io/gh/QuanEstimation/QuanEstimation.jl/graph/badge.svg
[codecov-url]: https://codecov.io/gh/QuanEstimation/QuanEstimation.jl


QuanEstimation.jl is an open-source toolkit for quantum parameter estimation, which can be used to perform general evaluations of many metrological 
tools and scheme designs in quantum parameter estimation. 

This package is also the Julia implementation of [QuanEstimation](https://github.com/QuanEstimation/QuanEstimation).

## Installation

Run the command in the julia REPL to install QuanEstimation:  

~~~
julia > using Pkg 

julia > Pkg.add("QuanEstimation")
~~~

## Documentation
[![][docs-img]][docs-url]

[docs-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-url]: https://quanestimation.github.io/QuanEstimation/

The documentation for both QuanEstimation and QuanEstimation.jl is [here](https://quanestimation.github.io/QuanEstimation/). An independent documentation 
for QuanEstimation is [here](https://quanestimation.github.io/QuanEstimation.jl/), which is still under construction and we will make it functional 
as soon as we can. 

## Citation
If you use QuanEstimation in your research, please cite the following papers:

[1] M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and J. Liu, 
QuanEstimation: An open-source toolkit for quantum parameter estimation, 
[Phys. Rev. Res. **4**, 043057 (2022).](https://doi.org/10.1103/PhysRevResearch.4.043057)

[2] H.-M. Yu and J. Liu, QuanEstimation.jl: An open-source Julia framework for quantum parameter estimation, 
[Fundam. Res. (2025).](https://doi.org/10.1016/j.fmre.2025.02.020)

* Development of the GRAPE algorithm:

  * **auto-GRAPE**:
  
    M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and J. Liu,  
    QuanEstimation: An open-source toolkit for quantum parameter estimation,  
    [Phys. Rev. Res. **4**, 043057 (2022).](https://doi.org/10.1103/PhysRevResearch.4.043057)

  * **GRAPE for single-parameter estimation**:
  
    J. Liu and H. Yuan, Quantum parameter estimation with optimal control,  
    [Phys. Rev. A **96**, 012117 (2017).](https://doi.org/10.1103/PhysRevA.96.012117)

  * **GRAPE for multiparameter estimation**:
  
    J. Liu and H. Yuan, Control-enhanced multiparameter quantum estimation,  
    [Phys. Rev. A **96**, 042114 (2017).](https://doi.org/10.1103/PhysRevA.96.042114)
