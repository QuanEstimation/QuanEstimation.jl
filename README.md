# QuanEstimation.jl

[![][docs-img]][docs-url]
[![][action-img]][action-url]
[![][codecov-img]][codecov-url]


QuanEstimation is a Python-Julia based open-source toolkit for quantum parameter estimation, which consist in the calculation of the quantum metrological tools and quantum resources, and the optimizations with respect to probe states, controls or measurements, as well as comprehensive optimizations in quantum metrology. Futhermore, QuanEstimation can generate not only optimal quantum parameter estimation schemes, but also adaptive measurement schemes.

This package is a Julia implementation of [QuanEstimation](https://github.com/QuanEstimation/QuanEstimation).

## :warning: This package is under structural refactoring
It may exhibit unstable features; please use it with caution for now.

## Installation

Run the command in the julia REPL to install QuanEstimation:  

~~~
import Pkg; Pkg.add("QuanEstimation")
~~~

## Citation
If you find QuanEstimation useful in your research, feel free to cite the following papers:

[1] M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and J. Liu, 
QuanEstimation: An open-source toolkit for quantum parameter estimation, 
[Phys. Rev. Research **4**, 043057 (2022).](https://doi.org/10.1103/PhysRevResearch.4.043057)

[2] Huai-Ming Yu and Jing Liu, QuanEstimation.jl: An open-source Julia framework for quantum parameter estimation, 
[arXiv: 2405.12066.](https://doi.org/10.48550/arXiv.2405.12066)


[action-img]: https://github.com/QuanEstimation/QuanEstimation.jl/actions/workflows/CI.yml/badge.svg
[action-url]: https://github.com/QuanEstimation/QuanEstimation.jl/actions
[codecov-img]: https://codecov.io/gh/QuanEstimation/QuanEstimation.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/QuanEstimation/QuanEstimation.jl?branch=test-codecov
[docs-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-url]: https://quanestimation.github.io/QuanEstimation/
