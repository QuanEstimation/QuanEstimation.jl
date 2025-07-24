# QuanEstimationBase

QuanEstimationBase.jl is the base package for QuanEstimation.jl.

## Installation
Generally, this package does not need to be installed independently. It would be automatcally installed 
as long as QuanEstimation.jl is installed. However, if someone does need to install it independently, 
just run the command in the julia REPL:  

~~~
julia > using Pkg 

julia > Pkg.add("NVMagnetometer")
~~~

## Citation
If you use QuanEstimation in your research, please cite the following papers:

[1] M. Zhang, H.-M. Yu, H. Yuan, X. Wang, R. Demkowicz-Dobrzański, and J. Liu, 
QuanEstimation: An open-source toolkit for quantum parameter estimation, 
[Phys. Rev. Res. **4**, 043057 (2022).](https://doi.org/10.1103/PhysRevResearch.4.043057)

[2] H.-Ming Yu and J. Liu, QuanEstimation.jl: An open-source Julia framework for quantum parameter estimation, 
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