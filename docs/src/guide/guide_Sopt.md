# **State optimization**
For state optimization in QuanEstimation, the probe state is expanded as
$|\psi\rangle=\sum_i c_i|i\rangle$ in a specific basis $\{|i\rangle\}$. Thus, search of the
optimal probe states is equal to search of the normalized complex coefficients $\{c_i\}$. In
QuanEstimation, the state optimization algorithms are the automatic differentiation (AD)
[[1]](#Baydin2018), reverse iterative (RI)[[2]](#Rafal2011) algorithm, particle swarm optimization (PSO)
[[3]](#Kennedy1995), differential evolution (DE) [[4]](#Storn1997), and Nelder-Mead (NM)
[[5]](#Nelder1965).
Call the following codes to perform state optimization

``` jl
# dynamics and probe state
dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
scheme = GeneralScheme(; probe=psi0, param=dynamics)
# optimization setup
opt = StateOpt(psi=psi0, seed=1234)
alg = AD()
```

``` jl
# objective function: QFI
obj = QFIM_obj(W=nothing, LDtype=:SLD)
```

``` jl
# objective function: CFI
obj = CFIM_obj(M=nothing, W=nothing)
```

``` jl
# objective function: HCRB
obj = HCRB_obj(W=nothing)
```

``` jl
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

The initial state (optimization variable) is input via `psi=psi0` in `StateOpt()` and the
same state is used as the probe in `GeneralScheme(; probe=psi0, param=dynamics)`. `psi0` is
an array representing the normalized complex coefficients of the state. `StateOpt` stores
this initial guess; if `psi=nothing` (the default), a random normalized pure state is
generated. `Lindblad` accepts the time grid `tspan`, the free Hamiltonian matrix `H0`, and
`dH = [∂a H0, ∂b H0, ...]`. Parameter-dependent Hamiltonians can be supplied through
`Hamiltonian` as described in [Parameterization process](guide_dynamics.md). Fixed control
Hamiltonians are passed as `Hc` together with their coefficients through `ctrl` in
`Lindblad(H0, dH, tspan, Hc; ctrl=ctrl, ...)`; `ctrl` contains one sequence per control
Hamiltonian. `decay` contains `[Γ, γ]` pairs with scalar decay rates and is passed as a
positional argument. Omitting `decay` and `Hc` gives unitary dynamics; omitting `ctrl` uses
zero control sequences. `seed` controls reproducibility. `dyn_method=:Expm` selects the
matrix exponential solver, while `dyn_method=:Ode` (the default) selects the ODE solver.

The objective functions for state optimization can be set as QFI ($\mathrm{Tr}(W\mathcal{F}^
{-1})$), CFI ($\mathrm{Tr}(W\mathcal{I}^{-1})$), and HCRB, the corresponding codes for them are
`QFIM_obj()` (default), `CFIM_obj()`, and `HCRB_obj()`. Here $\mathcal{F}$ and
$\mathcal{I}$ are the QFIM and CFIM, and `W` is the weight matrix which
defaults to the identity matrix. If the users call `HCRB_obj()` for a single parameter
scenario, the HCRB is equivalent to the QFI: the program prints a notice and returns the QFI
value. `LDtype` in `QFIM_obj()` represents the types of the QFIM, it can be set as
`LDtype=:SLD` (default), `LDtype=:RLD` and `LDtype=:LLD`. `M` represents a set of positive
operator-valued measure (POVM) with default value `nothing`. In the package, a set of rank-one
symmetric informationally complete POVM (SIC-POVM) is used when `M=nothing`.

`savefile` means whether to save all the states. If set `false` (default) the states in the
final episode are saved in `states.dat` (JLD2 format) and the values of the objective function
in all episodes are saved in `f.csv` after the optimization finishes. If set `true` then the
states and the values of the objective function obtained in all episodes are saved during the
training, with `f.csv` appended at each episode. The saved data can be loaded with

``` jl
using JLD2, DelimitedFiles
f = readdlm("f.csv")
jldopen("states.dat", "r") do file
    states = file["states"]
end
```

See [Output files](output_files.md) for the exact data layout. The algorithms used to optimize
the states in QuanEstimation are AD, RI, PSO, DE, and NM; RI is available for the Kraus
parameterization. `kwargs...` contains the keywords and default values corresponding to the
optimization algorithm which will be introduced in detail below.

---
## **AD**
The code for state optimization with AD is as follows

``` jl
alg = AD(Adam=true, max_episode=300, epsilon=0.01, beta1=0.90,
         beta2=0.99)
```

The keywords and the default values of AD can be seen in the following
table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "Adam"                           | true                       |
| "max_episode"                    | 300                        |
| "epsilon"                        | 0.01                       |
| "beta1"                          | 0.90                       |
| "beta2"                          | 0.99                       |

By default (`Adam=true`) the Adam algorithm is used to update the states. The Adam parameters
include the learning rate and the exponential decay rates for the first and the second moment
estimates, which can be set by the user via `epsilon`, `beta1`, and `beta2`. If `Adam=false`,
the states update according to the learning rate `epsilon` only.

## **RI**
The code for state optimization with RI is as follows

``` jl
alg = RI(max_episode=300)
```

The keywords and the default values of RI can be seen in the following
table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "max_episode"                    | 300                        |

`max_episode` is the number of episodes. RI is used with the Kraus parameterization.

## **PSO**
The code for state optimization with PSO is as follows

``` jl
alg = PSO(p_num=10, ini_particle=nothing,  max_episode=[1000,100],
          c0=1.0, c1=2.0, c2=2.0)
```

The keywords and the default values of PSO can be seen in the following
table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "p_num"                          | 10                         |
| "ini_particle"                   | nothing                    |
| "max_episode"                    | [1000,100]                 |
| "c0"                             | 1.0                        |
| "c1"                             | 2.0                        |
| "c2"                             | 2.0                        |

`p_num` is the number of particles. `ini_particle` is a tuple representing the initial
guesses of states and `max_episode` is the number of episodes. Here `max_episode` accepts
both integers and arrays with two elements. If it is an integer, for example `max_episode=1000`,
it means the program will continuously run 1000 episodes. However, if it is an array, for example
`max_episode=[1000,100]`, the program will run 1000 episodes in total but replace states of
all the particles with global best every 100 episodes. `c0`, `c1`, and `c2` are the PSO
parameters representing the inertia weight, cognitive learning factor, and social learning
factor, respectively.

## **DE**
The code for state optimization with DE is as follows

``` jl
alg = DE(p_num=10, ini_population=nothing, max_episode=1000,
         c=1.0, cr=0.5)
```

The keywords and the default values of DE can be seen in the following
table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "p_num"                          | 10                         |
| "ini_population"                 | nothing                    |
| "max_episode"                    | 1000                       |
| "c"                              | 1.0                        |
| "cr"                             | 0.5                        |

`ini_population` is a tuple representing the initial guesses of states, `p_num` is the
population size, and `max_episode` is the number of generations. `c` and `cr` are the mutation
constant and crossover constant.

## **NM**
The code for state optimization with NM is as follows

``` jl
alg = NM(p_num=10, nelder_mead=nothing, max_episode=1000, ar=1.0,
         ae=2.0, ac=0.5, as0=0.5)
```

The keywords and the default values of NM can be seen in the following
table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "p_num"                          | 10                         |
| "nelder_mead"                    | nothing                    |
| "max_episode"                    | 1000                       |
| "ar"                             | 1.0                        |
| "ae"                             | 2.0                        |
| "ac"                             | 0.5                        |
| "as0"                            | 0.5                        |

`nelder_mead` represents the initial simplex; when it is `nothing`, the current probe
state is used. `ar`, `ae`, `ac`, and `as0` are
constants for reflection, expansion, contraction, and shrink, respectively.



**Example 6.1**
The Hamiltonian of the Lipkin–Meshkov–Glick (LMG) model is
```math
\begin{align}
H_{\mathrm{LMG}}=-\frac{\lambda}{N}(J_1^2+gJ_2^2)-hJ_3,
\end{align}
```

where $N$ is the number of spins of the system, $\lambda$ is the spin–spin interaction strength,
$h$ is the strength of the external field and $g$ is the anisotropic parameter.
$J_i=\frac{1}{2}\sum_{j=1}^N \sigma_i^{(j)}$ ($i=1,2,3$) is the collective spin operator with
$\sigma_i^{(j)}$ the $i$th Pauli matrix for the $j$th spin. In single-parameter scenario, we take
$g$ as the unknown parameter to be estimated. The states are expanded as
$|\psi\rangle=\sum^J_{m=-J}c_m|J,m\rangle$ with $|J,m\rangle$ the Dicke state and $c_m$ a complex
coefficient. Here we fixed $J=N/2$. In this example, the probe state is optimized for both noiseless
scenario and collective dephasing noise. The dynamics under collective dephasing can be expressed as
```math
\partial_t\rho = -i[H_{\mathrm{LMG}},\rho]+\gamma \left(J_3\rho J_3-\frac{1}{2}\left\{\rho, J^2_3\right\}\right)
```
with $\gamma$ the decay rate.

In this case, all searches with different algorithms start from the coherent spin state defined by
$|\theta=\frac{\pi}{2},\phi=\frac{\pi}{2}\rangle=\exp(-\frac{\theta}{2}e^{-i\phi}J_{+}+\frac{\theta}{2}e^{i\phi}J_{-})|J,J\rangle$ with $J_{\pm}=J_1{\pm}iJ_2$.

``` jl
using QuanEstimation
using Random
using StableRNGs
using LinearAlgebra
using SparseArrays

# the dimension of the system
N = 8
# generation of the coherent spin state
j, theta, phi = N÷2, 0.5pi, 0.5pi
Jp = Matrix(spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end]))
Jm = Jp'
psi0 = exp(0.5*theta*exp(im*phi)*Jm - 0.5*theta*exp(-im*phi)*Jp)*
       QuanEstimation.basis(Int(2*j+1), 1)
dim = length(psi0)
# free Hamiltonian
lambda, g, h = 1.0, 0.5, 0.1
Jx = 0.5*(Jp + Jm)
Jy = -0.5im*(Jp - Jm)
Jz = spdiagm(j:-1:-j)
H0 = -lambda*(Jx*Jx + g*Jy*Jy) / N - h*Jz
# derivative of the free Hamiltonian on g
dH = [-lambda*Jy*Jy/N]
# dissipation
decay = [[Jz, 0.1]]
# time length for the evolution
tspan = range(0., 10., length=2500)
# set the optimization type
opt = QuanEstimation.StateOpt(psi=psi0, seed=1234)
```

##### AD
``` jl
# state optimization algorithm: AD
alg = QuanEstimation.AD(Adam=false, max_episode=300, epsilon=0.01,
                        beta1=0.90, beta2=0.99)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`
- **CFIM:** `obj = QuanEstimation.CFIM_obj()`

##### PSO
``` jl
# state optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`
- **CFIM:** `obj = QuanEstimation.CFIM_obj()`

##### DE
``` jl
# state optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`
- **CFIM:** `obj = QuanEstimation.CFIM_obj()`

##### NM
``` jl
# state optimization algorithm: NM
alg = QuanEstimation.NM(p_num=10, max_episode=1000, ar=1.0,
                        ae=2.0, ac=0.5, as0=0.5)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`
- **CFIM:** `obj = QuanEstimation.CFIM_obj()`

``` jl
# input the dynamics data and build the scheme
dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
scheme = GeneralScheme(; probe=psi0, param=dynamics)
# run the state optimization problem
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

**Example 6.2**
In the multiparameter scenario, $g$ and $h$ are chosen to be the unknown parameters to be estimated.

``` jl
using QuanEstimation
using Random
using StableRNGs
using LinearAlgebra
using SparseArrays

# the dimension of the system
N = 8
# generation of the coherent spin state
j, theta, phi = N÷2, 0.5pi, 0.5pi
Jp = Matrix(spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end]))
Jm = Jp'
psi0 = exp(0.5*theta*exp(im*phi)*Jm - 0.5*theta*exp(-im*phi)*Jp)*
       QuanEstimation.basis(Int(2*j+1), 1)
dim = length(psi0)
# free Hamiltonian
lambda, g, h = 1.0, 0.5, 0.1
Jx = 0.5*(Jp + Jm)
Jy = -0.5im*(Jp - Jm)
Jz = spdiagm(j:-1:-j)
H0 = -lambda*(Jx*Jx + g*Jy*Jy) / N - h*Jz
# derivatives of the free Hamiltonian on g and h
dH = [-lambda*Jy*Jy/N, -Jz]
# dissipation
decay = [[Jz, 0.1]]
# time length for the evolution
tspan = range(0., 10., length=2500)
# weight matrix
W = [1/3 0.; 0. 2/3]
# set the optimization type
opt = QuanEstimation.StateOpt(psi=psi0, seed=1234)
```

##### AD
``` jl
# state optimization algorithm: AD
alg = QuanEstimation.AD(Adam=false, max_episode=300, epsilon=0.01,
                        beta1=0.90, beta2=0.99)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj(W=W)`
- **CFIM:** `obj = QuanEstimation.CFIM_obj(W=W)`

##### PSO
``` jl
# state optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj(W=W)`
- **CFIM:** `obj = QuanEstimation.CFIM_obj(W=W)`
- **HCRB:** `obj = QuanEstimation.HCRB_obj(W=W)`

##### DE
``` jl
# state optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj(W=W)`
- **CFIM:** `obj = QuanEstimation.CFIM_obj(W=W)`
- **HCRB:** `obj = QuanEstimation.HCRB_obj(W=W)`

##### NM
``` jl
# state optimization algorithm: NM
alg = QuanEstimation.NM(p_num=10, max_episode=1000, ar=1.0,
                        ae=2.0, ac=0.5, as0=0.5)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj(W=W)`
- **CFIM:** `obj = QuanEstimation.CFIM_obj(W=W)`
- **HCRB:** `obj = QuanEstimation.HCRB_obj(W=W)`

``` jl
# input the dynamics data and build the scheme
dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
scheme = GeneralScheme(; probe=psi0, param=dynamics)
# run the state optimization problem
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

If the parameterization process is implemented with the Kraus operators, then the corresponding
codes are

``` jl
# dynamics with Kraus operators and probe state
dynamics = Kraus(K, dK)
scheme = GeneralScheme(; probe=psi, param=dynamics)
# optimization setup
opt = StateOpt(psi=psi, seed=1234)
alg = AD()
```

``` jl
# objective function: QFI
obj = QFIM_obj(W=nothing, LDtype=:SLD)
```

``` jl
# objective function: CFI
obj = CFIM_obj(M=nothing, W=nothing)
```

``` jl
# objective function: HCRB
obj = HCRB_obj(W=nothing)
```

``` jl
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

where `K` and `dK` are the Kraus operators and their derivatives with respect to the
unknown parameters.

**Example 6.3**
The Kraus operators for the amplitude damping channel are

```math
\begin{align*}
K_1 = \left(\begin{array}{cc}
1 & 0  \\
0 & \sqrt{1-\gamma}
\end{array}\right),
K_2 = \left(\begin{array}{cc}
0 & \sqrt{\gamma} \\
0 & 0
\end{array}\right),
\end{align*}
```

where $\gamma$ is the unknown parameter to be estimated which represents the decay probability.
The probe state is taken as $|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$.

``` jl
using QuanEstimation

# initial state |+> = (|0> + |1>)/sqrt(2)
psi0 = [1.0, 1.0]/sqrt(2)
# Kraus operators for the amplitude damping channel
gamma = 0.1
K1 = [1. 0.; 0. sqrt(1-gamma)]
K2 = [0. sqrt(gamma); 0. 0.]
K = [K1, K2]
# derivatives of Kraus operators on gamma
dK1 = [1. 0.; 0. -0.5/sqrt(1-gamma)]
dK2 = [0. 0.5/sqrt(gamma); 0. 0.]
dK = [[dK1], [dK2]]
# set the optimization type
opt = QuanEstimation.StateOpt(psi=psi0, seed=1234)
```

##### AD
``` jl
# state optimization algorithm: AD
alg = QuanEstimation.AD(Adam=false, max_episode=300, epsilon=0.01,
                        beta1=0.90, beta2=0.99)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`
- **CFIM:** `obj = QuanEstimation.CFIM_obj()`

##### RI
``` jl
# state optimization algorithm: RI
alg = QuanEstimation.RI(max_episode=300)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`

##### PSO
``` jl
# state optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`
- **CFIM:** `obj = QuanEstimation.CFIM_obj()`

##### DE
``` jl
# state optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`
- **CFIM:** `obj = QuanEstimation.CFIM_obj()`

##### NM
``` jl
# state optimization algorithm: NM
alg = QuanEstimation.NM(p_num=10, max_episode=1000, ar=1.0,
                        ae=2.0, ac=0.5, as0=0.5)
```

- **QFIM:** `obj = QuanEstimation.QFIM_obj()`
- **CFIM:** `obj = QuanEstimation.CFIM_obj()`

``` jl
# input the dynamics data and build the scheme
dynamics = Kraus(K, dK)
scheme = GeneralScheme(; probe=psi0, param=dynamics)
# run the state optimization problem
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

---
## **Bibliography**

```@raw html
<a id="Baydin2018"></a>
```

**[1]**
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind,
Automatic differentiation in machine learning: a survey,
[J. Mach. Learn. Res. **18**, 1-43 (2018).](http://jmlr.org/papers/v18/17-468.html)

```@raw html
<a id="Rafal2011"></a>
```

**[2]**
R. Demkowicz-Dobrzański,
Optimal phase estimation with arbitrary a priori knowledge,
[Phys. Rev. A **83**, 061802(R) (2011).
](https://doi.org/10.1103/PhysRevA.83.061802)

```@raw html
<a id="Kennedy1995"></a>
```

**[3]**
J. Kennedy and R. Eberhar,
Particle swarm optimization,
[Proc. 1995 IEEE International Conference on Neural Networks **4**, 1942-1948 (1995).
](https://doi.org/10.1109/ICNN.1995.488968)

```@raw html
<a id="Storn1997"></a>
```

**[4]**
R. Storn and K. Price,
Differential Evolution-A Simple and Efficient Heuristic for global
Optimization over Continuous Spaces,
[J. Global Optim. **11**, 341 (1997).](https://doi.org/10.1023/A:1008202821328)

```@raw html
<a id="Nelder1965"></a>
```

**[5]**
J. A. Nelder and R. Mead,
A Simplex Method for Function Minimization,
[Comput. J. **7**, 308–313 (1965).](https://doi.org/10.1093/comjnl/7.4.308)

**See also**: [Parameterization process](guide_dynamics.md) — construct the system dynamics. [Quantum metrological tools](guide_bounds.md) — the objective functions being optimized. [Output files](output_files.md) — how to load the saved `states.dat` (JLD2) and `f.csv` results.
