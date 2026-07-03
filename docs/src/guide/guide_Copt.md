---
header-includes:
  - \usepackage{caption}
---

# **Control optimization**
The Hamiltonian of a controlled system can be written as
```math
\begin{align}
H = H_0(\textbf{x})+\sum_{k=1}^K u_k(t) H_k,
\end{align}
```

where $H_0(\textbf{x})$ is the free evolution Hamiltonian with unknown parameters $\textbf{x}$ 
and $H_k$ represents the $k$th control Hamiltonian with $u_k$ the corresponding control 
coefficients. In QuanEstimation, different algorithms are invoked to update the control 
coefficients. The control optimization algorithms are the gradient ascent pulse engineering 
(GRAPE) [[1,2,3]](#Khaneja2005), GRAPE algorithm based on the automatic differentiation 
(auto-GRAPE) [[4]](#Baydin2018), particle swarm optimization (PSO) [[5]](#Kennedy1995), 
and differential evolution (DE) [[6]](#Storn1997). The control optimization workflow is as follows:

``` jl
opt = QuanEstimation.ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
alg = QuanEstimation.autoGRAPE(kwargs...)
dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc, decay=missing, 
                                   dyn_method=:Expm) 
```

The objective function can be set using one of the following:

#### QFIM
``` jl
obj = QuanEstimation.QFIM_obj(W=missing, LDtype=:SLD)
```

#### CFIM
``` jl
obj = QuanEstimation.CFIM_obj(M=missing, W=missing)
```

#### HCRB
``` jl
obj = QuanEstimation.HCRB_obj(W=missing)
```

``` jl
QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
```

In control optimization, a set of control coefficients (optimization variable) and its boundary 
value can be input via `ctrl=ctrl` and `ctrl_bound=ctrl_bound` in `ControlOpt()`.
`ctrl` is a list of arrays with the length equal to control Hamiltonians, 
`ctrl_bound` is an array with two elements representing the lower and upper bound of the 
control coefficients, respectively. The default value of `ctrl_bound=missing` which means 
the control coefficients are in the regime $[-\infty,\infty]$. `seed` is the random seed 
which can ensure the reproducibility of results.

The package can deal with the parameterization process in the form of master equation, the 
dynamics parameters should be input via `Lindblad()`. Here `tspan` is the time length for 
the evolution and `rho0` represents the density matrix of the initial state. `H0` and `dH` 
are the free Hamiltonian and its derivatives with respect to the unknown parameters to be 
estimated. `H0` is a matrix when the free Hamiltonian is time-independent and a list of matrices
with the length equal to `tspan` when it is time-dependent. `dH` should be input as $[\partial_a{H_0}, 
\partial_b{H_0}, \cdots]$. `Hc` is a list representing the control Hamiltonians. `decay` 
contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates 
$(\gamma_1, \gamma_2, \cdots)$ with the input rule decay=[[$\Gamma_1$, $\gamma_1$], 
[$\Gamma_2$, $\gamma_2$],...]. The default value `decay` is `missing` which means the 
dynamics is unitary. `dyn_method=:Expm` represents the method for solving the dynamics is 
matrix exponential, it can also be set as `dyn_method=:Ode` which means the dynamics 
(differential equation) is directly solved with the ODE solvers.

The objective functions for control optimization can be set as QFI $\left[\mathrm{Tr}(W
\mathcal{F}^{-1})\right]$, CFI $\left[\mathrm{Tr}(W\mathcal{I}^{-1})\right]$, and HCRB, the 
corresponding codes for them are `QFIM_obj()` (default), `CFIM_obj()`, and `HCRB_obj()`. Here 
$\mathcal{F}$ and $\mathcal{I}$ are the QFIM and CFIM, $W$ corresponds to `W` is the weight 
matrix which defaults to the identity matrix. If the users call `HCRB_obj()` for single parameter 
scenario, the program will exit and print `"Program terminated. In the single-parameter scenario, 
the HCRB is equivalent to the QFI. Please choose 'QFIM_obj' as the objective function"`. 
`LDtype` in `QFIM_obj()` represents the types of the QFIM, it can be set as `LDtype=:SLD` 
(default), `LDtype=:RLD`, and `LDtype=:LLD`. `M` in `CFIM_obj()` represents a set of positive 
operator-valued measure (POVM) with default value `missing` which means a set of rank-one 
symmetric informationally complete POVM (SIC-POVM) is used in the calculation.

The variable `savefile` indicates whether to save all the control coefficients and its default 
value is `false` which means the control coefficients for the final episode and the values of 
the objective function in all episodes will be saved in "controls.csv" and "f.csv", respectively.
If set `true` then the control coefficients and the values of the objective function in all 
episodes will be saved during the training. The algorithm used for optimizing the control
coefficients in QuanEstimation are GRAPE, auto-GRAPE, PSO, DE, and DDPG. `kwargs...` contains 
the keywords and defaults value corresponding to the optimization algorithm which will be 
introduced in detail below.

---
## **GRAPE and auto-GRAPE**
The codes for control optimization with GRAPE and auto-GRAPE are as follows

``` jl
alg = QuanEstimation.GRAPE(Adam=true, max_episode=300, epsilon=0.01, beta1=0.90, 
                           beta2=0.99)
```

``` jl
alg = QuanEstimation.autoGRAPE(Adam=true, max_episode=300, epsilon=0.01, beta1=0.90, 
                               beta2=0.99)
```

The keywords and the default values of GRAPE and auto-GRAPE can be seen in the following table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "Adam"                           | true                       |
| "max_episode"                    | 300                        |
| "epsilon"                        | 0.01                       |
| "beta1"                          | 0.90                       |
| "beta2"                          | 0.99                       |

Adam algorithm can be introduced to update the control coefficients when using GRAPE and 
auto-GRAPE for control optimization, which can be realized by setting `Adam=true`. In this 
case, the Adam parameters include learning rate, the exponential decay rate for the first 
moment estimates and the second moment estimates can be set by the users via `epsilon`, `beta1`,
and `beta2`, respectively. If `Adam=false`, the control coefficients will update according to 
the learning rate `"epsilon"`.

## **PSO**
The code for control optimization with PSO is as follows

``` jl
alg = QuanEstimation.PSO(p_num=10, ini_particle=missing, max_episode=[1000,100], 
                         c0=1.0, c1=2.0, c2=2.0)
```

The keywords and the default values of PSO can be seen in the following table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "p_num"                          | 10                         |
| "ini_particle"                   | missing                    |
| "max_episode"                    | [1000,100]                 |
| "c0"                             | 1.0                        |
| "c1"                             | 2.0                        |
| "c2"                             | 2.0                        |

Here `p_num` is the number of particles. `c0`, `c1`, and `c2` are the PSO parameters 
representing the inertia weight, cognitive learning factor, and social learning factor, 
respectively. `max_episode` accepts both integer and array with two elements. If it is an 
integer, for example `max_episode=1000`, it means the program will continuously run 1000 
episodes. However, if it is an array, for example `max_episode=[1000,100]`, the program will 
run 1000 episodes in total but replace control coefficients of all the particles with global 
best every 100 episodes. `ini_particle` is a tuple representing the initial guesses of control 
coefficients.

## **DE**
The code for control optimization with DE is as follows

``` jl
alg = QuanEstimation.DE(p_num=10, ini_population=missing, max_episode=1000, 
                        c=1.0, cr=0.5)
```

The keywords and the default values of DE can be seen in the following table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "p_num"                          | 10                         |
| "ini_population"                 | missing                    |
| "max_episode"                    | 1000                       |
| "c"                              | 1.0                        |
| "cr"                             | 0.5                        |

`ini_population` is a tuple representing the initial guesses of control and `max_episode` 
represents the number of populations and episodes. `c` and `cr` are the mutation constant 
and the crossover constant.

**Example 5.1**  
<a id="example5_1"></a>
In this example, the free evolution Hamiltonian of a single qubit system is $H_0=\frac{1}{2}\omega 
\sigma_3$ with $\omega$ the frequency and $\sigma_3$ a Pauli matrix. The dynamics of the system 
is governed by

```math
\begin{align}
\partial_t\rho=-i[H_0, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}
\sigma_{+},\rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}\sigma_{-},
\rho\}\right),
\end{align}
```

where $\gamma_{+}$, $\gamma_{-}$ are decay rates and $\sigma_{\pm}=(\sigma_1 \pm \sigma_2)/2$. 
The control Hamiltonian 
```math
\begin{align}
H_\mathrm{c}=u_1(t)\sigma_1+u_2(t)\sigma_2+u_3(t)\sigma_3.
\end{align}
```

Here $\sigma_{1}$, $\sigma_{2}$ are also Pauli matrices. The probe state is taken as $|+\rangle$ and 
the measurement for CFI is $\{|+\rangle\langle+|, |-\rangle\langle-|\}$ with
$|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$. $|0\rangle$ $(|1\rangle)$ is the eigenstate 
of $\sigma_3$ with respect to the eigenvalue $1$ $(-1)$.

``` jl
using QuanEstimation
using Random

# initial state
rho0 = 0.5*ones(2, 2)
# free Hamiltonian
omega = 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
H0 = 0.5*omega*sz
# derivative of the free Hamiltonian on omega
dH = [0.5*sz]
# control Hamiltonians 
Hc = [sx, sy, sz]
# dissipation
sp = [0. 1.; 0. 0.0im]
sm = [0. 0.; 1. 0.0im]
decay = [[sp, 0.], [sm, 0.1]]
# measurement
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., 10., length=2500)
# guessed control coefficients
cnum = length(tspan)-1
ctrl = [zeros(cnum) for _ in 1:length(Hc)]
ctrl_bound = [-2., 2.]
# set the optimization type
opt = QuanEstimation.ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
```

#### auto-GRAPE
``` jl
# control algorithm: auto-GRAPE
alg = QuanEstimation.autoGRAPE(Adam=true, max_episode=300, epsilon=0.01, 
                               beta1=0.90, beta2=0.99)
```

##### QFIM
``` jl
# objective function: QFI
obj = QuanEstimation.QFIM_obj()
```

##### CFIM
``` jl
# objective function: CFI
obj = QuanEstimation.CFIM_obj(M=M)
```

#### GRAPE
``` jl
# control algorithm: GRAPE
alg = QuanEstimation.GRAPE(Adam=true, max_episode=300, epsilon=0.01, 
                           beta1=0.90, beta2=0.99)
```

##### QFIM
``` jl
# objective function: QFI
obj = QuanEstimation.QFIM_obj()
```

##### CFIM
``` jl
# objective function: CFI
obj = QuanEstimation.CFIM_obj(M=M)
```

#### PSO
``` jl
# control algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, ini_particle=([ctrl],), 
                         max_episode=[1000,100], c0=1.0, 
                         c1=2.0, c2=2.0)
```

##### QFIM
``` jl
# objective function: QFI
obj = QuanEstimation.QFIM_obj()
```

##### CFIM
``` jl
# objective function: CFI
obj = QuanEstimation.CFIM_obj(M=M)
```

#### DE
``` jl
# control algorithm: DE
alg = QuanEstimation.DE(p_num=10, ini_population=([ctrl],), 
                        max_episode=1000, c=1.0, cr=0.5)
```

##### QFIM
``` jl
# objective function: QFI
obj = QuanEstimation.QFIM_obj()
```

##### CFIM
``` jl
# objective function: CFI
obj = QuanEstimation.CFIM_obj(M=M)
```

``` jl
# input the dynamics data
dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc, decay, 
                                   dyn_method=:Expm)  
# run the control optimization problem
QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
```

**Example 5.2**  
<a id="example5_2"></a>
In the multiparameter scenario, the dynamics of electron and nuclear coupling in NV$^{-}$ can 
be expressed as
```math
\begin{align}
\partial_t\rho=-i[H_0,\rho]+\frac{\gamma}{2}(S_3\rho S_3-S^2_3\rho-\rho S^2_3)
\end{align}
```

with $\gamma$ the dephasing rate. And
```math
\begin{align}
H_0/\hbar=DS^2_3+g_{\mathrm{S}}\vec{B}\cdot\vec{S}+g_{\mathrm{I}}\vec{B}\cdot\vec{I}+\vec{S}^
{\,\mathrm{T}}\mathcal{A}\vec{I}
\end{align}
```

is the free evolution Hamiltonian, where $\vec{S}=(S_1,S_2,S_3)^{\mathrm{T}}$ and 
$\vec{I}=(I_1,I_2,I_3)^{\mathrm{T}}$ with $S_i=s_i\otimes I$ and $I_i=I\otimes \sigma_i$ 
$(i=1,2,3)$ the electron and nuclear operators. $s_1, s_2, and s_3$ are spin-1 operators with 

```math
\begin{align*}
s_1 = \frac{1}{\sqrt{2}}\left(\begin{array}{ccc}
0 & 1 & 0 \\
1 & 0 & 1 \\
0 & 1 & 0
\end{array}\right),
s_2 = \frac{1}{\sqrt{2}}\left(\begin{array}{ccc}
0 & -i & 0\\
i & 0 & -i\\
0 & i & 0
\end{array}\right),
\end{align*}
```

and $s_3=\mathrm{diag}(1,0,-1)$ and $\sigma_i (i=1,2,3)$ is Pauli matrix. $\mathcal{A}=\mathrm{diag}
(A_1,A_1,A_2)$ is the hyperfine tensor with $A_1$ and $A_2$ the axial and transverse magnetic 
hyperfine coupling coefficients. The coefficients $g_{\mathrm{S}}=g_\mathrm{e}\mu_\mathrm{B}/\hbar$ 
and $g_{\mathrm{I}}=g_\mathrm{n}\mu_\mathrm{n}/\hbar$, where $g_\mathrm{e}$ ($g_\mathrm{n}$) is 
the $g$ factor of the electron (nuclear), $\mu_\mathrm{B}$ ($\mu_\mathrm{n}$) is the Bohr (nuclear) 
magneton, and $\hbar$ is the Plank's constant. $\vec{B}$ is the magnetic field which be estimated.

In this case, the initial state is taken as $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)\otimes
|\!\!\uparrow\rangle$, where $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)$ is an electron state 
with $|1\rangle$ $\left(|-\!1\rangle\right)$ the eigenstate of $s_3$ with respect to the eigenvalue $1$ 
($-1$). $|\uparrow\rangle$ is a nuclear state and the eigenstate of $\sigma_3$ with respect 
to the eigenvalue 1. $W$ is set to be identity.

Here three types of algorithms are invoked to search the optimal controls: GRAPE,
auto-GRAPE, PSO, and DE. For GRAPE and auto-GRAPE, the objective functions are QFI and CFI.
For PSO and DE, besides QFI and CFI, HCRB is also considered.

``` jl
using QuanEstimation
using Random
using LinearAlgebra

# initial state
rho0 = zeros(ComplexF64, 6, 6)
rho0[1:4:5, 1:4:5] .= 0.5
dim = size(rho0, 1)
# Hamiltonian
sx = [0. 1.; 1. 0.]
sy = [0. -im; im 0.]
sz = [1. 0.; 0. -1.]
s1 = [0. 1. 0.; 1. 0. 1.; 0. 1. 0.]/sqrt(2)
s2 = [0. -im 0.; im 0. -im; 0. im 0.]/sqrt(2)
s3 = [1. 0. 0.; 0. 0. 0.; 0. 0. -1.]
Is = I1, I2, I3 = [kron(I(3), sx), kron(I(3), sy), kron(I(3), sz)]
S = S1, S2, S3 = [kron(s1, I(2)), kron(s2, I(2)), kron(s3, I(2))]
B = B1, B2, B3 = [5.0e-4, 5.0e-4, 5.0e-4]
# All numbers are divided by 100 in this example 
# for better calculation accuracy
cons = 100
D = (2pi*2.87*1000)/cons
gS = (2pi*28.03*1000)/cons
gI = (2pi*4.32)/cons
A1 = (2pi*3.65)/cons
A2 = (2pi*3.03)/cons
H0 = sum([D*kron(s3^2, I(2)), sum(gS*B.*S), sum(gI*B.*Is),
            A1*(kron(s1, sx) + kron(s2, sy)), A2*kron(s3, sz)])
# derivatives of the free Hamiltonian on B1, B2 and B3
dH = gS*S+gI*Is
# control Hamiltonians 
Hc = [S1, S2, S3]
# dissipation
decay = [[S3, 2*pi/cons]]
# measurement
M = [QuanEstimation.basis(dim, i)*QuanEstimation.basis(dim, i)' 
     for i in 1:dim]
# time length for the evolution 
tspan = range(0., 2., length=4000)
# guessed control coefficients
cnum = 10
rng = MersenneTwister(1234)
ini_1 = [zeros(cnum) for _ in 1:length(Hc)]
ini_2 = 0.2.*[ones(cnum) for _ in 1:length(Hc)]
ini_3 = -0.2.*[ones(cnum) for _ in 1:length(Hc)]
ini_4 = [[range(-0.2, 0.2, length=cnum)...] for _ in 1:length(Hc)]
ini_5 = [[range(-0.2, 0., length=cnum)...] for _ in 1:length(Hc)]
ini_6 = [[range(0., 0.2, length=cnum)...] for _ in 1:length(Hc)]
ini_7 = [-0.2*ones(cnum)+0.01*rand(rng,cnum) for _ in 1:length(Hc)]
ini_8 = [-0.2*ones(cnum)+0.01*rand(rng,cnum) for _ in 1:length(Hc)]
ini_9 = [-0.2*ones(cnum)+0.05*rand(rng,cnum) for _ in 1:length(Hc)]
ini_10 = [-0.2*ones(cnum)+0.05*rand(rng,cnum) for _ in 1:length(Hc)]
ctrl0 = [Symbol("ini_", i)|>eval for i in 1:10]
# set the optimization type
opt = QuanEstimation.ControlOpt(ctrl=ini_1, ctrl_bound=[-0.2, 0.2], seed=1234)
```

#### auto-GRAPE
``` jl
# control algorithm: auto-GRAPE
alg = QuanEstimation.autoGRAPE(Adam=true, max_episode=300, epsilon=0.01, 
                               beta1=0.90, beta2=0.99)
```

##### QFIM
``` jl
# objective function: tr(WF^{-1})
obj = QuanEstimation.QFIM_obj()
```

##### CFIM
``` jl
# objective function: tr(WI^{-1})
obj = QuanEstimation.CFIM_obj(M=M)
```

#### PSO
``` jl
# control algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, ini_particle=(ctrl0,), 
                         max_episode=[1000,100], c0=1.0, 
                         c1=2.0, c2=2.0)
```

##### QFIM
``` jl
# objective function: tr(WF^{-1})
obj = QuanEstimation.QFIM_obj()
```

##### CFIM
``` jl
# objective function: tr(WI^{-1})
obj = QuanEstimation.CFIM_obj(M=M)
```

##### HCRB
``` jl
# objective function: HCRB
obj = QuanEstimation.HCRB_obj()
```

#### DE
``` jl
# control algorithm: DE
alg = QuanEstimation.DE(p_num=10, ini_population=(ctrl0,), 
                        max_episode=1000, c=1.0, cr=0.5)
```

##### QFIM
``` jl
# objective function: tr(WF^{-1})
obj = QuanEstimation.QFIM_obj()
```

##### CFIM
``` jl
# objective function: tr(WI^{-1})
obj = QuanEstimation.CFIM_obj(M=M)
```

##### HCRB
``` jl
# objective function: HCRB
obj = QuanEstimation.HCRB_obj()
```

``` jl
# input the dynamics data
dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc, decay, 
                                   dyn_method=:Expm)  
# run the control optimization problem
QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
```

---
## **Minimum parameterization time optimization**
Search of the minimum time to reach a given value of the objective function.

``` jl
QuanEstimation.mintime(f, opt, alg, obj, dynamics; method="binary", savefile=false)
```

`f` is the given value of the objective function. In the package, two methods for searching 
the minimum time are provided which are logarithmic search and forward search from the 
beginning of time. It can be realized by setting `method=binary` (default) and 
`method=forward`. The objective function type (QFI $\left[\mathrm{Tr}(WF^{-1})\right]$, 
CFI $\left[\mathrm{Tr}(WI^{-1})\right]$, or HCRB) and the logarithmic derivative types 
(for QFIM: `:SLD`, `:RLD`, `:LLD`) are configured through the `obj` argument passed to 
`mintime`.

---
## **Bibliography**
**[1]**
N. Khaneja, T. Reiss, C. Hehlet, T. Schulte-Herbruggen, and S. J. Glaser,
Optimal control of coupled spin dynamics: Design of NMR pulse sequences by gradient 
ascent algorithms,
[J. Magn. Reson. **172**, 296 (2005).](https://doi.org/10.1016/j.jmr.2004.11.004)

**[2]**
J. Liu and H. Yuan,
Quantum parameter estimation with optimal control,
[Phys. Rev. A **96**, 012117 (2017).](https://doi.org/10.1103/PhysRevA.96.012117)

**[3]**
J. Liu and H. Yuan,
Control-enhanced multiparameter quantum estimation,
[Phys. Rev. A **96**, 042114 (2017).](https://doi.org/10.1103/PhysRevA.96.042114)

**[4]**
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind,
Automatic differentiation in machine learning: a survey,
[J. Mach. Learn. Res. **18**, 1-43 (2018).](http://jmlr.org/papers/v18/17-468.html)

**[5]**
J. Kennedy and R. Eberhar,
Particle swarm optimization,
[Proc. 1995 IEEE International Conference on Neural Networks **4**, 1942-1948 (1995).
](https://doi.org/10.1109/ICNN.1995.488968)

**[6]**
R. Storn and K. Price,
Differential Evolution-A Simple and Efficient Heuristic for global
Optimization over Continuous Spaces,
[J. Global Optim. **11**, 341 (1997).](https://doi.org/10.1023/A:1008202821328)

**See also**: [Parameterization process](guide_dynamics.md) — construct the controlled Hamiltonian and define dynamics parameters. [Quantum metrological tools](guide_bounds.md) — the objective functions being optimized. [Output files](output_files.md) — how to load the saved `controls.npy` results.
