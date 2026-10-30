# **Comprehensive optimization**
In order to obtain the optimal parameter estimation schemes, it is necessary to
simultaneously optimize the probe state, control, and measurement. The comprehensive
optimization for the probe state and measurement (SM), the probe state and control (SC), the
control and measurement (CM) and the probe state, control and measurement (SCM) are proposed
in QuanEstimation. In the package, the comprehensive optimization algorithms are particle
swarm optimization (PSO) [[1]](#Kennedy1995), differential evolution (DE) [[2]](#Storn1997),
and automatic differentiation (AD) [[3]](#Baydin2018). PSO and DE can be used for all four
scenarios; AD is currently available for SC with QFIM and CFIM objectives.

### SM
```jl
opt = SMopt(psi=psi, M=C, seed=1234)
alg = DE(p_num=10, ini_population=nothing, max_episode=1000, c=1.0, cr=0.5)
dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)  # omit decay if unitary
scheme = GeneralScheme(probe=rho0, param=dynamics)
obj = CFIM_obj(W=nothing)
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

### SC
```jl
opt = SCopt(psi=psi, ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
alg = DE(p_num=10, ini_population=nothing, max_episode=1000, c=1.0, cr=0.5)
dynamics = Lindblad(H0, dH, tspan, Hc; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(probe=rho0, param=dynamics)
```

#### QFIM
```jl
obj = QFIM_obj(W=nothing, LDtype=:SLD)
```

#### CFIM
```jl
obj = CFIM_obj(M=M_povm, W=nothing)
```

#### HCRB
```jl
obj = HCRB_obj(W=nothing)
```

```jl
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

### CM
```jl
opt = CMopt(ctrl=ctrl, M=C, ctrl_bound=ctrl_bound, seed=1234)
alg = DE(p_num=10, ini_population=nothing, max_episode=1000, c=1.0, cr=0.5)
dynamics = Lindblad(H0, dH, tspan, Hc; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(probe=rho0, param=dynamics)
obj = CFIM_obj(W=nothing)
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

### SCM
```jl
opt = SCMopt(psi=psi, ctrl=ctrl, M=C, ctrl_bound=ctrl_bound, seed=1234)
alg = DE(p_num=10, ini_population=nothing, max_episode=1000, c=1.0, cr=0.5)
dynamics = Lindblad(H0, dH, tspan, Hc; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(probe=rho0, param=dynamics)
obj = CFIM_obj(W=nothing)
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```

QuanEstimation contains four comprehensive optimizations which are `SMopt()`, `SCopt()`,
`CMopt()`, and `SCMopt()`. The optimization variables including initial state, control,
and measurement can be input via `ctrl=ctrl`, `psi=psi`, and `M=C` for constructing a
comprehensive optimization problem. Here, `ctrl` is a list of arrays with the length
equal to control Hamiltonians, `psi` is a state vector representing the initial probe
state, and `C` is a list of basis vectors representing the initial projective measurement basis.
The POVM used by a fixed-measurement CFIM objective is instead supplied to the objective
as `CFIM_obj(M=M_povm)`. The optimized measurement itself is represented internally by its
basis vectors; the saved measurement consists of the corresponding POVM elements.
Besides, the boundary value of each control coefficients can be input via `ctrl_bound` when
the optimized variable contains control. `ctrl_bound` is an array with two elements
representing the lower and upper bound of the control coefficients, respectively. The default
value of `ctrl_bound` is `[-Inf, Inf]`, which means the control coefficients are unbounded.
`seed` is the random seed which can ensure the reproducibility of results.

The objective function of `SCopt()` can be chosen as `QFIM_obj()` (default), `CFIM_obj()`,
and `HCRB_obj()` for the corresponding objective functions are QFI ($\mathrm{Tr}(W\mathcal{F}^
{-1})$), CFI ($\mathrm{Tr}(W\mathcal{I}^{-1})$), and HCRB, respectively. Here $\mathcal{F}$
and $\mathcal{I}$ are the QFIM and CFIM, $W$ corresponds to `W` is the weight matrix which
defaults to the identity matrix. For the SC scenario the measurement is fixed, so a CFIM
objective should be built with the POVM, e.g. `CFIM_obj(M=M_povm)`. HCRB is defined for the
multiparameter scenario; for a single parameter use `QFIM_obj()`, since the HCRB reduces to
the QFI in that case. `LDtype` represents the types of the QFIM, it can be set as
`LDtype=:SLD` (default), `LDtype=:RLD`, and `LDtype=:LLD`. For the other three scenarios,
where the measurement is optimized, the objective function is `CFIM_obj()`.

If the dynamics of the system can be described by the master equation, then the dynamics data
`tspan`, `H0`, and `dH` should be input. `tspan` is the time length for the evolution, `H0`
and `dH` are the free Hamiltonian matrix and its derivatives, `dH = [∂a H0, ∂b H0, ...]`.
Parameter-dependent Hamiltonians can be supplied through `Hamiltonian` as described in
[Parameterization process](guide_dynamics.md). `Hc` and `ctrl` are two lists representing the
control Hamiltonians and the corresponding control coefficients. In the current interface
`Hc` is the fourth positional argument of `Lindblad` and `ctrl` is a keyword, i.e.
`Lindblad(H0, dH, tspan, Hc; ctrl=ctrl, dyn_method=:Expm)`. `decay` contains decay operators
$(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$
with the input rule decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...]. If both
controls and decay are present, `decay` is the fifth positional argument:
`Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)`.
`dyn_method=:Expm` represents the method for solving the dynamics is matrix exponential, it
can also be set as `dyn_method=:Ode` which means the dynamics (differential equation) is
directly solved with the ODE solvers. The default is `dyn_method=:Ode`.

The variable `savefile` means whether to save all the optimized variables (probe states,
control coefficients, and measurements). If set `true` then the optimized variables and the
values of the objective function obtained in all episodes will be saved during the training,
otherwise, the optimized variables in the final episode and the values of the objective
function in all episodes will be saved. The saved variables are written as JLD2 files:
`states.dat` (key `"states"`), `controls.dat` (key `"controls"`), and `measurements.dat`
(key `"measurements"`); only the files corresponding to the optimization variables of the
chosen scenario are produced. The objective values are written to `f.csv`. The algorithm used
in QuanEstimation for comprehensive optimization are PSO, DE, and AD; AD is available for
SC with QFIM and CFIM objectives (use PSO or DE for HCRB). `kwargs...` is the keywords and
the default values corresponding to the optimization algorithm which will be introduced in
detail below.

---
## **PSO**
The code for comprehensive optimization with PSO is as follows
```jl
alg = PSO(p_num=10, ini_particle=nothing, max_episode=[1000,100],
          c0=1.0, c1=2.0, c2=2.0)
```
The keywords and the default values of PSO can be seen in the following
table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| `p_num`                          | 10                         |
| `ini_particle`                   | nothing                    |
| `max_episode`                    | [1000,100]                 |
| `c0`                             | 1.0                        |
| `c1`                             | 2.0                        |
| `c2`                             | 2.0                        |

`ini_particle` is a tuple whose elements are lists of initial guesses, one list for each
optimization variable. The input rules are `([psi], [measurement])` for SM,
`([psi], [ctrl])` for SC, `([ctrl], [measurement])` for CM, and
`([psi], [ctrl], [measurement])` for SCM. Here `psi` is a state vector, `ctrl` is a list of
control-coefficient arrays, and `measurement` is a list of basis vectors representing one
projective measurement. Each list may contain up to `p_num` guesses; variables for which no
guess is supplied are initialized randomly (controls within `ctrl_bound`).
Here `p_num` is the number of particles, `c0`, `c1`, and `c2` are the PSO parameters
representing the inertia weight, cognitive learning factor, and social learning
factor, respectively. `max_episode` accepts both integers and arrays with two
elements. If it is an integer, for example max_episode=1000, it means the program
will continuously run 1000 episodes. However, if it is an array, for example
max_episode=[1000,100], the program will run 1000 episodes in total but replace
the optimization variables of all the particles with the global best every 100 episodes.

## **DE**
The code for comprehensive optimization with DE is as follows
```jl
alg = DE(p_num=10, ini_population=nothing, max_episode=1000,
         c=1.0, cr=0.5)
```
The keywords and the default values of DE can be seen in the following
table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| `p_num`                          | 10                         |
| `ini_population`                 | nothing                    |
| `max_episode`                    | 1000                       |
| `c`                              | 1.0                        |
| `cr`                             | 0.5                        |

Here `max_episode` is an integer representing the number of episodes. `p_num`
represents the number of populations. `c` and `cr` are constants for mutation
and crossover. `ini_population` is a tuple containing the initial guesses of the
optimization variables. Its input rule is the same as that of `ini_particle`:
`([psi], [measurement])` for SM, `([psi], [ctrl])` for SC,
`([ctrl], [measurement])` for CM, and `([psi], [ctrl], [measurement])` for SCM.

## **AD**
The code for comprehensive optimization with AD is as follows
```jl
alg = AD(Adam=true, max_episode=300, epsilon=0.01, beta1=0.90,
        beta2=0.99)
```
The keywords and the default values of AD can be seen in the
following table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| `Adam`                           | true                       |
| `max_episode`                    | 300                        |
| `epsilon`                        | 0.01                       |
| `beta1`                          | 0.90                       |
| `beta2`                          | 0.99                       |

By default `Adam=true`, and the Adam parameters include the learning rate, the exponential
decay rate for the first moment estimates, and the second moment estimates, which can be set
by the user via `epsilon`, `beta1`, and `beta2`. If `Adam=false`, the optimization variables
update according to the learning rate `epsilon`. Among the comprehensive scenarios, AD is
implemented for SC, with QFIM and CFIM objectives; for HCRB use PSO or DE. Use PSO or DE for
SM, CM, and SCM.

**Example 8.1**

```@raw html
<a id="example8_1"></a>
```

A single qubit system whose free evolution Hamiltonian is $H_0 = \frac{1}{2}\omega \sigma_3$ with
$\omega$ the frequency and $\sigma_3$ a Pauli matrix. The dynamics of the system is governed by

```math
\begin{align}
\partial_t\rho=-i[H_0, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}\sigma_{+},\rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}\sigma_{-},\rho\}\right),
\end{align}
```

where $\gamma_{+}$, $\gamma_{-}$ are decay rates and $\sigma_{\pm}=(\sigma_1 \pm \sigma_2)/2$. The control Hamiltonian
```math
\begin{align}
H_\mathrm{c}=u_1(t)\sigma_1+u_2(t)\sigma_2+u_3(t)\sigma_3
\end{align}
```

with $u_i(t)$ $(i=1,2,3)$ the control field. Here $\sigma_{1}$, $\sigma_{2}$ are also Pauli matrices.

In this case, we consider the comprehensive optimization of probe state and measurement (SM),
probe state and control (SC), control and measurement (CM), and probe state, control and
measurement (SCM). QFI and CFI are used as objective functions in the examples below.

```jl
using QuanEstimation
using LinearAlgebra

# initial state
psi0 = ComplexF64[1.0, 1.0]/sqrt(2)
rho0 = psi0*psi0'
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
decay = [[sp, 0.0], [sm, 0.1]]
# measurement basis vectors (initial guess for SM, CM and SCM)
C = [ComplexF64[1.0, 1.0]/sqrt(2), ComplexF64[1.0, -1.0]/sqrt(2)]
# POVM (fixed measurement for the CFIM objective in SC)
M = [c*c' for c in C]
# time length for the evolution
tspan = range(0., 10., length=2500)
# guessed control coefficients
cnum = length(tspan) - 1
ctrl = [zeros(cnum) for _ in 1:length(Hc)]
```

### SM
```jl
opt = QuanEstimation.SMopt(psi=psi0, M=C, seed=1234)
dynamics = QuanEstimation.Lindblad(H0, dH, tspan, Hc, decay;
                                   ctrl=ctrl, dyn_method=:Expm)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
obj = QuanEstimation.CFIM_obj()
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

### SC
```jl
opt = QuanEstimation.SCopt(psi=psi0, ctrl=ctrl, ctrl_bound=[-2.0,2.0],
                           seed=1234)
dynamics = QuanEstimation.Lindblad(H0, dH, tspan, Hc, decay;
                                   ctrl=ctrl, dyn_method=:Expm)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
```

##### QFIM
```jl
# objective function: QFI
obj = QuanEstimation.QFIM_obj()
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

##### CFIM
```jl
# objective function: CFI with the fixed POVM M
obj = QuanEstimation.CFIM_obj(M=M)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
```

##### QFIM
```jl
# objective function: QFI
obj = QuanEstimation.QFIM_obj()
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

##### CFIM
```jl
# objective function: CFI with the fixed POVM M
obj = QuanEstimation.CFIM_obj(M=M)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### AD
```jl
# comprehensive optimization algorithm: AD
alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01,
                        beta1=0.90, beta2=0.99)
```

##### QFIM
```jl
# objective function: QFI
obj = QuanEstimation.QFIM_obj()
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

##### CFIM
```jl
# objective function: CFI with the fixed POVM M
obj = QuanEstimation.CFIM_obj(M=M)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

### CM
```jl
opt = QuanEstimation.CMopt(ctrl=ctrl, M=C, ctrl_bound=[-2.0,2.0],
                           seed=1234)
dynamics = QuanEstimation.Lindblad(H0, dH, tspan, Hc, decay;
                                   ctrl=ctrl, dyn_method=:Expm)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
obj = QuanEstimation.CFIM_obj()
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

### SCM
```jl
opt = QuanEstimation.SCMopt(psi=psi0, ctrl=ctrl, M=C,
                            ctrl_bound=[-2.0,2.0], seed=1234)
dynamics = QuanEstimation.Lindblad(H0, dH, tspan, Hc, decay;
                                   ctrl=ctrl, dyn_method=:Expm)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
obj = QuanEstimation.CFIM_obj()
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

The outputs are `f.csv` together with `states.dat` and `measurements.dat` for SM,
`controls.dat` and `states.dat` for SC, `controls.dat` and `measurements.dat` for CM,
and `controls.dat`, `states.dat` and `measurements.dat` for SCM. The saved files can be
loaded with

```jl
using DelimitedFiles
using JLD2

f = readdlm("f.csv")
if isfile("states.dat")
    states = jldopen("states.dat", "r") do file
        file["states"]
    end
end
if isfile("controls.dat")
    controls = jldopen("controls.dat", "r") do file
        file["controls"]
    end
end
if isfile("measurements.dat")
    measurements = jldopen("measurements.dat", "r") do file
        file["measurements"]
    end
end
```

Only the files produced by the chosen scenario exist; for example, SC does not produce
`measurements.dat`.

**Example 8.2**

```@raw html
<a id="example8_2"></a>
```

The Hamiltonian of a controlled system can be written as
```math
\begin{align}
H = H_0(\textbf{x})+\sum_{k=1}^K u_k(t) H_k,
\end{align}
```

where $H_0(\textbf{x})$ is the free evolution Hamiltonian with unknown parameters $\textbf{x}$ and $H_k$
represents the $k$th control Hamiltonian with $u_k$ the corresponding control coefficient.

In the multiparameter scenario, the dynamics of electron and nuclear coupling in NV$^{-}$ can be expressed as
```math
\begin{align}
\partial_t\rho=-i[H_0+H_{\mathrm{c}},\rho]+\frac{\gamma}{2}(S_3\rho S_3-S^2_3\rho-\rho S^2_3)
\end{align}
```

with $\gamma$ the dephasing rate. And
```math
\begin{align}
H_0/\hbar=DS^2_3+g_{\mathrm{S}}\vec{B}\cdot\vec{S}+g_{\mathrm{I}}\vec{B}\cdot\vec{I}+\vec{S}^{\,\mathrm{T}}\mathcal{A}\vec{I}
\end{align}
```

is the free evolution Hamiltonian, where $\vec{S}=(S_1,S_2,S_3)^{\mathrm{T}}$ and $\vec{I}=(I_1,I_2,I_3)^{\mathrm{T}}$ with
$S_i=s_i\otimes I$ and $I_i=I\otimes \sigma_i$ ($i=1,2,3$) the electron and nuclear operators. $\mathcal{A}=\mathrm{diag}
(A_1,A_1,A_2)$ is the hyperfine tensor with $A_1$ and $A_2$ the axial and transverse magnetic hyperfine coupling coefficients.
The coefficients $g_{\mathrm{S}}=g_\mathrm{e}\mu_\mathrm{B}/\hbar$ and $g_{\mathrm{I}}=g_\mathrm{n}\mu_\mathrm{n}/\hbar$,
where $g_\mathrm{e}$ ($g_\mathrm{n}$) is the $g$ factor of the electron (nuclear), $\mu_\mathrm{B}$ ($\mu_\mathrm{n}$) is the Bohr
(nuclear) magneton, and $\hbar$ is the Plank's constant. $\vec{B}$ is the magnetic field which be estimated. The control Hamiltonian is
```math
\begin{align}
H_{\mathrm{c}}/\hbar=\sum^3_{i=1}\Omega_i(t)S_i
\end{align}
```

with $\Omega_i(t)$ the time-dependent Rabi frequency.

In this case, the initial state is taken as $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)\otimes|\!\!\uparrow\rangle$,
where $\frac{1}{\sqrt{2}}(|1\rangle+|\!-\!1\rangle)$ is an electron state with $|1\rangle$ ($|\!-\!1\rangle$) the
eigenstate of $s_3$ with respect to the eigenvalue $1$ ($-1$). $|\!\!\uparrow\rangle$ is a nuclear state and
the eigenstate of $\sigma_3$ with respect to the eigenvalue 1. $W$ is set to be identity.

```jl
using QuanEstimation
using LinearAlgebra

# initial state
psi0 = zeros(ComplexF64, 6)
psi0[1] = 1/sqrt(2)
psi0[5] = 1/sqrt(2)
rho0 = psi0*psi0'
# free Hamiltonian
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
decay = [[S3, 2pi/cons]]
# measurement basis vectors (initial guess for SM, CM and SCM)
dim = size(rho0, 1)
C = [QuanEstimation.basis(dim, i) for i in 1:dim]
# POVM (fixed measurement for the CFIM objective in SC)
# a SIC-POVM is informationally complete, so the CFIM is not singular
M = QuanEstimation.SIC(dim)
# time length for the evolution
# 4000 intervals are divided into 10 control segments
tspan = range(0., 2., length=4001)
cnum = 10
ctrl = [-0.2*ones(cnum) for _ in 1:length(Hc)]
```

### SM
```jl
opt = QuanEstimation.SMopt(psi=psi0, M=C, seed=1234)
dynamics = QuanEstimation.Lindblad(H0, dH, tspan, Hc, decay;
                                   ctrl=ctrl, dyn_method=:Expm)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
obj = QuanEstimation.CFIM_obj()
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

### SC
```jl
opt = QuanEstimation.SCopt(psi=psi0, ctrl=ctrl, ctrl_bound=[-0.2,0.2],
                           seed=1234)
dynamics = QuanEstimation.Lindblad(H0, dH, tspan, Hc, decay;
                                   ctrl=ctrl, dyn_method=:Expm)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
```

##### QFIM
```jl
# objective function: tr(WF^{-1})
obj = QuanEstimation.QFIM_obj()
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

##### CFIM
```jl
# objective function: tr(WI^{-1}) with the fixed POVM M
obj = QuanEstimation.CFIM_obj(M=M)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

##### HCRB
```jl
# objective function: HCRB
obj = QuanEstimation.HCRB_obj()
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
```

##### QFIM
```jl
# objective function: tr(WF^{-1})
obj = QuanEstimation.QFIM_obj()
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

##### CFIM
```jl
# objective function: tr(WI^{-1}) with the fixed POVM M
obj = QuanEstimation.CFIM_obj(M=M)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

##### HCRB
```jl
# objective function: HCRB
obj = QuanEstimation.HCRB_obj()
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### AD
```jl
# comprehensive optimization algorithm: AD
alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01,
                        beta1=0.90, beta2=0.99)
```

##### QFIM
```jl
# objective function: tr(WF^{-1})
obj = QuanEstimation.QFIM_obj()
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

##### CFIM
```jl
# objective function: tr(WI^{-1}) with the fixed POVM M
obj = QuanEstimation.CFIM_obj(M=M)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

### CM
```jl
opt = QuanEstimation.CMopt(ctrl=ctrl, M=C, ctrl_bound=[-0.2,0.2],
                           seed=1234)
dynamics = QuanEstimation.Lindblad(H0, dH, tspan, Hc, decay;
                                   ctrl=ctrl, dyn_method=:Expm)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
obj = QuanEstimation.CFIM_obj()
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

### SCM
```jl
opt = QuanEstimation.SCMopt(psi=psi0, ctrl=ctrl, M=C,
                            ctrl_bound=[-0.2,0.2], seed=1234)
dynamics = QuanEstimation.Lindblad(H0, dH, tspan, Hc, decay;
                                   ctrl=ctrl, dyn_method=:Expm)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
obj = QuanEstimation.CFIM_obj()
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

The outputs are `f.csv` together with `states.dat` and `measurements.dat` for SM,
`controls.dat` and `states.dat` for SC, `controls.dat` and `measurements.dat` for CM,
and `controls.dat`, `states.dat` and `measurements.dat` for SCM. They can be loaded with

```jl
using DelimitedFiles
using JLD2

f = readdlm("f.csv")
if isfile("states.dat")
    states = jldopen("states.dat", "r") do file
        file["states"]
    end
end
if isfile("controls.dat")
    controls = jldopen("controls.dat", "r") do file
        file["controls"]
    end
end
if isfile("measurements.dat")
    measurements = jldopen("measurements.dat", "r") do file
        file["measurements"]
    end
end
```

Only the files produced by the chosen scenario exist.

For optimization of probe state and measurement, the parameterization can also be implemented
with the Kraus operators which can be realized by
```jl
# Kraus operators and their derivatives
dynamics = Kraus(K, dK)
scheme = GeneralScheme(probe=rho0, param=dynamics)

opt = SMopt(psi=psi0, M=C, seed=1234)
alg = DE(p_num=10, ini_population=nothing, max_episode=1000, c=1.0, cr=0.5)
obj = CFIM_obj()
optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
```
where `K` and `dK` are the Kraus operators and its derivatives with respect to the
unknown parameters, `rho0` is the probe state, and `C` is the initial projective measurement
basis used by `SMopt`.

**Example 8.3**
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

where $\gamma$ is the unknown parameter to be estimated which represents the decay
probability. In this example, the probe state is taken as $|+\rangle\langle+|$ with
$|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$. Here $|0\rangle$ $(|1\rangle)$ is
the eigenstate of $\sigma_3$ (Pauli matrix) with respect to the eigenvalue $1$ $(-1)$.

```jl
using QuanEstimation
using LinearAlgebra

# initial state
psi0 = ComplexF64[1.0, 1.0]/sqrt(2)
rho0 = psi0*psi0'
# Kraus operators for the amplitude damping channel
gamma = 0.1
K1 = [1. 0.; 0. sqrt(1-gamma)]
K2 = [0. sqrt(gamma); 0. 0.]
K = [K1, K2]
# derivatives of Kraus operators on gamma
dK1 = [1. 0.; 0. -0.5/sqrt(1-gamma)]
dK2 = [0. 0.5/sqrt(gamma); 0. 0.]
dK = [[dK1], [dK2]]
# comprehensive optimization
dim = size(rho0, 1)
C = [QuanEstimation.basis(dim, i) for i in 1:dim]
opt = QuanEstimation.SMopt(psi=psi0, M=C, seed=1234)
# dynamics and scheme
dynamics = QuanEstimation.Kraus(K, dK)
scheme = QuanEstimation.GeneralScheme(probe=rho0, param=dynamics)
obj = QuanEstimation.CFIM_obj()
```

#### DE
```jl
# comprehensive optimization algorithm: DE
alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

#### PSO
```jl
# comprehensive optimization algorithm: PSO
alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
QuanEstimation.optimize!(scheme, opt; algorithm=alg, objective=obj,
                         savefile=false)
```

The outputs are `f.csv`, `states.dat` and `measurements.dat`, which can be loaded with

```jl
using DelimitedFiles
using JLD2

f = readdlm("f.csv")
states = jldopen("states.dat", "r") do file
    file["states"]
end
measurements = jldopen("measurements.dat", "r") do file
    file["measurements"]
end
```

---
## **Bibliography**

```@raw html
<a id="Kennedy1995"></a>
```

**[1]**
J. Kennedy and R. Eberhart,
Particle swarm optimization,
[Proc. 1995 IEEE International Conference on Neural Networks **4**, 1942-1948 (1995).
](https://doi.org/10.1109/ICNN.1995.488968)

```@raw html
<a id="Storn1997"></a>
```

**[2]**
R. Storn and K. Price,
Differential Evolution-A Simple and Efficient Heuristic for global
Optimization over Continuous Spaces,
[J. Global Optim. **11**, 341 (1997).](https://doi.org/10.1023/A:1008202821328)

```@raw html
<a id="Baydin2018"></a>
```

**[3]**
A. G. Baydin, B. A. Pearlmutter, A. A. Radul, and J. M. Siskind,
Automatic differentiation in machine learning: a survey,
[J. Mach. Learn. Res. **18**, 1-43 (2018).](http://jmlr.org/papers/v18/17-468.html)

**See also**: [Parameterization process](guide_dynamics.md) — construct the system dynamics. [Quantum metrological tools](guide_bounds.md) — the objective functions being optimized. [Output files](output_files.md) — how to load the saved results.
