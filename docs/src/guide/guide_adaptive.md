# **Adaptive measurement schemes**
In QuanEstimation, the Hamiltonian of the adaptive system should be written as
$H(\textbf{x}+\textbf{u})$ with $\textbf{x}$ the unknown parameters and $\textbf{u}$
the tunable parameters. The tunable parameters $\textbf{u}$ are used to let the
Hamiltonian work at the optimal point $\textbf{x}_{\mathrm{opt}}$. In this scenario,
the adaptive estimation is represented by the same `GeneralScheme` object used for
the other metrological tasks, with the prior stored in an `AdaptiveStrategy`:
``` jl
strategy = AdaptiveStrategy(x=x, p=p, dp=dp)
dynamics = Lindblad(H0_func, dH_func, tspan; dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dynamics, strat=strategy)
res = rand(0:1, max_episode)      # simulated measurement outcomes (0/1)
adapt!(scheme; method="FOP", max_episode=max_episode, res=res)
```
where `x` is a vector of arrays representing the regime of the parameters for the
integral (for a single parameter, `x=[xgrid]`), `p` is an array representing the prior
distribution (multidimensional for multiparameter estimation), and `dp` is its derivative.
`rho0` is the density matrix
of the probe state; `H0_func` and `dH_func` return the Hamiltonian and its derivatives
at a parameter point (they receive the parameter values themselves; for a single
parameter this is a scalar). The number of iterations can be set via `max_episode` with the
default value 1000. `eps` is a legacy keyword; when it is `nothing`, the relative
precision policy described in the metrological-tools guide is used. If `res=nothing`,
the experimental outcome is read interactively; otherwise `res` supplies the simulated
outcomes and must contain at least `max_episode` entries. At the end of the program a single
JLD2 file, `adaptive.dat`, is generated
with keys `"p"` (posterior distributions), `"x"` (estimated values) and `"y"`
(experimental results). With `savefile=false` (the default), `"p"` contains only the
final posterior while `"x"` and `"y"` keep the history of all episodes; with
`savefile=true`, all three record every episode. The package contains two methods for
updating the tunable parameters.
The first one is
updating the tunable parameters with a fixed optimal point (`method="FOP"`), which is the default
method in QuanEstimation. The other is `method="MI"` which means updating the tunable parameters
by maximizing the mutual information which is defined as

```math
\begin{equation}
I(\textbf{u})=\int\mathrm{p}(\textbf{x}) \sum_{y}\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{log}_2
\left[\frac{\mathrm{p}(y|\textbf{x},\textbf{u})}{\int\mathrm{p}(\textbf{x})\mathrm{p}(y|\textbf{x},\textbf{u})\mathrm{d}\textbf{x}}\right]\mathrm{d}\textbf{x}.
\end{equation}
```

If `savefile=true`, the posterior distribution is appended during the training;
otherwise the posterior distribution in the final iteration is stored. In both cases the
estimates and experimental results from all episodes are stored in `adaptive.dat`, as
described above.

If the dynamics of the system can be described by the master equation, pass the dynamics
data `tspan` together with the Hamiltonian callables to `Lindblad`, for example
`Lindblad(H0_func, dH_func, tspan; dyn_method=:Expm)`. `tspan` is the time length for the
evolution. To tabulate the Hamiltonian and its derivatives on a parameter grid for direct
(non-scheme) routines, use the exported helper
``` jl
H, dH = QuanEstimation.BayesInput(x, func, dfunc; channel="dynamics")
```
(`channel="Kraus"` instead returns `K, dK`). Here `func` and `dfunc` are the functions
defined by the users which return `H` and `dH`, respectively, for a grid point. Furthermore,
for systems with noise and controls, the positional `decay` argument and the `Hc`/`ctrl`
arguments are passed to `Lindblad`, e.g.
`Lindblad(H0_func, dH_func, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)`. Here `Hc` and
`ctrl` are two lists representing the control Hamiltonians and the corresponding control
coefficients. `decay` contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the
corresponding decay rates $(\gamma_1, \gamma_2, \cdots)$ with the input rule
decay=[[$\Gamma_1$, $\gamma_1$], [$\Gamma_2$, $\gamma_2$],...].

The adaptive update is based on the CFIM. The fixed-point method
(`method="FOP"`) selects the parameter point that maximizes
$1/\mathrm{Tr}(W\mathcal{I}^{-1})$ (equivalently, minimizes
$\mathrm{Tr}(W\mathcal{I}^{-1})$); `method="MI"` maximizes the mutual information
$I(\textbf{u})$ defined above. Here $\mathcal{I}$ is the CFIM and `W` is the weight matrix,
which defaults to the identity matrix.

Adaptive estimation currently supports Hamiltonian parameterization
(`Lindblad` dynamics). Kraus-parameterized schemes can be used with the
non-adaptive bounds, but `adapt!` does not provide a Kraus update rule.

**Example 9.1**
The Hamiltonian of a qubit system is
```math
\begin{align}
H=\frac{B\omega_0}{2}(\sigma_1\cos{x}+\sigma_3\sin{x}),
\end{align}
```

where $B$ is the magnetic field in the XZ plane, $x$ is the unknown parameter and $\sigma_{1}$, $\sigma_{3}$ are the Pauli matrices.

The probe state is taken as $|\pm\rangle$. The measurement is
$\{|\!+\rangle\langle+\!|,|\!-\rangle\langle-\!|\}$. Here $|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$ with $|0\rangle$ $(|1\rangle)$ the eigenstate of $\sigma_3$ with respect to the eigenvalue $1$ $(-1)$. In this example, the prior distribution $p(x)$ is uniform.
``` jl
using QuanEstimation
using Random
using StatsBase

# free Hamiltonian
function H0_func(x)
    return 0.5*B*omega0*(sx*cos(x)+sz*sin(x))
end
# derivative of free Hamiltonian in x
function dH_func(x)
    return [0.5*B*omega0*(-sx*sin(x)+sz*cos(x))]
end

B, omega0 = pi/2.0, 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
# initial state
rho0 = 0.5*ones(2, 2)
# measurement
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., stop=1., length=1000) |>Vector
# prior distribution
x = range(-0.25*pi+0.1, stop=3.0*pi/4.0-0.1, length=1000) |>Vector
p = (1.0/(x[end]-x[1]))*ones(length(x))
# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
for i = 1:length(x)
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = QuanEstimation.expm(tspan, rho0, H0_tp, dH_tp)
    rho[i] = rho_tp[end]
end
# Bayesian estimation
Random.seed!(1234)
y = [0 for i in 1:500]
res_rand = sample(1:length(y), 125, replace=false)
for i in 1:length(res_rand)
    y[res_rand[i]] = 1
end
pout, xest = QuanEstimation.Bayes([x], p, rho, y, M=M, estimator="MAP", savefile=false)
# adaptive measurement
dynamics = QuanEstimation.Lindblad(H0_func, dH_func, tspan; dyn_method=:Expm)
strategy = QuanEstimation.AdaptiveStrategy(x=[x], p=pout, dp=[zeros(length(x))])
scheme = QuanEstimation.GeneralScheme(; probe=rho0, param=dynamics,
                                      measurement=M, strat=strategy)
res = rand(0:1, 100)
QuanEstimation.adapt!(scheme; method="FOP", max_episode=100, res=res)
```
---
Berry et al. [[1,2]](#Berry2000) introduced a famous adaptive scheme in phase estimation. The
phase for the $(n+1)$th round is updated via $\Phi_{n+1}=\Phi_{n}-(-1)^{y^{(n)}}\Delta
\Phi_{n+1}$ with $y^{(n)}$ the experimental result in the $n$th round and $\Delta\Phi_{n+1}$
the phase difference generated by the proper algorithms. This adaptive scheme can be performed
in QuanEstimation via
``` jl
apt = Adapt_MZI(x, p, rho0)
online(apt, target="sharpness", output="phi")
```
Here `x`, `p`, and `rho0` are the same as in the general adaptive scheme above.
`target="sharpness"` selects the sharpness as the target function for the tunable phase;
it can also be set to `"MI"` (mutual information). The offline routines accept the
corresponding symbols `:sharpness` and `:MI`. The output can be set
through `output="phi"` (default) and `output="dphi"` representing the phase and phase
difference, respectively. As in `adapt!`, `res` can be supplied to `online` to provide the
measurement outcomes; when `res=nothing` the outcomes are read interactively. Online and
offline strategies are both available in the package
and the code for calling offline strategy becomes `alg = QuanEstimation.DE(kwargs...)`
(`alg = QuanEstimation.PSO(kwargs...)`) and `offline(apt, alg, seed=seed)`.
`seed` is the random seed which can ensure the reproducibility of results. The `online`
routine writes `adaptive.dat` with keys `"x"` (the feedback phases or phase differences)
and `"y"` (the measurement outcomes); the MZI `offline` optimization writes the optimal
phase differences to `deltaphi.csv` and the objective history to `f.csv`.

If the optimization algorithm is PSO, the keywords and the default values are
``` jl
alg = PSO(p_num=10, ini_particle=nothing, max_episode=[1000,100],
          c0=1.0, c1=2.0, c2=2.0)
```
The keywords and the default values of PSO can be seen in the following table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "p_num"                          | 10                         |
| "ini_particle"                   | nothing                    |
| "max_episode"                    | [1000,100]                 |
| "c0"                             | 1.0                        |
| "c1"                             | 2.0                        |
| "c2"                             | 2.0                        |

Here `p_num` is the number of particles, `ini_particle` represents the initial guesses
of phase difference. `max_episode` accepts both integer and array with two elements.
If it is an integer, for example `max_episode=1000`, it means the program will
continuously run 1000 episodes. However, if it is an array, for example
`max_episode=[1000,100]`, the program will run 1000 episodes in total but replace the
data of all the particles with global best every 100 episodes. `c0`, `c1` and `c2` are
the PSO parameters representing the inertia weight, cognitive learning factor and
social learning factor, respectively.

If the optimization algorithm is DE, the keywords and the default values are
```jl
alg = DE(p_num=10, ini_population=nothing, max_episode=1000,
         c=1.0, cr=0.5)
```
The keywords and the default values of DE can be seen in the following table

| $~~~~~~~~~~$keywords$~~~~~~~~~~$ | $~~~~$default values$~~~~$ |
| :----------:                     | :----------:               |
| "p_num"                          | 10                         |
| "ini_population"                 | nothing                    |
| "max_episode"                    | 1000                       |
| "c"                              | 1.0                        |
| "cr"                             | 0.5                        |

`ini_population` represents the initial guesses of phase difference. `p_num` and
`max_episode` are the number of populations and training episodes. `c` and `cr` are
DE parameters representing the mutation and crossover constants.

**Example 9.2**
In this example, the adaptive measurement scheme is designed for the MZI [[3,4]](#Hentschel2010).
The input state is
```math
\begin{align}
\sqrt{\frac{2}{N+2}}\sum^{N/2}_{m=-N/2}\sin\left(\frac{(2m+N+2)\pi}{2(N+2)}\right)|m\rangle,
\end{align}
```

where $N$ is the number of photon, $|m\rangle$ is the eigenstate of $J_y$ with the eigenvalue
$m$.
``` jl
using QuanEstimation
using SparseArrays

# the number of photons
N = 8
# probe state
psi = sum([sin(k*pi/(N+2))*kron(QuanEstimation.basis(N+1,k),
      QuanEstimation.basis(N+1, N-k+2)) for k in 1:(N+1)]) |> sparse
psi = psi*sqrt(2/(2+N))
rho0 = psi*psi'
# prior distribution
x = range(-pi, pi, length=100)
p = (1.0/(x[end]-x[1]))*ones(length(x))
apt = QuanEstimation.Adapt_MZI(x, p, rho0)
```
**Online:**
```jl
QuanEstimation.online(apt, target="sharpness", output="phi")
```

**Offline with DE:**
```jl
alg = QuanEstimation.DE(p_num=10, ini_population=nothing,
                        max_episode=1000, c=1.0, cr=0.5)
QuanEstimation.offline(apt, alg, target=:sharpness, seed=1234)
```

**Offline with PSO:**
```jl
alg = QuanEstimation.PSO(p_num=10, ini_particle=nothing,
                         max_episode=[1000,100], c0=1.0,
                         c1=2.0, c2=2.0)
QuanEstimation.offline(apt, alg, target=:sharpness, seed=1234)
```
---
## **Bibliography**

```@raw html
<a id="Berry2000"></a>
```

**[1]**
D. W. Berry and H. M. Wiseman,
Optimal States and Almost Optimal Adaptive Measurements for Quantum Interferometry,
[Phys. Rev. Lett. **85**, 5098 (2000).](https://doi.org/10.1103/PhysRevLett.85.5098)

**[2]**
D. W. Berry, H. M. Wiseman, and J. K. Breslin,
Optimal input states and feedback for interferometric phase estimation,
[Phys. Rev. A **63**, 053804 (2001).](https://doi.org/10.1103/PhysRevA.63.053804)

```@raw html
<a id="Hentschel2010"></a>
```

**[3]**
A. Hentschel and B. C. Sanders,
Machine Learning for Precise Quantum Measurement,
[Phys. Rev. Lett. **104**, 063603 (2010).](https://doi.org/10.1103/PhysRevLett.104.063603)

**[4]**
A. Hentschel and B. C. Sanders,
Efficient Algorithm for Optimizing Adaptive Quantum Metrology Processes,
[Phys. Rev. Lett. **107**, 233601 (2011).](https://doi.org/10.1103/PhysRevLett.107.233601)

**See also**: [Quantum metrological tools](guide_bounds.md) — Bayesian estimation provides the prior distribution for adaptive schemes.
