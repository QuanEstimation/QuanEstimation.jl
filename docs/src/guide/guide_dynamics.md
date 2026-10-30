# **Parameterization process**

In QuanEstimation, two types of parameterization processes are considered. The first one is
the master equation of the form

```math
\begin{align*}
\partial_t\rho &=\mathcal{L}\rho \\
&=-i[H,\rho]+\sum_i \gamma_i\left(\Gamma_i\rho\Gamma^{\dagger}_i-\frac{1}{2}
\left\{\rho,\Gamma^{\dagger}_i \Gamma_i \right\}\right),
\end{align*}
```

where $\rho$ is the evolved density matrix, $H$ is the Hamiltonian of the system, $\Gamma_i$
and $\gamma_i$ are the $i\mathrm{th}$ decay operator and the corresponding decay rate.
Numerically, the evolved state at $j$th time interval is obtained by $\rho_j=e^{\Delta t\mathcal{L}}
\rho_{j-1}$ with $\Delta t$ the time interval. The derivatives of $\rho_j$ on $\textbf{x}$
is calculated via
```math
\begin{align}
\partial_{\textbf{x}}\rho_j =\Delta t(\partial_{\textbf{x}}\mathcal{L})\rho_j
+e^{\Delta t \mathcal{L}}(\partial_{\textbf{x}}\rho_{j-1}),
\end{align}
```
where $\rho_{j-1}$ is the evolved density matrix at $(j-1)$th time interval.

The dynamics can also be solved by the ordinary differential equation (ODE) solvers, thus
$\rho$ and $\partial_{\textbf{x}}\rho$ are obtained via directly solving the differential equations.
Here, $\partial_{\textbf{x}}\rho$ satisfies

```math
\begin{align}
\partial_t(\partial_{\textbf{x}}\rho) =-i[\partial_{\textbf{x}}H,\rho] + \mathcal{L}(\partial_{\textbf{x}}\rho).
\end{align}
```

The evolved density matrix $\rho$ and its derivatives ($\partial_{\textbf{x}}\rho$) with
respect to $\textbf{x}$ can be calculated by

### Using the matrix exponential method (`expm`)

```jl
rho, drho = expm(tspan, rho0, H0, dH; decay=nothing, Hc=nothing,
                 ctrl=nothing)
```

### Using the ODE solver method (`ode`)

```jl
rho, drho = ode(tspan, rho0, H0, dH; decay=nothing, Hc=nothing,
                ctrl=nothing, abstol=1e-8, reltol=1e-8)
```

Here `tspan` is the time length for the evolution and `rho0` represents the density matrix of
the probe state. `H0` is the free Hamiltonian and `dH` contains its derivatives with respect
to the unknown parameters to be estimated, `dH = [∂a H0, ∂b H0, ...]`. For a single parameter,
the low-level `expm` and `ode` functions also accept a single derivative matrix in place of
the one-element vector. In the direct calls above, `H0` must be a single matrix; a vector of
matrices is not a valid input in this interface. Parameter-dependent Hamiltonians can instead be
provided as functions through the `Hamiltonian`/`Lindblad` scheme interface described below.

`decay` contains decay operators $(\Gamma_1, \Gamma_2, \cdots)$ and the corresponding decay
rates $(\gamma_1, \gamma_2, \cdots)$ with the input rule
`decay=[[Γ1, γ1], [Γ2, γ2], ...]`. Each decay rate must be a scalar; arrays of
time-dependent rates are not accepted by these direct `expm` and `ode` calls. `Hc` and `ctrl` are
two lists representing the control Hamiltonians and the corresponding control coefficients.
The default values for `decay`, `Hc`, and `ctrl` are `nothing`, which means the dynamics is
unitary and only governed by the free Hamiltonian. The ODE solver additionally accepts the
`abstol` and `reltol` keywords shown above; `expm` does not take tolerance keywords.

The output (`rho` and `drho`) of `expm()` (`ode()`) are two lists with the length equal to
`tspan`. Here `rho` represents the parameterized density matrix and `drho` is the
corresponding derivatives with respect to all the parameters; the $i$th entry of `drho`
is $[\partial_a{\rho},\partial_b{\rho},\cdots]$.

### Using the scheme interface

The same Lindblad dynamics can be packaged in a `GeneralScheme` and evolved with `evolve`:

```jl
# unitary dynamics
dynamics = Lindblad(H0, dH, tspan; dyn_method=:Expm)

# with decay (decay is a positional argument)
# dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)

# with controls
# dynamics = Lindblad(H0, dH, tspan, Hc; ctrl=ctrl, dyn_method=:Expm)

scheme = GeneralScheme(; probe=rho0, param=dynamics)
rho_final, drho_final = evolve(scheme)   # final state and its derivatives
rho_t, drho_t = expm(scheme)             # full trajectory; use ode(scheme) for the ODE solver
```

Here `dyn_method=:Expm` selects the matrix exponential solver and `dyn_method=:Ode` selects
the ODE solver. `evolve(scheme)` returns only the final state, whereas `expm(scheme)` and
`ode(scheme)` return the full time trajectory. For a parameter-dependent Hamiltonian, store
the parameter values in a `Hamiltonian` and pass it to `Lindblad`:

```jl
H0_func(p) = 0.5 * p * sz          # p is the parameter value
dH_func(p) = [0.5 * sz]

ham = Hamiltonian(H0_func, dH_func, omega)   # omega = 1.0
dynamics = Lindblad(ham, tspan; dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dynamics)
rho, drho = evolve(scheme)
```

The functions stored in `Hamiltonian` are called with the stored parameter values, so a
multi-parameter problem should use functions with one argument per estimated parameter (or a
`params` tuple/vector matching that signature).

**Example 2.1**
In this example, the free evolution Hamiltonian of a single qubit system is $H_0=\frac{1}{2}
\omega \sigma_3$ with $\omega$ the frequency and $\sigma_3$ a Pauli matrix.
The dynamics of the system is governed by
```math
\begin{align}
\partial_t\rho=-i[H_0, \rho],
\end{align}
```

where $\rho$ is the parameterized density matrix. The probe state is taken as $|+\rangle\langle+|$
with $|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$. Here $|0\rangle$ $(|1\rangle)$ is the
eigenstate of $\sigma_3$ with respect to the eigenvalue $1$ $(-1)$.

```jl
using QuanEstimation

# initial state
rho0 = 0.5*ones(2, 2)
# free Hamiltonian
omega = 1.0
sz = [1. 0.0im; 0. -1.]
H0 = 0.5*omega*sz
# derivative of the free Hamiltonian on omega
dH = [0.5*sz]
# time length for the evolution
tspan = range(0., 10., length=2500)
# dynamics
rho, drho = expm(tspan, rho0, H0, dH)
```

The parameterization process can also be implemented with the Kraus operators. In this case,
the parameterized density matrix and its derivatives with respect to the unknown parameters
can be calculated via
```math
\begin{align}
\rho=\sum_i K_i\rho_0K_i^{\dagger},
\end{align}
```

```math
\begin{align}
\partial_{\textbf{x}}\rho=\sum_i \partial_{\textbf{x}}K_i\rho_0K_i^{\dagger}
+ K_i\rho_0\partial_{\textbf{x}}K_i^{\dagger},
\end{align}
```

where $K_i$ is a Kraus operator satisfying $\sum_{i}K^{\dagger}_i K_i=I$ and $\partial_{\textbf{x}}K_i$
represents its derivatives with respect to $\textbf{x}$. Here $I$ is the identity
operator and $\rho_0$ is the probe state.

In QuanEstimation, a Kraus parameterization is constructed with `Kraus(K, dK)` and then
combined with the probe state in a `GeneralScheme`:

```jl
param = Kraus(K, dK)
scheme = GeneralScheme(; probe=rho0, param=param)
rho, drho = evolve(scheme)
```

Here `K = [K1, K2, ...]` is the vector of Kraus operators and
`dK = [[∂a K1, ∂b K1, ...], [∂a K2, ∂b K2, ...], ...]` contains their derivatives, with the
outer index labelling the Kraus operator and the inner index labelling the estimated
parameter. Function-valued Kraus operators can be supplied as
`Kraus(K_func, dK_func, params)`, where `K_func(params...)` and `dK_func(params...)` return
the same structures.

**Example 2.2**
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

where $\gamma$ is the decay probability. In this example,
the probe state is taken as $|+\rangle\langle+|$ with $|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+
|1\rangle)$. Here $|0\rangle$ $(|1\rangle)$ is the eigenstate of $\sigma_3$ with respect to the
eigenvalue $1$ $(-1)$.

```jl
using QuanEstimation

# initial state
rho0 = 0.5*ones(2, 2)
# Kraus operators for the amplitude damping channel
gamma = 0.1
K1 = [1. 0.; 0. sqrt(1-gamma)]
K2 = [0. sqrt(gamma); 0. 0.]
K = [K1, K2]
# derivatives of Kraus operators on gamma
dK1 = [1. 0.; 0. -0.5/sqrt(1-gamma)]
dK2 = [0. 0.5/sqrt(gamma); 0. 0.]
dK = [[dK1], [dK2]]
# parameterization process
param = Kraus(K, dK)
scheme = GeneralScheme(; probe=rho0, param=param)
rho, drho = evolve(scheme)
```
