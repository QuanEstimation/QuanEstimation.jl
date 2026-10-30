# **Quantum metrological tools**
QuanEstimation can be used to calculate several well-used metrological tools including
Quantum Cramér-Rao bounds, Holevo Cramér-Rao bound, Bayesian Cramér-Rao bounds, Quantum
Ziv-Zakai bound and perform Bayesian estimation.

Notes: Results may vary due to numerical precision (e.g., the retained significant digits
in calculations). This difference has no effect on optimization. To get consistent results,
use the same number of significant digits (e.g., keep 8 decimal places).

## **Quantum Cramér-Rao bounds**
In quantum metrology, quantum Cramér-Rao bounds are well used metrological tools for
parameter estimation. It can be expressed as [[1,2,3]](#Helstrom1976)
```math
\begin{align}
\mathrm{cov}\left(\hat{\textbf{x}}, \{\Pi_y\}\right) \geq \frac{1}{n}\mathcal{I}^{-1}
\left(\{\Pi_y\}\right) \geq \frac{1}{n} \mathcal{F}^{-1},
\end{align}
```

where $\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\sum_y\mathrm{Tr}(\rho\Pi_y)(\hat{\textbf{x}}
-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}$ is the covariance matrix for the
unknown parameters $\hat{\textbf{x}}=(\hat{x}_0,\hat{x}_1,\dots)^{\mathrm{T}}$ to be estimated.
$\{\Pi_y\}$ is a set of positive operator-valued measure (POVM) and $\rho$ represents the
parameterized density matrix. $n$ is the repetition of the experiment, $\mathcal{I}$ and
$\mathcal{F}$ are the classical Fisher information matrix (CFIM) and quantum Fisher information
matrix (QFIM), respectively. The $ab$th entry of CFIM is defined as
```math
\begin{align}
\mathcal{I}_{ab}=\sum_y\frac{1}{p(y|\textbf{x})}[\partial_a p(y|\textbf{x})][\partial_b
p(y|\textbf{x})]
\end{align}
```

with $p(y|\textbf{x})=\mathrm{Tr}(\rho\Pi_y)$. The most well-used type of the QFIM is
SLD-based QFIM of the form
```math
\begin{align}
\mathcal{F}_{ab}=\frac{1}{2}\mathrm{Tr}[\rho (L_aL_b+ L_bL_a)]
\end{align}
```

with $\mathcal{F}_{ab}$ the $ab$th entry of $\mathcal{F}$ and $L_{a}(L_{b})$ the symmetric
logarithmic derivative (SLD) operator for $x_{a}(x_b)$. The SLD operator is determined by
```math
\begin{align}
\partial_{a}\rho=\frac{1}{2}(\rho L_{a}+L_{a}\rho).
\end{align}
```

The $ij$th entry of SLD can be calculated by
```math
\begin{align}
\langle\lambda_i|L_{a}|\lambda_j\rangle=\frac{2\langle\lambda_i| \partial_{a}\rho
|\lambda_j\rangle}
{\lambda_i+\lambda_j}, ~~\lambda_i (\lambda_j)\neq 0 .
\end{align}
```

For $\lambda_i (\lambda_j)=0$, the above equation is set to be zero.

Besides, there are right logarithmic derivative (RLD) and left logarithmic derivative (LLD)
defined by $\partial_{a}\rho=\rho \mathcal{R}_a$ and $\partial_{a}\rho=\mathcal{R}_a^{\dagger}
\rho$ with the  corresponding QFIM  $\mathcal{F}_{ab}=\mathrm{Tr}(\rho \mathcal{R}_a
\mathcal{R}^{\dagger}_b)$. The RLD and LLD operators are calculated via

```math
\begin{align}
\langle\lambda_i| \mathcal{R}_{a} |\lambda_j\rangle
&= \frac{1}{\lambda_i}\langle\lambda_i| \partial_{a}\rho |\lambda_j\rangle,~~\lambda_i\neq 0, \\
\langle\lambda_i| \mathcal{R}_{a}^{\dagger} |\lambda_j\rangle
&= \frac{1}{\lambda_j}\langle\lambda_i| \partial_{a}\rho |\lambda_j\rangle,~~\lambda_j\neq 0.
\end{align}
```

In QuanEstimation, three types of the logarithmic derivatives can be solved by calling the
codes
``` jl
SLD(rho, drho; rep="original", tol=1e-10, input_error=0.0, guard=nothing,
    refine=true, diagnostics=false)
```
``` jl
RLD(rho, drho; rep="original", tol=1e-10, input_error=0.0, guard=nothing,
    diagnostics=false)
```
``` jl
LLD(rho, drho; rep="original", tol=1e-10, input_error=0.0, guard=nothing,
    diagnostics=false)
```
where `rho` and `drho` are the density matrix of the state and its derivatives with respect to
the unknown parameters to be estimated. `drho` should be input as $[\partial_a{\rho},
\partial_b{\rho}, \cdots]$. For single parameter estimation (the length of `drho` is equal to
one), the output is a matrix and for multiparameter estimation (the length of `drho` is more
than one), it returns a vector. There are two output choices for the logarithmic derivatives
basis which can be set through `rep`. The default basis (`rep="original"`) of the logarithmic
derivatives is the same with `rho` and the users can also request the logarithmic derivatives
written in the eigenspace of `rho` by `rep="eigen"`. The evaluation is controlled by the
relative precision keywords described in
[Numerical precision policy and diagnostics](#numerical-precision-policy-and-diagnostics):
`tol`, `input_error`, `guard` and `diagnostics`. `SLD` additionally accepts `refine=true`;
`RLD` and `LLD` do not accept `refine` and passing it is an error, because only the SLD
solver performs residual refinement. The legacy keyword `eps` is deprecated and, when
supplied, is interpreted as a relative spectral-floor factor.

In QuanEstimation, the QFI and QFIM can be calculated via the following function
``` jl
QFIM(rho, drho; LDtype=:SLD, exportLD=false, tol=1e-10, input_error=0.0,
     guard=nothing, refine=true, diagnostics=false)
```
`LDtype` represents the types of QFI (QFIM) can be set. Options are `LDtype=:SLD` (default),
`LDtype=:RLD` and `LDtype=:LLD`. This function will return QFI (QFIM) if `exportLD=false`,
however, if the users set `exportLD=true`, it will return logarithmic derivatives apart
from QFI (QFIM). `refine` controls residual refinement only for `LDtype=:SLD`; it is accepted
but has no effect for `LDtype=:RLD` and `LDtype=:LLD` (and the corresponding `RLD`/`LLD`
functions themselves do not accept it). With `diagnostics=true` it additionally returns a
`NumericalDiagnostics` record, see below.

### **Numerical precision policy and diagnostics**

```@raw html
<a id="numerical-precision-policy-and-diagnostics"></a>
```

All Fisher-information and logarithmic-derivative routines share one relative precision
policy. Its five quantities are:

| quantity | control | role |
|----------|---------|------|
| unit roundoff ``u`` | working type | machine precision (``2.2\times10^{-16}`` for `Float64`, ``1.19\times10^{-7}`` for `Float32`) |
| input-data error ``\eta`` | `input_error` | relative error of the supplied state and its derivatives (default `0.0`) |
| relative spectral floor ``\tau`` | `guard` | eigenvalues, probabilities or singular values below ``\tau`` times the largest one are treated as zero; defaults to `default_guard(T)` (``10^{-14}`` for `Float64`) |
| stopping threshold ``\tau_{\mathrm{stop}}`` | derived | residual threshold used by the iterative refinement; follows from ``u``, the condition number and the residual precision |
| target accuracy | `tol` | requested relative accuracy of the delivered quantity (default ``10^{-10}``) |

All thresholds are **relative**: they are scaled by the largest eigenvalue (or probability,
or singular value) of the object they guard, so the same keywords can be used for states of
any dimension and normalization. The residual-refinement step is separate from these
thresholds and is controlled by `refine`: `SLD`, `QFIM` with `LDtype=:SLD`, `QFIM_Kraus`, the
QFIM fallback of `HCRB`, and the SLD-based Bayesian routines accept `refine=true`; `RLD` and
`LLD` do not accept it, and the CFIM/FIM/BCFIM solvers do not perform residual refinement.

When `diagnostics=true`, the functions return `(value, diag)` (or `(F, LD, diag)` for `QFIM`
with `exportLD=true`) where `diag` is a `NumericalDiagnostics` record with the fields
`quantity`, `precision_layer`, `spectral_floor`, `numerical_rank`, `backward_error`,
`forward_error_bound`, `input_error_propagation`, `truncation_bias_bound`,
`condition_number`, `branch`, `cross_validation_gap`, `upper_bound`, `deleted_mass`,
`dropped_fisher_mass`, `flags` and `bottleneck`. The `flags` field reports conditions such as
`:resolution_floor` (the returned value is a lower bound because terms were removed),
`:truncation_nonnegligible`, `:rank_change_suspected`, `:input_error_dominant`,
`:stagnated`, `:hermiticity_corrected`, `:negative_eigenvalue`, `:nonfinite_solution`
and `:objective_nullspace`. For example

``` jl
F, diag = QuanEstimation.QFIM(rho, drho; diagnostics=true)
diag.flags
diag.truncation_bias_bound
```

**Example 3.1**

```@raw html
<a id="example3_1"></a>
```

The Hamiltonian of a single qubit system is $H=\frac{1}{2}\omega \sigma_3$ with $\omega$
the frequency and $\sigma_3$ a Pauli matrix. The dynamics of the system is governed by

```math
\begin{align}
\partial_t\rho=-i[H, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}
\sigma_{+},\rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}
\sigma_{-},\rho\}\right),
\end{align}
```

where $\sigma_{\pm}=\frac{1}{2}(\sigma_1 \pm \sigma_2)$ with $\sigma_{1}$, $\sigma_{2}$ Pauli
matrices and $\gamma_{+}$, $\gamma_{-}$ are decay rates. The probe state is taken as $|+\rangle$
with $|+\rangle=\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$. Here $|0\rangle$ and $|1\rangle$ are
the eigenstates of $\sigma_3$ with respect to the eigenvalues $1$ and $-1$.
``` jl
using QuanEstimation

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
# dissipation
sp = [0. 1.; 0. 0.0im]
sm = [0. 0.; 1. 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# time length for the evolution
tspan = range(0., 50., length=2000)
# dynamics (the decay list is a keyword argument)
rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH; decay=decay)
# calculation of the QFI
F = Float64[]
for ti in 2:length(tspan)
    # QFI
    F_tp = QuanEstimation.QFIM(rho[ti], drho[ti])
    append!(F, F_tp)
end
```
If the parameterization process is executed via the Kraus operators, the QFI (QFIM) can be
calculated by calling the function
``` jl
QFIM_Kraus(rho0, K, dK; LDtype=:SLD, exportLD=false, tol=1e-10,
           input_error=0.0, guard=nothing, refine=true, diagnostics=false)
```
where `K` and `dK` are the Kraus operators and the derivatives with respect to the unknown
parameters to be estimated.

**Example 3.2**
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

where $\gamma$ is unknown parameter to be estimated which represents the decay probability.
In this example, the probe state is taken as $|+\rangle\langle+|$ with $|+\rangle:=\frac{1}
{\sqrt{2}}(|0\rangle+|1\rangle)$. $|0\rangle$ $(|1\rangle)$ is the eigenstate of $\sigma_3$
(Pauli matrix) with respect to the eigenvalue $1$ $(-1)$.
``` jl
using QuanEstimation

# initial state
rho0 = [0.5+0im 0.5; 0.5 0.5]
# Kraus operators for the amplitude damping channel
gamma = 0.1
K1 = [1. 0.; 0. sqrt(1-gamma)]
K2 = [0. sqrt(gamma); 0. 0.]
K = [K1, K2]
# derivatives of Kraus operators on gamma
dK1 = [1. 0.; 0. -0.5/sqrt(1-gamma)]
dK2 = [0. 0.5/sqrt(gamma); 0. 0.]
dK = [[dK1], [dK2]]
F = QuanEstimation.QFIM_Kraus(rho0, K, dK)
```

The FI (FIM) for a set of the probabilities `p` can be calculated by
``` jl
FIM(p, dp; tol=1e-10, input_error=0.0, guard=nothing, diagnostics=false)
```
where `dp` is a vector representing the derivatives of the probabilities `p` with respect to
the unknown parameters.

**Example 3.3**
``` jl
using QuanEstimation

p = [0.54, 0.46]
dp = [[0.54], [-0.54]]
F = QuanEstimation.FIM(p, dp)
```

The FI can also be calculated based on the experiment data
``` jl
FI_Expt(y1, y2, dx; ftype=:norm)
```
`y1` and `y2` are two arrays representing the experimental data obtained at $x$ and $x+\delta x$, respectively.  $\delta x$ is a known small drift corresponds to `dx`.
`ftype` represents the distribution that the data follows, which can be chosen as `:norm`,
`:gamma`, `:rayleigh`, and `:poisson`.  `ftype=:norm` represents the normal (Gaussian) distribution
with the probability density function
```math
\begin{align}
f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}
\end{align}
```
for distribution fitting, where $\mu$ and $\sigma$ are the mean and variance of the distribution,
respectively.

`ftype=:gamma` represents the gamma distribution of the form
```math
\begin{align}
f(x)=\frac{x^{\alpha-1}e^{-\beta x}\beta^\alpha}{\Gamma(\alpha)}
\end{align}
```
with $\alpha$ the shape and $\beta$ the rate of the distribution. $\Gamma(\alpha)$ is the gamma function.

If the data follows a rayleigh distribution, the data can be fit through setting
`ftype=:rayleigh`. The probability density function of the rayleigh distribution is
```math
\begin{align}
f(x)=\frac{x-\mu}{\sigma^2}e^{-(x-\mu)^2/2\sigma^2}
\end{align}
```
with $\mu$ and $\sigma$ the mean and variance of the distribution.

`ftype=:poisson` represents a discrete Poisson distribution with the probability mass function
```math
\begin{align}
f(k)=\frac{\lambda^k e^{-\lambda}}{k!},
\end{align}
```
where $k=0,1,2,\cdots$ represents the number of occurrences, $\lambda$ represents the variance
of the data, $!$ is the factorial function.

In quantum metrology, the CFI (CFIM) are solved by
``` jl
CFIM(rho, drho; M=nothing, tol=1e-10, input_error=0.0, guard=nothing,
     diagnostics=false)
```
Here `M` represents a set of positive operator-valued measure (POVM) with default value `nothing`.
In this function, a set of rank-one symmetric informationally complete POVM (SIC-POVM) is used
when `M=nothing`. SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state
which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).

**Example 3.4**

```@raw html
<a id="example3_4"></a>
```

The Hamiltonian of a single qubit system is $H=\frac{1}{2}\omega \sigma_3$ with $\omega$
the frequency and $\sigma_3$ a Pauli matrix. The dynamics of the system is governed by
```math
\begin{align}
\partial_t\rho=-i[H, \rho]+ \gamma_{+}\left(\sigma_{+}\rho\sigma_{-}-\frac{1}{2}\{\sigma_{-}
\sigma_{+}, \rho\}\right)+ \gamma_{-}\left(\sigma_{-}\rho\sigma_{+}-\frac{1}{2}\{\sigma_{+}
\sigma_{-},\rho\}\right),
\end{align}
```

where $\sigma_{\pm}=\frac{1}{2}(\sigma_1 \pm \sigma_2)$ with $\sigma_{1}$, $\sigma_{2}$ Pauli
matrices and $\gamma_{+}$, $\gamma_{-}$ are decay rates. The probe state is taken as $|+\rangle$
and the measurement for CFI is $\{|+\rangle\langle+|, |-\rangle\langle-|\}$ with
$|\pm\rangle=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$. Here $|0\rangle$ and $|1\rangle$ are
the eigenstates of $\sigma_3$ with respect to the eigenvalues $1$ and $-1$.
``` jl
using QuanEstimation

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
# dissipation
sp = [0. 1.; 0. 0.0im]
sm = [0. 0.; 1. 0.0im]
decay = [[sp, 0.0], [sm, 0.1]]
# measurement
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., 50., length=2000)
# dynamics (the decay list is a keyword argument)
rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH; decay=decay)
# calculation of the CFI
Im = Float64[]
for ti in 2:length(tspan)
    # CFI
    I_tp = QuanEstimation.CFIM(rho[ti], drho[ti], M)
    append!(Im, I_tp)
end
```

In Bloch representation, the SLD based QFI (QFIM) is calculated by
``` jl
QFIM_Bloch(r, dr; tol=1e-10, input_error=0.0, guard=nothing, diagnostics=false)
```
`r` and `dr` are the parameterized Bloch vector and its derivatives with respect to the
unknown parameters to be estimated.

**Example 3.5**
The arbitrary single-qubit state can be written as
```math
\begin{align}
|\psi\rangle=\cos\frac{\theta}{2}|0\rangle+e^{i\phi}\sin\frac{\theta}{2}|1\rangle
\end{align}
```

with $\theta$ and $\phi$ the parameters to be estimated. The Bloch vector for this state is
$r=(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)^{\mathrm{T}}$ and the derivatives
with respect to $\theta$ and $\phi$ are
$\partial_\theta r=(\cos\theta\cos\phi, \cos\theta\sin\phi, -\sin\theta)^{\mathrm{T}}$ and
$\partial_\phi r=(-\sin\theta\sin\phi, \sin\theta\cos\phi, 0)^{\mathrm{T}}$
``` jl
using QuanEstimation
using LinearAlgebra

theta, phi = 0.25*pi, 0.25*pi
r = [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
dr_theta = [cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)]
dr_phi = [-sin(theta)*sin(phi), sin(theta)*cos(phi), 0.]
dr = [dr_theta, dr_phi]
F = QuanEstimation.QFIM_Bloch(r, dr)
```

The package can also calculate the SLD based QFI (QFIM) with Gaussian states.
``` jl
QFIM_Gauss(R, dR, D, dD)
```
The variable `R` is the expected value $\left(\langle[\textbf{R}]_i\rangle\right)$ of
$\textbf{R}$ with respect to $\rho$, it is an array representing the first-order moment.
Here $\textbf{R}=(q_1,p_1,q_2,p_2,\dots)^{\mathrm{T}}$ with $q_i=\frac{1}{\sqrt{2}}
(a_i+a^{\dagger}_i)$ and $p_i=\frac{1}{i\sqrt{2}}(a_i-a^{\dagger}_i)$ represents a vector
of quadrature operators. `dR` is a vector of derivatives of `R` with respect to the unknown
parameters. The $i$th entry of `dR` is $\partial_{\textbf{x}} \langle[\textbf{R}]_i\rangle$.
`D` and `dD` represent the second-order moment matrix with the $ij$th entry $D_{ij}=\langle
[\textbf{R}]_i [\textbf{R}]_j+[\textbf{R}]_j [\textbf{R}]_i\rangle/2$ and its derivatives with
respect to the unknown parameters.

**Example 3.6**
The first and second moments [[13]](#Safranek2019) are

```math
\begin{align*}
\langle[\textbf{R}]_i\rangle = \left(\begin{array}{cc}
0   \\
0
\end{array}\right),
D = \lambda\left(\begin{array}{cc}
\cosh 2r & -\sinh 2r \\
-\sinh 2r & \cosh 2r
\end{array}\right),
\end{align*}
```

where $\lambda=\coth\frac{\beta}{2}$.  $r$ and $\beta$ are the parameters to be estimated.
``` jl
using QuanEstimation

dim = 2
r, beta = 0.2, 1.0
Lambda = coth(0.5*beta)
# the first-order moment
R = zeros(dim)
dR = [zeros(dim), zeros(dim)]
D = Lambda*[cosh(2*r) -sinh(2*r); -sinh(2*r) cosh(2*r)]
dD_r = 2*Lambda*[sinh(2*r) -cosh(2*r); -cosh(2*r) sinh(2*r)]
dD_Lambda = 0.5*(Lambda^2-1)*[-cosh(2*r) sinh(2*r); sinh(2*r) -cosh(2*r)]
dD = [dD_r, dD_Lambda]
F = QuanEstimation.QFIM_Gauss(R, dR, D, dD)
```
---
## **Holevo Cramér-Rao bound**
Holevo Cramér-Rao bound (HCRB) is of the form [[4,5]](#Holevo1973)
```math
\begin{align}
\mathrm{Tr}(W\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\}))\geq \min_{\textbf{X},V} \mathrm{Tr}(WV),
\end{align}
```
where $W$ is the weight matrix and $V\geq Z(\textbf{X})$ with $[Z(\textbf{X})]_{ab}=\mathrm{Tr}
(\rho X_a X_b)$. $\textbf{X}=[X_0,X_1,\cdots]$ with $X_i:=\sum_y (\hat{x}_i(y)-x_i)\Pi_y$.
The HCRB can be calculated via semidefinite programming as

```math
\begin{align*}
& \min_{\textbf{X},V}~\mathrm{Tr}(WV), \\
& \mathrm{subject}~\mathrm{to}
\begin{cases}
\left(\begin{array}{cc}
V & \Lambda^{\mathrm{T}}R^{\dagger} \\
R\Lambda & I\\
\end{array}\right)\geq 0, \\
\sum_i[\Lambda]_{ai}\mathrm{Tr}(\lambda_i\partial_b\rho)=\delta_{ab}.
\end{cases}
\end{align*}
```

Here $X_i$ is expanded in a specific basis $\{\lambda_i\}$ as $X_i=\sum_j [\Lambda]_{ij}\lambda_j$,
the Hermitian matrix $Z(\textbf{X})$ satisfies $Z(\textbf{X})=\Lambda^{\mathrm{T}}R^{\dagger}
R\Lambda$. In QuanEstimation, the HCRB can be solved by
``` jl
HCRB(rho, drho, W; tol=1e-10, input_error=0.0, guard=nothing, refine=true,
     diagnostics=false)
```
where `rho` and `drho` are the density matrix of the state and its derivatives with respect to
the unknown parameters to be estimated, respectively. `W` is the weight matrix and is a
**required third positional argument** of this matrix-level method; it must be an
`AbstractMatrix` (for example `W = Matrix{Float64}(I, n, n)` or `W = I(n)`; bare `I` is a `UniformScaling` and is rejected). The scheme-level method
`HCRB(scheme; W=nothing)` defaults `W` to
the identity matrix of size equal to the number of parameters. For a single parameter the
matrix-level method returns the QFI; for a rank-one `W` it returns
$\mathrm{Tr}(WF^{-1})$ from the QFIM; otherwise the semidefinite program above is used. The
numerical evaluation is controlled by the relative precision keywords `tol`, `input_error`,
`guard`, `refine` and `diagnostics`, see
[Numerical precision policy and diagnostics](#numerical-precision-policy-and-diagnostics);
`refine` only affects the QFIM fallback path, not the SDP branch.

## **Nagaoka-Hayashi bound**
Nagaoka-Hayashi bound (NHB) is another available bound in quantum parameter estimation and
it is tighter than HCRB in general. The NHB can be expressed as [[6-8]](#Conlon2021)
```math
\begin{equation}
\mathrm{Tr}(W\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\}))\geq \min_{\textbf{X},\mathcal{Q}} \mathrm{Tr}\left((W\otimes\rho)\mathcal{Q}\right)
\end{equation}
```

with $W$ the weight matrix and $\mathcal{Q}$ a matrix satisfying $\mathcal{Q}\geq\textbf{X}^{\mathrm{T}}\textbf{X}$.
The NHB can be solved by the semidefinite programming as

```math
\begin{align*}
& \min_{\textbf{X},\mathcal{Q}}~\mathrm{Tr}\left((W\otimes\rho)\mathcal{Q}\right), \\
& \mathrm{subject}~\mathrm{to}~
\begin{cases}
\left(\begin{array}{cc}
\mathcal{Q} & \textbf{X}^{\mathrm{T}} \\
\textbf{X} & I\\
\end{array}\right)\geq 0, \\
\mathrm{Tr}(\rho X_a)=0,\,\forall a, \\
\mathrm{Tr}(X_a\partial_b\rho)=\delta_{ab},\,\forall a, b.\\
\end{cases}
\end{align*}
```

In QuanEstimation, the NHB can be solved by
``` jl
NHB(rho, drho, W)
```
Here `W` is again a required `AbstractMatrix` in the matrix-level method; the scheme-level
method `NHB(scheme; W=nothing)` supplies the identity weight matrix by default.

---

**Example 3.7**

```@raw html
<a id="example3_7"></a>
```

The Hamiltonian of a two-qubit system with $XX$ coupling is
```math
\begin{align}
H=\omega_1\sigma_3^{(1)}+\omega_2\sigma_3^{(2)}+g\sigma_1^{(1)}\sigma_1^{(2)},
\end{align}
```

where $\omega_1$, $\omega_2$ are the frequencies of the first and second qubit, $\sigma_i^{(1)}
=\sigma_i\otimes I$ and $\sigma_i^{(2)}=I\otimes\sigma_i$ for $i=1,2,3$. $\sigma_1$,
$\sigma_2$, $\sigma_3$ are Pauli matrices and $I$ denotes the identity matrix. The dynamics
is described by the master equation
```math
\begin{align}
\partial_t\rho=-i[H, \rho]+\sum_{i=1,2}\gamma_i\left(\sigma_3^{(i)}\rho\sigma_3^{(i)}-\rho\right)
\end{align}
```

with $\gamma_i$ the decay rate for the $i$th qubit.

The probe state is taken as $\frac{1}{\sqrt{2}}(|00\rangle+|11\rangle)$ and the weight matrix
is set to be identity. The measurement for $\mathrm{Tr}(W\mathcal{I^{-1}})$ is $\{\Pi_1$,
$\Pi_2$, $I-\Pi_1-\Pi_2\}$ with $\Pi_1=0.85|00\rangle\langle 00|$ and $\Pi_2=0.4|\!+
\!+\rangle\langle+\!+\!|$. Here $|\pm\rangle:=\frac{1}{\sqrt{2}}(|0\rangle\pm|1\rangle)$ with
$|0\rangle$ $(|1\rangle)$ the eigenstate of $\sigma_3$ with respect to the eigenvalue $1$ ($-1$).
``` jl
using QuanEstimation
using LinearAlgebra

# initial state
psi0 = [1., 0., 0., 1.]/sqrt(2)
rho0 = psi0*psi0'
# free Hamiltonian
omega1, omega2, g = 1.0, 1.0, 0.1
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
H0 = omega1*kron(sz, I(2)) + omega2*kron(I(2), sz) + g*kron(sx, sx)
# derivatives of the free Hamiltonian with respect to omega2 and g
dH = [kron(I(2), sz), kron(sx, sx)]
# dissipation
decay = [[kron(sz, I(2)), 0.05], [kron(I(2), sz), 0.05]]
# measurement
m1 = [1., 0., 0., 0.]
M1 = 0.85*m1*m1'
M2 = 0.1*ones(4, 4)
M = [M1, M2, I(4)-M1-M2]
# time length for the evolution
tspan = range(0., 5., length=200)
# dynamics (the decay list is a keyword argument)
rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH; decay=decay)
# weight matrix
W = one(zeros(2, 2))
# calculation of the HCRB and NHB
f_HCRB, f_NHB = [], []
for ti in 2:length(tspan)
    # HCRB
    f_tp1 = QuanEstimation.HCRB(rho[ti], drho[ti], W)
    append!(f_HCRB, f_tp1)
    # NHB
    f_tp2 = QuanEstimation.NHB(rho[ti], drho[ti], W)
    append!(f_NHB, f_tp2)
end
```
---

## **Bayesian Cramér-Rao bounds**
The Bayesian version of the classical Fisher information (matrix) and quantum Fisher information
(matrix) can be calculated by

```math
\mathcal{I}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{I}\mathrm{d}\textbf{x}
```

and

```math
\mathcal{F}_{\mathrm{Bayes}}=\int p(\textbf{x})\mathcal{F}\mathrm{d}\textbf{x},
```

where $p(\textbf{x})$ is the prior distribution, $\mathcal{I}$ and $\mathcal{F}$ are CFI (CFIM)
and QFI (QFIM) of all types, respectively.

In QuanEstimation, BCFI (BCFIM) and BQFI (BQFIM) can be solved via
``` jl
BCFIM(x, p, rho, drho; M=nothing, tol=1e-10, input_error=0.0,
      guard=nothing, refine=true, diagnostics=false)
```
``` jl
BQFIM(x, p, rho, drho; LDtype=:SLD, tol=1e-10, input_error=0.0,
      guard=nothing, refine=true, diagnostics=false)
```
where `x` represents the regimes of the parameters for the integral, it should be input as a
vector of arrays. `p` is an array representing the prior distribution. The input variables `rho`
and `drho` are two multidimensional arrays with the dimensions as `x`. For example, for three
parameter ($x_0, x_1, x_2$) estimation, the $ijk$th entry of `rho` and `drho` are $\rho$ and
$[\partial_0\rho, \partial_1\rho, \partial_2\rho]$ with respect to the values $[x_0]_i$,
$[x_1]_j$ and $[x_2]_k$, respectively. `LDtype` represents the types of QFI (QFIM) can be set,
options are `LDtype=:SLD` (default), `LDtype=:RLD` and `LDtype=:LLD`. `refine` is accepted by
`BCFIM` for API uniformity but has no effect (the CFIM solver does not refine); in `BQFIM` it
controls refinement of the inner SLD solves. `M` (available in `BCFIM`) represents a set of
positive operator-valued measure (POVM) with default value `nothing`. In QuanEstimation, a set of
rank-one symmetric informationally complete POVM (SIC-POVM) is loaded when `M=nothing`. SIC-POVM is
calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded
from [here](http://www.physics.umb.edu/Research/QBism/solutions.html). `BCFIM`, `BQFIM`, `OBB`,
`BCB`, `BayesCost` and `BayesInput` are exported by the package and may be called without the
`QuanEstimation.` prefix.

In the Bayesian scenarios, the covariance matrix with a prior distribution $p(\textbf{x})$ is
defined as
```math
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})=\int p(\textbf{x})\sum_y\mathrm{Tr}(\rho\Pi_y)
(\hat{\textbf{x}}-\textbf{x})(\hat{\textbf{x}}-\textbf{x})^{\mathrm{T}}\mathrm{d}\textbf{x},
\end{align}
```

where $\textbf{x}=(x_0,x_1,\dots)^{\mathrm{T}}$ are the unknown parameters to be estimated and
the integral $\int\mathrm{d}\textbf{x}:=\int\mathrm{d}x_0\int\mathrm{d}x_1\cdots$. $\{\Pi_y\}$ is
a set of POVM and $\rho$ represents the parameterized density matrix. Two types of Bayesian
Cramér-Rao bound (BCRB) are calculated in this package, the first one is
```math
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})\left(B\mathcal{I}^{-1}B
+\textbf{b}\textbf{b}^{\mathrm{T}}\right)\mathrm{d}\textbf{x},
\end{align}
```

where $\textbf{b}$ and $\textbf{b}'$ are the vectors of bias and its derivatives with respect to
$\textbf{x}$. $B$ is a diagonal matrix with the $i$th entry $B_{ii}=1+[\textbf{b}']_{i}$ and
$\mathcal{I}$ is the CFIM. The second one is
```math
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \mathcal{B}\,\mathcal{I}_{\mathrm{Bayes}}^{-1}\,
\mathcal{B}+\int p(\textbf{x})\textbf{b}\textbf{b}^{\mathrm{T}}\mathrm{d}\textbf{x},
\end{align}
```

where $\mathcal{B}=\int p(\textbf{x})B\mathrm{d}\textbf{x}$ is the average of $B$ and
$\mathcal{I}_{\mathrm{Bayes}}$ is the average of the CFIM.

The third one is
```math
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})
\mathcal{G}\left(\mathcal{I}_p+\mathcal{I}\right)^{-1}\mathcal{G}^{\mathrm{T}}\mathrm{d}\textbf{x}
\end{align}
```

with $[\mathcal{I}_{p}]_{ab}:=[\partial_a \ln p(\textbf{x})][\partial_b \ln p(\textbf{x})]$ and
$\mathcal{G}_{ab}:=[\partial_b\ln p(\textbf{x})][\textbf{b}]_a+B_{aa}\delta_{ab}$.

Three types of Bayesian Quantum Cramér-Rao bound (BQCRB) are calculated, the first one is
```math
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq\int p(\textbf{x})\left(B\mathcal{F}^{-1}B
+\textbf{b}\textbf{b}^{\mathrm{T}}\right)\mathrm{d}\textbf{x}
\end{align}
```

with $\mathcal{F}$ the QFIM for all types. The second one is
```math
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \mathcal{B}\,\mathcal{F}_{\mathrm{Bayes}}^{-1}\,
\mathcal{B}+\int p(\textbf{x})\textbf{b}\textbf{b}^{\mathrm{T}}\mathrm{d}\textbf{x}
\end{align}
```

with $\mathcal{F}_{\mathrm{Bayes}}$ the average of the QFIM.

The third one is
```math
\begin{align}
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \int p(\textbf{x})
\mathcal{G}\left(\mathcal{I}_p+\mathcal{F}\right)^{-1}\mathcal{G}^{\mathrm{T}}\mathrm{d}\textbf{x}.
\end{align}
```

In QuanEstimation, the BCRB and BQCRB are calculated via
``` jl
BCRB(x, p, dp, rho, drho; M=nothing, b=nothing, db=nothing,
     btype=1, tol=1e-10, input_error=0.0, guard=nothing, refine=true,
     diagnostics=false)
```
``` jl
BQCRB(x, p, dp, rho, drho; b=nothing, db=nothing, btype=1,
      LDtype=:SLD, tol=1e-10, input_error=0.0, guard=nothing,
      refine=true, diagnostics=false)
```
where `b` and `db` are the vectors of biases and its derivatives on the unknown parameters.
For unbiased estimates, `b=nothing` and `db=nothing`. In QuanEstimation, the users can set
the types of BCRB and BQCRB via the variable `btype`.

For single parameter estimation, Ref [[9]](#Liu2016) calculates the optimal biased bound based
on the first type of the BQCRB, it can be realized numerically via
``` jl
OBB(x, p, dp, rho, drho, d2rho; LDtype=:SLD, tol=1e-10,
    input_error=0.0, guard=nothing, refine=true, diagnostics=false)
```
`d2rho` is a vector representing the second order derivatives of `rho` on `x`; for a single
parameter it has the same one-entry-per-grid-point form as `drho`, with each entry holding the
second derivative matrix.

Van Trees in 1968 [[10]](#vanTrees1968) provides a well used Bayesian version of Cramér-Rao
bound known as Van Trees bound (VTB).
```math
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \left(\mathcal{I}_{\mathrm{prior}}
+\mathcal{I}_{\mathrm{Bayes}}\right)^{-1},
```

where $\mathcal{I}_{\mathrm{prior}}=\int p(\textbf{x})\mathcal{I}_{p}\mathrm{d}\textbf{x}$
is the CFIM for $p(\textbf{x})$ and $\mathcal{I}_{\mathrm{Bayes}}$ is the average of the CFIM.

The quantum version (QVTB) provided by Tsang, Wiseman
and Caves [[12]](#Tsang2011).
```math
\mathrm{cov}(\hat{\textbf{x}},\{\Pi_y\})\geq \left(\mathcal{I}_{\mathrm{prior}}
+\mathcal{F}_{\mathrm{Bayes}}\right)^{-1},
```

with $\mathcal{F}_{\mathrm{Bayes}}$ the average of the QFIM of all types.

The functions to calculate the VTB and QVTB are
``` jl
VTB(x, p, dp, rho, drho; M=nothing, tol=1e-10, input_error=0.0,
    guard=nothing, refine=true, diagnostics=false)
```
``` jl
QVTB(x, p, dp, rho, drho; LDtype=:SLD, tol=1e-10, input_error=0.0,
     guard=nothing, refine=true, diagnostics=false)
```
Here the variables in the codes are the same with `BCRB` and `BQCRB`.

## **Quantum Ziv-Zakai bound**
The expression of Quantum Ziv-Zakai bound (QZZB) with a prior distribution $p(x)$ in a finite
regime $[\alpha,\beta]$ is

```math
\begin{align*}
\mathrm{var}(\hat{x},\{\Pi_y\}) &\geq & \frac{1}{2}\int_0^\infty \mathrm{d}\tau\tau
\mathcal{V}\int_{-\infty}^{\infty} \mathrm{d}x\min\!\left\{p(x), p(x+\tau)\right\} \\
& & \times\left(1-\frac{1}{2}||\rho(x)-\rho(x+\tau)||\right),
\end{align*}
```

where $||\cdot||$ represents the trace norm and $\mathcal{V}$ is the "valley-filling"
operator satisfying $\mathcal{V}f(\tau)=\max_{h\geq 0}f(\tau+h)$. $\rho(x)$ is the
parameterized density matrix.

In QuanEstimation, the QZZB can be calculated via the function:
``` jl
QZZB(x, p, rho; eps=nothing, ν=1)
```
where `x` is a vector of arrays representing the regime of the parameter for the integral, `p` is
an array representing the prior distribution and `rho` is a multidimensional array representing
the density matrix. `ν` is an optional exponent forwarded to the Helstrom bound (default
`ν=1`); it only affects the pure-state Helstrom bound and has no effect for density-matrix
inputs. `eps` is a legacy keyword accepted for backward compatibility; it is not used by the
QZZB computation.

---
**Example 3.8**

```@raw html
<a id="example3_8"></a>
```

The Hamiltonian of a qubit system under a magnetic field $B$ in the XZ plane is
```math
\begin{align}
H=\frac{B\omega_0}{2}(\sigma_1\cos{x}+\sigma_3\sin{x})
\end{align}
```

with $x$ the unknown parameter and $\sigma_{1}$, $\sigma_{3}$ Pauli matrices. The probe state
is taken as $\frac{1}{\sqrt{2}}(|0\rangle+|1\rangle)$ with $|0\rangle$ ($|1\rangle$) the
eigenstates of $\sigma_3$ with respect to the eigenvalues $1$ ($-1$). The measurement
for classical bounds is a set of rank-one symmetric informationally complete positive
operator-valued measure (SIC-POVM).

Take the Gaussian prior distribution $p(x)=\frac{1}{c\eta\sqrt{2\pi}}\exp\left({-\frac{(x-\mu)^2}
{2\eta^2}}\right)$ on $[-\pi/2, \pi/2]$ with $\mu$ and $\eta$ the expectation and standard
deviation, respectively. Here $c=\frac{1}{2}\big[\mathrm{erf}(\frac{\pi-2\mu}{2\sqrt{2}\eta})
+\mathrm{erf}(\frac{\pi+2\mu}{2\sqrt{2}\eta})\big]$ is the normalized coefficient with
$\mathrm{erf}(x):=\frac{2}{\sqrt{\pi}}\int^x_0 e^{-t^2}\mathrm{d}t$ the error function.
``` jl
using QuanEstimation
using Trapz

# free Hamiltonian
function H0_func(x)
    return 0.5*B*omega0*(sx*cos(x)+sz*sin(x))
end
# derivative of the free Hamiltonian on x
function dH_func(x)
    return [0.5*B*omega0*(-sx*sin(x)+sz*cos(x))]
end
# prior distribution
function p_func(x, mu, eta)
    return exp(-(x-mu)^2/(2*eta^2))/(eta*sqrt(2*pi))
end
function dp_func(x, mu, eta)
    return -(x-mu)*exp(-(x-mu)^2/(2*eta^2))/(eta^3*sqrt(2*pi))
end

B, omega0 = 0.5*pi, 1.0
sx = [0. 1.; 1. 0.0im]
sy = [0. -im; im 0.]
sz = [1. 0.0im; 0. -1.]
# initial state
rho0 = 0.5*ones(2, 2)
# prior distribution
x = range(-0.5*pi, stop=0.5*pi, length=100) |>Vector
mu, eta = 0.0, 0.2
p_tp = [p_func(x[i], mu, eta) for i in 1:length(x)]
dp_tp = [dp_func(x[i], mu, eta) for i in 1:length(x)]
# normalization of the distribution
c = trapz(x, p_tp)
p = p_tp/c
dp = dp_tp/c
# time length for the evolution
tspan = range(0., stop=1., length=1000)
# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
drho = Vector{Vector{Matrix{ComplexF64}}}(undef, length(x))
for i = 1:length(x)
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = QuanEstimation.expm(tspan, rho0, H0_tp, dH_tp)
    rho[i], drho[i] = rho_tp[end], drho_tp[end]
end
```
``` jl
# Classical Bayesian bounds
f_BCRB1 = QuanEstimation.BCRB([x], p, dp, rho, drho; btype=1)
f_BCRB2 = QuanEstimation.BCRB([x], p, dp, rho, drho; btype=2)
f_BCRB3 = QuanEstimation.BCRB([x], p, dp, rho, drho; btype=3)
f_VTB = QuanEstimation.VTB([x], p, dp, rho, drho)
```
``` jl
# Quantum Bayesian bounds
f_BQCRB1 = QuanEstimation.BQCRB([x], p, dp, rho, drho; btype=1)
f_BQCRB2 = QuanEstimation.BQCRB([x], p, dp, rho, drho; btype=2)
f_BQCRB3 = QuanEstimation.BQCRB([x], p, dp, rho, drho; btype=3)
f_QVTB = QuanEstimation.QVTB([x], p, dp, rho, drho)
f_QZZB = QuanEstimation.QZZB([x], p, rho)
```
---
## **Bayesian estimation**
In QuanEstimation, two estimation routines are considered: maximum a posteriori
estimation (MAP), which is Bayesian, and maximum likelihood estimation (MLE). In Bayesian
estimation, the prior distribution
is updated as
```math
\begin{align}
p(\textbf{x}|y)=\frac{p(y|\textbf{x})p(\textbf{x})}{\int p(y|\textbf{x})p(\textbf{x})
\mathrm{d}\textbf{x}}
\end{align}
```

with $p(\textbf{x})$ the current prior distribution and $y$ the outcome of the experiment.
In practice, the prior distribution is replaced with $p(\textbf{x}|y)$ and the estimated
value of $\textbf{x}$ can be evaluated by
``` jl
Bayes(x, p, rho, y; M=nothing, estimator="mean", savefile=false)
```
``` jl
MLE(x, rho, y; M=nothing, savefile=false)
```
where `x` is a vector of arrays representing the regimes of the parameters for the integral and
`p` is an array representing the prior distribution. For multiparameter estimation, `p` is
multidimensional. The input variable `rho` is a multidimensional array with the dimensions as `x`
representing the parameterized density matrix. `M` contains a set of positive operator-valued
measure (POVM). In QuanEstimation, a set of rank-one symmetric informationally complete POVM
(SIC-POVM) is used when `M=nothing`. SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM
fiducial state which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).
`y` contains the observed outcome indices, starting from `0`, of the POVM elements in `M`.
`estimator` in `Bayes()` represents the estimator which defaults to the mean value of the
parameters. Also, it can be set as `"MAP"`.

Both routines write a JLD2 file rather than CSV files. `Bayes()` writes `bayes.dat` with keys
`"p"` (the posterior distribution(s)) and `"x"` (the estimated values from all iterations);
`MLE()` writes `MLE.dat` with keys `"L"` (the log-likelihood) and `"x"` (the estimated values
from all iterations). With `savefile=false` (the default), `"p"` (or `"L"`) stores only the
final posterior (or final cumulative log-likelihood), while `"x"` still stores the estimate
from every iteration. With `savefile=true`, every iteration's posterior (or log-likelihood)
and estimate is stored. Both functions return a two-element tuple: the first element is the
final posterior (`Bayes`) or the final cumulative log-likelihood array (`MLE`), and the
second element is the final estimate, i.e. the last entry of the full history recorded in the
file. The second return value is therefore a single estimate, not the full history. See
[Output files](output_files.md) for the exact JLD2 layout of `bayes.dat` and `MLE.dat`.

**Example 3.9**

```@raw html
<a id="example3_9"></a>
```

The Hamiltonian of a qubit system is
```math
\begin{align}
H=\frac{B\omega_0}{2}(\sigma_1\cos{x}+\sigma_3\sin{x}),
\end{align}
```

where $B$ is the magnetic field in the XZ plane, $x$ is the unknown parameter and $\sigma_{1}$,
$\sigma_{3}$ are the Pauli matrices. The probe state is taken as $|\pm\rangle$. The measurement
is $\{|\!+\rangle\langle+\!|,|\!-\rangle\langle-\!|\}$. Here $|\pm\rangle:=\frac{1}{\sqrt{2}}
(|0\rangle\pm|1\rangle)$ with $|0\rangle$ $(|1\rangle)$ the eigenstate of $\sigma_3$ with respect
to the eigenvalue $1$ $(-1)$. In this example, the prior distribution $p(x)$ is uniform on
$[0, \pi/2]$.
``` jl
using QuanEstimation
using Random
using StatsBase

# free Hamiltonian
function H0_func(x)
    return 0.5*B*omega0*(sx*cos(x)+sz*sin(x))
end
# derivative of the free Hamiltonian on x
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
# prior distribution
x = range(0., stop=0.5*pi, length=100) |>Vector
p = (1.0/(x[end]-x[1]))*ones(length(x))
# time length for the evolution
tspan = range(0., stop=1., length=1000)
# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
for i = 1:length(x)
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = QuanEstimation.expm(tspan, rho0, H0_tp, dH_tp)
    rho[i] = rho_tp[end]
end
```
``` jl
# Generation of the experimental results
Random.seed!(1234)
y = [0 for i in 1:500]
res_rand = sample(1:length(y), 125, replace=false)
for i in 1:length(res_rand)
    y[res_rand[i]] = 1
end
```
``` jl
# Maximum a posteriori estimation
pout, xest = QuanEstimation.Bayes([x], p, rho, y; M=M, estimator="MAP",
                                  savefile=false)
```
``` jl
# Maximum likelihood estimation
Lout, xest = QuanEstimation.MLE([x], rho, y; M=M, savefile=false)
```

The average Bayesian cost [[14]](#Robert2007) for a quadratic cost function can be
calculated via
```math
\begin{equation}
\bar{C}:=\int p(\textbf{x})\sum_y p(y|\textbf{x})(\textbf{x}-\hat{\textbf{x}})^{\mathrm{T}}
W(\textbf{x}-\hat{\textbf{x}})\,\mathrm{d}\textbf{x}
\end{equation}
```

In QuanEstimation, this can be realized by calling
``` jl
BayesCost(x, p, xest, rho, M; W=nothing, eps=nothing)
```
`xest` represents the estimators for the parameters: `xest[j]` is the estimate associated
with the `j`th POVM element in `M`, and for multiparameter estimation each entry is a vector
with one component per parameter. `W` is the weight matrix; it defaults to the identity for
multiple parameters and is ignored in the single-parameter case. `eps` is a legacy keyword.

Besides, the average Bayesian cost bounded by [[5]](#Rafal2020)
```math
\begin{equation}
\bar{C}\geq\int p(\textbf{x})\left(\textbf{x}^{\mathrm{T}}W\textbf{x}\right)\mathrm{d}\textbf{x}
-\sum_{ab}W_{ab}\mathrm{Tr}\left(\bar{\rho}\bar{L}_a \bar{L}_b\right),
\label{eq:BCB}
\end{equation}
```

and for single-parameter scenario, this inequality reduces to
```math
\begin{equation}
\bar{C}\geq \int p(x) x^2\,\mathrm{d}x-\mathrm{Tr}(\bar{\rho}\bar{L}^2).
\end{equation}
```

The function for calculating the Bayesian cost bound (BCB) is
``` jl
BCB(x, p, rho; W=nothing, tol=1e-10, input_error=0.0, guard=nothing,
    refine=true, diagnostics=false)
```
`W` defaults to the identity matrix for multiple parameters and is ignored in the
single-parameter case; `eps` is a legacy keyword accepted in addition to the keywords shown.
**Example 3.10**

```@raw html
<a id="example3_10"></a>
```

The Hamiltonian of a qubit system is
```math
\begin{align}
H=\frac{B\omega_0}{2}(\sigma_1\cos{x}+\sigma_3\sin{x}),
\end{align}
```

where $B$ is the magnetic field in the XZ plane, $x$ is the unknown parameter and $\sigma_{1}$,
$\sigma_{3}$ are the Pauli matrices. The probe state is taken as $|\pm\rangle$. The measurement
is $\{|\!+\rangle\langle+\!|,|\!-\rangle\langle-\!|\}$. Here $|\pm\rangle:=\frac{1}{\sqrt{2}}
(|0\rangle\pm|1\rangle)$ with $|0\rangle$ $(|1\rangle)$ the eigenstate of $\sigma_3$ with respect
to the eigenvalue $1$ $(-1)$. In this example, the prior distribution $p(x)$ is uniform on
$[0, \pi/2]$.
``` jl
using QuanEstimation
using Random
using StatsBase

# free Hamiltonian
function H0_func(x)
    return 0.5*B*omega0*(sx*cos(x)+sz*sin(x))
end
# derivative of the free Hamiltonian on x
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
# prior distribution
x = range(0., stop=0.5*pi, length=100) |>Vector
p = (1.0/(x[end]-x[1]))*ones(length(x))
# time length for the evolution
tspan = range(0., stop=1., length=1000)
# dynamics
rho = Vector{Matrix{ComplexF64}}(undef, length(x))
for i = 1:length(x)
    H0_tp = H0_func(x[i])
    dH_tp = dH_func(x[i])
    rho_tp, drho_tp = QuanEstimation.expm(tspan, rho0, H0_tp, dH_tp)
    rho[i] = rho_tp[end]
end
```
``` jl
# average Bayesian cost
M = QuanEstimation.SIC(2)
xest = [[0.8], [0.9], [1.0], [1.2]]
C = QuanEstimation.BayesCost([x], p, xest, rho, M)
```
``` jl
# Bayesian cost Bound
C = QuanEstimation.BCB([x], p, rho)
```
---
## **Bibliography**

```@raw html
<a id="Helstrom1976"></a>
```

**[1]**
C. W. Helstrom,
*Quantum Detection and Estimation Theory*
(New York: Academic, 1976).

**[2]**
A. S. Holevo,
*Probabilistic and Statistical Aspects of Quantum Theory*
(Amsterdam: North-Holland, 1982).

**[3]**
J. Liu, H. Yuan, X.-M. Lu, and X. Wang,
Quantum Fisher information matrix and multiparameter estimation,
[J. Phys. A: Math. Theor. **53**, 023001 (2020).](https://doi.org/10.1088/1751-8121/ab5d4d)

```@raw html
<a id="Holevo1973"></a>
```

**[4]**
A. S Holevo,
Statistical decision theory for quantum systems,
[J. Multivariate Anal. **3**, 337-394 (1973).](https://doi.org/10.1016/0047-259X(73)90028-6)

```@raw html
<a id="Rafal2020"></a>
```

**[5]**
R. Demkowicz-Dobrzański, W. Górecki, and M. Guţă,
Multi-parameter estimation beyond Quantum Fisher Information,
[J. Phys. A: Math. Theor. **53**, 363001 (2020).](https://doi.org/10.1088/1751-8121/ab8ef3)

```@raw html
<a id="Conlon2021"></a>
```

**[6]**
H. Nagaoka,
A New Approach to Cra ´ mer–Rao Bounds for Quantum State Estimation,
Tech. Rep. IT89-42, 9 (1989).

**[7]**
M. Hayashi,
On simultaneous measurement of noncommutative observables. In Development of
Infinite-Dimensional Non-Commutative Analysis,
96–188 (Kyoto Univ., 1999).

**[8]**
L. O. Conlon, J. Suzuki, P. K. Lam, and S. M. Assad,
Efficient computation of the Nagaoka–Hayashi bound for multiparameter estimation with separable measurements,
[npj Quantum Inf.  **7**, 110 (2021).](https://doi.org/10.1038/s41534-021-00414-1)

```@raw html
<a id="Liu2016"></a>
```

**[9]**
J. Liu and H. Yuan,
Valid lower bound for all estimators in quantum parameter estimation,
[New J. Phys. **18**, 093009 (2016).](https://doi.org/10.1088/1367-2630/18/9/093009)

```@raw html
<a id="vanTrees1968"></a>
```

**[10]**
H. L. Van Trees,
*Detection, estimation, and modulation theory: Part I*
(Wiley, New York, 1968).

**[11]**
W. Zhong, Z. Sun, J. Ma, X. Wang, and F. Nori,
Fisher information under decoherence in Bloch representation,
[Phys. Rev. A **87**, 022337 (2013).](https://doi.org/10.1103/PhysRevA.87.022337)

```@raw html
<a id="Tsang2011"></a>
```

**[12]**
M. Tsang, H. M. Wiseman, and C. M. Caves,
Fundamental quantum limit to waveform estimation,
[Phys. Rev. Lett. **106**, 090401 (2011).](https://doi.org/10.1103/PhysRevLett.106.090401)

```@raw html
<a id="Safranek2019"></a>
```

**[13]**
D. Šafránek,
Estimation of Gaussian quantum states,
[J. Phys. A: Math. Theor. **52**, 035304 (2019).](https://doi.org/10.1088/1751-8121/aaf068)

```@raw html
<a id="Robert2007"></a>
```

**[14]**
C. P. Robert,
*The Bayesian Choice* (Berlin: Springer, 2007).

**See also**: [Control optimization](guide_Copt.md), [State optimization](guide_Sopt.md), [Measurement optimization](guide_Mopt.md), [Comprehensive optimization](guide_Compopt.md) — use the bounds as objective functions to search for optimal quantum metrological schemes.
