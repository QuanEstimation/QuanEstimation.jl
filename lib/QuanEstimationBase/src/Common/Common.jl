include("BayesEstimation.jl")

@doc raw"""

    destroy(N::Int)

Construct the ``N``-dimensional bosonic annihilation operator.

```math
a = \sum_{n=1}^{N-1} \sqrt{n} \, |n-1\rangle\langle n|
```
raw"""
destroy(N) = diagm(1 => [sqrt(n) + 0.0im for n = 1:N-1])

raw"""

    bases(dim; T=ComplexF64)

Return the standard basis vectors ``\{|e_i\rangle\}_{i=1}^{\mathrm{dim}}`` as a vector of vectors.
"""
bases(dim; T = ComplexF64) = [e for e in I(dim) .|> T |> eachrow]

raw"""

    σx = SigmaX

Pauli-X matrix ``\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}``.
raw"""
σx = SigmaX = () -> complex([0.0 1; 1 0])

raw"""

    σy = SigmaY

Pauli-Y matrix ``\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}``.
"""
σy = SigmaY = () -> complex([0.0 -im; im 0])

raw"""

    σz = SigmaZ

Pauli-Z matrix ``\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}``.
raw"""
σz = SigmaZ = () -> complex([1.0 0; 0 -1])

raw"""

    vec2mat(x)

Convert a vector to a square matrix via column-major reshape. For a vector of length ``n``, returns an ``\sqrt{n} \times \sqrt{n}`` matrix. When given an array of vectors, applies element-wise.
"""
function vec2mat(x::Vector{T}) where {T<:Number}
    reshape(x, x |> length |> sqrt |> Int, :)
end

function vec2mat(x)
    vec2mat.(x)
end

"""
    repeat_copy(scheme, N)

Deep-copy `scheme` `N` times, used for parallel sampling in population-based optimizers (DE, PSO).
"""
function repeat_copy(scheme, N)
    [deepcopy(scheme) for _ = 1:N]
end

"""
    filterZeros(x::AbstractVecOrMat)

Replace elements of `x` that are within floating-point tolerance of zero with exact zero,
removing numerical noise.
raw"""
function filterZeros(x::AbstractVecOrMat{T}) where {T<:Number}
    [x + 1 ≈ 1 ? zero(T) : x for x in x]
end

raw"""
    basis(dim, si, ::T) where {T<:Complex}

Return the standard basis matrix ``|e_{si}\rangle\langle e_{si}|`` as a complex-typed matrix of dimension `dim × dim`.
raw"""
function basis(dim, si, ::T)::Array{T} where {T<:Complex}
    result = zeros(T, dim)
    result[si] = 1.0
    result
end

raw"""

    suN_generatorU(n, k)

Return the SU(``n``) symmetric (real) generator for index ``k``: ``|i\rangle\langle j| + |j\rangle\langle i|``.
raw"""
function suN_generatorU(n, k)
    tmp1, tmp2 = ceil((1 + sqrt(1 + 8k)) / 2), ceil((-1 + sqrt(1 + 8k)) / 2)
    i = k - tmp2 * (tmp2 - 1) / 2 |> Int
    j = tmp1 |> Int
    return sparse([i, j], [j, i], [1, 1], n, n)
end

raw"""

    suN_generatorV(n, k)

Return the SU(``n``) antisymmetric (imaginary) generator for index ``k``: ``-i|i\rangle\langle j| + i|j\rangle\langle i|``.
raw"""
function suN_generatorV(n, k)
    tmp1, tmp2 = ceil((1 + sqrt(1 + 8k)) / 2), ceil((-1 + sqrt(1 + 8k)) / 2)
    i = k - tmp2 * (tmp2 - 1) / 2 |> Int
    j = tmp1 |> Int
    return sparse([i, j], [j, i], [-im, im], n, n)
end

raw"""

    suN_generatorW(n, k)

Return the SU(``n``) diagonal generator for index ``k``, normalized as ``\mathrm{diag}(1,\dots,1,-k,0,\dots,0)``.
raw"""
function suN_generatorW(n, k)
    diagw = spzeros(n)
    diagw[1:k] .= 1
    diagw[k+1] = -k
    return spdiagm(n, n, diagw)
end

@doc raw"""

    suN_generator(n::Int64)

Generation of the SU(``N``) generators with ``N`` the dimension of the system.
- `N`: The dimension of the system.
raw"""
function suN_generator(n::Int64)
    result = Vector{SparseMatrixCSC{ComplexF64,Int64}}(undef, n^2 - 1)
    idx = 2
    itr = 1

    for i = 1:n-1
        idx_t = idx
        while idx_t > 0
            result[itr] =
                iseven(idx_t) ? suN_generatorU(n, (i * (i - 1) + idx - idx_t + 2) / 2) :
                suN_generatorV(n, (i * (i - 1) + idx - idx_t + 1) / 2)
            itr += 1
            idx_t -= 1
        end
        result[itr] = sqrt(2 / (i + i * i)) * suN_generatorW(n, i)
        itr += 1
        idx += 2
    end
    return result
end

raw"""
    basis(dim, index)

Return the standard basis vector ``|e_{\mathrm{index}}\rangle`` as a `Vector{Float64}` of length `dim` with a `1.0` at the given index.
"""
function basis(dim, index)
    x = zeros(dim)
    x[index] = 1.0
    return x
end

"""

    sic_povm(fiducial)

Generate a set of rank-one SIC-POVM elements by applying the ``d^2`` Weyl-Heisenberg displacement operators to a fiducial state (Fuchs et al., [doi:10.3390/axioms6030021](https://doi.org/10.3390/axioms6030021)).
"""
function sic_povm(fiducial)
    d = length(fiducial)
    w = exp(2.0 * pi * 1.0im / d)
    Z = diagm([w^(i - 1) for i = 1:d])
    X = zeros(ComplexF64, d, d)
    for i = 1:d
        for j = 1:d
            if j != d
                X += basis(d, j + 1) * basis(d, j)'
            else
                X += basis(d, 1) * basis(d, j)'
            end
        end
    end
    X = X / d

    D = [[Matrix{ComplexF64}(undef, d, d) for i = 1:d] for j = 1:d]
    for a = 1:d
        for b = 1:d
            X_a = X^(b - 1)
            Z_b = Z^(a - 1)
            D[a][b] = (-exp(1.0im * pi / d))^((a - 1) * (b - 1)) * X_a * Z_b
        end
    end

    res = Vector{Matrix{ComplexF64}}()
    for m = 1:d
        for n = 1:d
            res_tp = D[m][n] * fiducial
            res_tp = res_tp / norm(res_tp)
            push!(res, res_tp * res_tp' / d)
        end
    end
    return res
end

"""

    SIC(dim::Int64)

Generation of a set of rank-one symmetric informationally complete positive operator-valued measure (SIC-POVM).
- `dim`: The dimension of the system.
Note: SIC-POVM is calculated by the Weyl-Heisenberg covariant SIC-POVM fiducial state which can be downloaded from [here](http://www.physics.umb.edu/Research/QBism/solutions.html).
raw"""
function SIC(dim::Int64)
    data = readdlm("$(pkgpath)/sic_fiducial_vectors/d$(dim).txt", '\t', Float64, '\n')
    fiducial = data[:, 1] + 1.0im * data[:, 2]
    M = sic_povm(fiducial)
end

raw"""

    PlusState

The ``|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}`` state.
raw"""
PlusState() = complex([1.0, 1.0] / sqrt(2))

raw"""

    MinusState

The ``|-\rangle = (|0\rangle - |1\rangle)/\sqrt{2}`` state.
raw"""
MinusState() = complex([1.0, -1.0] / sqrt(2))
raw"""

    BellState(n::Int=1)

Return the ``n``-th Bell state (``n = 1, 2, 3, 4``):
- ``n=1``: ``|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}``
- ``n=2``: ``|\Phi^-\rangle = (|00\rangle - |11\rangle)/\sqrt{2}``
- ``n=3``: ``|\Psi^+\rangle = (|01\rangle + |10\rangle)/\sqrt{2}``
- ``n=4``: ``|\Psi^-\rangle = (|01\rangle - |10\rangle)/\sqrt{2}``
raw"""
BellState() = complex([1.0, 0.0, 0.0, 1.0] / sqrt(2))

function BellState(n::Int)
    if n == 1
        return complex([1.0, 0.0, 0.0, 1.0] / sqrt(2))
    elseif n == 2
        return complex([1.0, 0.0, 0.0, -1.0] / sqrt(2))
    elseif n == 3
        return complex([0.0, 1.0, 1.0, 0.0] / sqrt(2))
    elseif n == 4
        return complex([0.0, 1.0, -1.0, 0.0] / sqrt(2))
    else
        throw(DomainError(n, "Supported values for n are 1 to 4."))
    end
end

raw"""
    BayesInput(x, func, dfunc; channel="dynamics")

Compute ``\rho(\boldsymbol{x})`` and ``\partial\rho(\boldsymbol{x})`` for all parameter points
in the Bayesian grid `x`.

# Arguments
- `x`: Vector of parameter grid axes (one per parameter).
- `func`: Function that returns the density matrix (or Kraus operators) at a parameter point.
- `dfunc`: Function that returns the derivatives at a parameter point.
- `channel`: `"dynamics"` for Hamiltonian-based evolution, `"Kraus"` for Kraus operators.
"""
function BayesInput(x, func, dfunc; channel = "dynamics")
    para_num = length(x)
    x_size = [x[i] for i = 1:para_num]
    x_list = Iterators.product(x...)
    if channel == "dynamics"
        H = [func(xi) for xi in x_list]
        dH = [dfunc(xi) for xi in x_list]
        return H, dH
    elseif channel == "Kraus"
        K = [func(xi) for xi in x_list]
        dK = [dfunc(xi) for xi in x_list]
        return K, dK
    else
        throw(ArgumentError("Supported values for channel are 'dynamics' and 'Kraus'"))
    end
end

"""

    bound!(ctrl, ctrl_bound)

Clip control coefficients to the range `[ctrl_bound[1], ctrl_bound[2]]`. Operates in-place. Works with both nested vectors and flat vectors.
raw"""
function bound!(ctrl::Vector{Vector{Float64}}, ctrl_bound)
    for ck in eachindex(ctrl)
        for tk in eachindex(ctrl[1])
            ctrl[ck][tk] = (
                x ->
                    x < ctrl_bound[1] ? ctrl_bound[1] :
                    x > ctrl_bound[2] ? ctrl_bound[2] : x
            )(
                ctrl[ck][tk],
            )
        end
    end
end

function bound!(ctrl::Vector{Float64}, ctrl_bound)
    for ck in eachindex(ctrl)
        ctrl[ck] =
            (x -> x < ctrl_bound[1] ? ctrl_bound[1] : x > ctrl_bound[2] ? ctrl_bound[2] : x)(
                ctrl[ck],
            )
    end
end

@doc raw"""
    Adam(gt, t, para, mt, vt, epsilon, beta1, beta2, eps)

Perform one step of the Adam optimization algorithm.

```math
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= m_t / (1-\beta_1^t) \\
\hat{v}_t &= v_t / (1-\beta_2^t) \\
\theta_t &= \theta_{t-1} + \epsilon \, \hat{m}_t / (\sqrt{\hat{v}_t} + \varepsilon)
\end{aligned}
```

- `gt`: Gradient at current step.
- `t`: Current iteration index.
- `para`: Current parameter value.
- `mt`, `vt`: First and second moment estimates from previous step.
- `epsilon`: Learning rate.
- `beta1`, `beta2`: Exponential decay rates for moment estimates.
- `eps`: Small constant for numerical stability.
"""
function Adam(gt, t, para, mt, vt, epsilon, beta1, beta2, eps)
    t = t + 1
    mt = beta1 * mt + (1 - beta1) * gt
    vt = beta2 * vt + (1 - beta2) * (gt * gt)
    m_cap = mt / (1 - (beta1^t))
    v_cap = vt / (1 - (beta2^t))
    para = para + (epsilon * m_cap) / (sqrt(v_cap) + eps)
    return para, mt, vt
end

#### bound coefficients of linear combination in Mopt ####
raw"""

    bound_LC_coeff!(coefficients::Vector{Vector{Float64}}, rng)

Clip linear-combination coefficients to ``[0, 1]`` and normalize columns to sum to 1. Ensures each row and column has at least one nonzero entry. Operates in-place.
raw"""
function bound_LC_coeff!(coefficients::Vector{Vector{Float64}}, rng)
    M_num = length(coefficients)
    basis_num = length(coefficients[1])
    for ck = 1:M_num
        for tk = 1:basis_num
            coefficients[ck][tk] =
                (x -> x < 0.0 ? 0.0 : x > 1.0 ? 1.0 : x)(coefficients[ck][tk])
        end
    end

    Sum_col = [sum([coefficients[m][n] for m = 1:M_num]) for n = 1:basis_num]
    for si = 1:basis_num
        if Sum_col[si] == 0.0
            int_num = sample(rng, 1:M_num, 1, replace = false)[1]
            coefficients[int_num][si] = 1.0
        end
    end

    Sum_row = [sum([coefficients[m][n] for n = 1:basis_num]) for m = 1:M_num]
    for mi = 1:M_num
        if Sum_row[mi] == 0.0
            int_num = sample(rng, 1:basis_num, 1, replace = false)[1]
            coefficients[mi][int_num] = rand(rng)
        end
    end

    Sum_col = [sum([coefficients[m][n] for m = 1:M_num]) for n = 1:basis_num]
    for i = 1:M_num
        for j = 1:basis_num
            coefficients[i][j] = coefficients[i][j] / Sum_col[j]
        end
    end
end

#### bound coefficients of rotation in Mopt ####
raw"""

    bound_rot_coeff!(coefficients::Vector{Float64})

Clip rotation coefficients to the range ``[0, 2\pi]``. Operates in-place.
"""
function bound_rot_coeff!(coefficients::Vector{Float64})
    n = length(coefficients)
    for tk = 1:n
        coefficients[tk] = (x -> x < 0.0 ? 0.0 : x > 2 * pi ? 2 * pi : x)(coefficients[tk])
    end
end

"""

    gramschmidt(A::Vector{Vector{ComplexF64}})

Perform Gram-Schmidt orthonormalization on a set of vectors.
raw"""
function gramschmidt(A::Vector{Vector{ComplexF64}})
    n = length(A[1])
    m = length(A)
    Q = [zeros(ComplexF64, n) for i = 1:m]
    for j = 1:m
        q = A[j]
        for i = 1:j-1
            rij = dot(Q[i], q)
            q = q - rij * Q[i]
        end
        Q[j] = q / norm(q)
    end
    return Q
end

@doc raw"""
    rotation_matrix(coefficients, Lambda)

Construct a unitary matrix via exponential parameterization:

```math
U = \prod_i \exp(i \, c_i \, \Lambda_i)
```

where ``c_i`` are the coefficients and ``\Lambda_i`` are the generator matrices.
"""
function rotation_matrix(coefficients, Lambda)
    dim = size(Lambda[1])[1]
    U = Matrix{ComplexF64}(I, dim, dim)
    for i in eachindex(Lambda)
        U = U * exp(1.0im * coefficients[i] * Matrix(Lambda[i]))
    end
    return U
end

#### initialization states for DE and PSO method ####
"""

    initial_state!(psi0, scheme, p_num, rng)

Initialize the state for each particle in the DE/PSO population. Uses provided initial states for the first entries and random pure states for the rest.
"""
function initial_state!(psi0, scheme, p_num, rng)
    dim = get_dim(scheme[1])
    if length(psi0) > p_num
        psi0 = [psi0[i] for i = 1:p_num]
    end
    for pj in eachindex(psi0)
        scheme[pj].StatePreparation.data = [psi0[pj][i] for i = 1:dim] |> x -> x * x'
    end
    for pj = (length(psi0)+1):p_num
        r_ini = 2 * rand(rng, dim) - ones(dim)
        r = r_ini / norm(r_ini)
        phi = 2 * pi * rand(rng, dim)
        scheme[pj].StatePreparation.data =
            [r[i] * exp(1.0im * phi[i]) for i = 1:dim] |> x -> x * x'
    end
end

#### initialization control coefficients for DE and PSO method ####
"""

    initial_ctrl!(opt, ctrl0, scheme, p_num, rng)

Initialize control coefficients for each particle in the DE/PSO population. Uses provided controls for the first entries and random values within `opt.ctrl_bound` for the rest.
"""
function initial_ctrl!(opt, ctrl0, scheme, p_num, rng)
    all_ctrl = [param_data(s).ctrl for s in scheme]
    ctrl_length = get_ctrl_length(scheme[1])
    ctrl_num = get_ctrl_num(scheme[1])

    if length(ctrl0) > p_num
        ctrl0 = [ctrl0[i] for i = 1:p_num]
    end
    for pj in eachindex(ctrl0)
        all_ctrl[pj] = deepcopy(ctrl0[pj])
    end
    if opt.ctrl_bound[1] == -Inf || opt.ctrl_bound[2] == Inf
        for pj = (length(ctrl0)+1):p_num
            all_ctrl[pj] = [[2 * rand(rng) - 1.0 for j = 1:ctrl_length] for i = 1:ctrl_num]
        end
    else
        a = opt.ctrl_bound[1]
        b = opt.ctrl_bound[2]
        for pj = (length(ctrl0)+1):p_num
            all_ctrl[pj] =
                [[(b - a) * rand(rng) + a for j = 1:ctrl_length] for i = 1:ctrl_num]
        end
    end

    for i = 1:p_num
        param_data(scheme[i]).ctrl = all_ctrl[i]

    end

end

#### initialization velocity for PSO ####
"""
    initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num, rng)

Initialize the velocity matrix for PSO control optimization. Returns a `ctrl_num × ctrl_length × p_num` array with random values scaled by 0.1, bounded by `opt.ctrl_bound`.
"""
function initial_velocity_ctrl(opt, ctrl_length, ctrl_num, p_num, rng)
    if opt.ctrl_bound[1] == -Inf || opt.ctrl_bound[2] == Inf
        velocity =
            0.1 * (
                2.0 * rand(rng, ctrl_num, ctrl_length, p_num) -
                ones(ctrl_num, ctrl_length, p_num)
            )
    else
        a = opt.ctrl_bound[1]
        b = opt.ctrl_bound[2]
        velocity =
            0.1 * (
                (b - a) * rand(rng, ctrl_num, ctrl_length, p_num) +
                a * ones(ctrl_num, ctrl_length, p_num)
            )
    end
    return velocity
end

#### initialization measurements for DE and PSO ####
"""

    initial_M!(measurement0, C_all, dim, p_num, M_num, rng)

Initialize measurement vectors for each particle in the DE/PSO population. Uses provided measurements for the first entries and random orthonormal vectors for the rest.
"""
function initial_M!(measurement0, C_all, dim, p_num, M_num, rng)
    if length(measurement0) > p_num
        measurement0 = [measurement0[i] for i = 1:p_num]
    end
    for pj in eachindex(measurement0)
        C_all[pj] = deepcopy(measurement0[pj])
    end
    for pj = (length(measurement0)+1):p_num
        M_tp = [Vector{ComplexF64}(undef, dim) for i = 1:M_num]
        for mi = 1:M_num
            r_ini = 2 * rand(rng, dim) - ones(dim)
            r = r_ini / norm(r_ini)
            phi = 2 * pi * rand(rng, dim)
            M_tp[mi] = [r[i] * exp(1.0im * phi[i]) for i = 1:dim]
        end
        C_all[pj] = [[M_tp[i][j] for j = 1:dim] for i = 1:M_num]
        # orthogonality and normalization 
        C_all[pj] = gramschmidt(C_all[pj])
    end
end

"""
    initial_LinearComb!(measurement0, B_all, basis_num, M_num, p_num, rng)

Initialize the linear-combination POVM coefficients for each particle in the DE/PSO population.

Uses provided `measurement0` matrices for the first entries and random values (subsequently normalized by [`bound_LC_coeff!`](@ref)) for the remaining particles.
"""
function initial_LinearComb!(measurement0, B_all, basis_num, M_num, p_num, rng)
    if length(measurement0) > p_num
        measurement0 = [measurement0[i] for i = 1:p_num]
    end
    for pj in eachindex(measurement0)
        B_all[pj] = deepcopy(measurement0[pj])
    end

    for pj = (length(measurement0)+1):p_num
        B_all[pj] = [rand(rng, basis_num) for i = 1:M_num]
        bound_LC_coeff!(B_all[pj], rng)
    end
end

"""
    initial_Rotation!(measurement0, s_all, dim, p_num, rng)

Initialize the rotation-parameterized POVM coefficients for each particle in the DE/PSO population.

Uses provided `measurement0` rotation parameters for the first entries and random values for the remaining particles.
"""
function initial_Rotation!(measurement0, s_all, dim, p_num, rng)
    if length(measurement0) > p_num
        measurement0 = [measurement0[i] for i = 1:p_num]
    end
    for pj in eachindex(measurement0)
        s_all[pj] = [measurement0[pj][i] for i = 1:dim*dim]
    end

    for pj = (length(measurement0)+1):p_num
        s_all[pj] = rand(rng, dim * dim)
    end
end
