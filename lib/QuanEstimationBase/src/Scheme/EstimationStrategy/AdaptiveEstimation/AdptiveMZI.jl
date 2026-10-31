"""

    Adapt_MZI

Mutable scheme for adaptive Mach-Zehnder interferometer phase estimation.

**Fields:**
- `x`: Parameter grid (phases).
- `p`: Prior distribution.
- `rho0`: Initial probe state.
"""
mutable struct Adapt_MZI <: AbstractScheme
    x::Union{Nothing, AbstractVector}
    p::Union{Nothing, AbstractVector}
    rho0::Union{Nothing, AbstractMatrix}
end

"""
    MIZtargetType

Abstract supertype for MZI adaptation target functions (sharpness or mutual information).
raw"""
abstract type MIZtargetType end
raw"""
    sharpness <: MIZtargetType

Uses sharpness (``|\langle e^{ix}\rangle|``) as the target for MZI feedback.
"""
abstract type sharpness <: MIZtargetType end
"""
    MI <: MIZtargetType

Uses mutual information as the target for MZI feedback.
"""
abstract type MI <: MIZtargetType end

const MZI_TARGET_MAP = Dict{Symbol, Type{<:MIZtargetType}}(
    :sharpness => sharpness,
    :MI        => MI,
)

# Loop-invariant MZI device data. The union sparsity template and source
# maps let the hot kernels refill a_u without rebuilding two Kronecker
# products or allocating a fresh sparse matrix for every grid point.
struct _MZIExec{F<:AbstractFloat}
    a::SparseMatrixCSC{Complex{F},Int}
    a_in::SparseMatrixCSC{Complex{F},Int}
    b_in::SparseMatrixCSC{Complex{F},Int}
    u_colptr::Vector{Int}
    u_rowval::Vector{Int}
    u_ia::Vector{Int}
    u_ib::Vector{Int}
end

function _mzi_embed_map(U::SparseMatrixCSC, S::SparseMatrixCSC)
    out = zeros(Int, nnz(U))
    for j in 1:size(U, 2)
        ps, pe = S.colptr[j], S.colptr[j + 1] - 1
        p = ps
        for iu in U.colptr[j]:(U.colptr[j + 1] - 1)
            r = U.rowval[iu]
            while p <= pe && S.rowval[p] < r
                p += 1
            end
            if p <= pe && S.rowval[p] == r
                out[iu] = p
            end
        end
    end
    return out
end

function _mzi_exec_bake(a::SparseMatrixCSC{Complex{F},Int}) where {F}
    N = size(a, 1) - 1
    Id = spdiagm(0 => ones(Complex{F}, N + 1))
    a_in = kron(a, Id)
    b_in = kron(Id, a)
    U = a_in + b_in
    return _MZIExec{F}(
        a,
        a_in,
        b_in,
        U.colptr,
        U.rowval,
        _mzi_embed_map(U, a_in),
        _mzi_embed_map(U, b_in),
    )
end
_mzi_exec_bake(a::AbstractMatrix) = _mzi_exec_bake(sparse(a))

function _mzi_au_buffer(ex::_MZIExec{F}) where {F}
    T = promote_type(Complex{F}, Float64)
    m = size(ex.a_in, 1)
    return SparseMatrixCSC{T,Int}(
        m,
        m,
        ex.u_colptr,
        ex.u_rowval,
        Vector{T}(undef, length(ex.u_ia)),
    )
end

function _mzi_fill_a_u!(Au::SparseMatrixCSC, ex::_MZIExec, x::Real, phi::Real, u::Integer)
    value = 0.5 * (x - phi) + 0.5 * pi * u
    s, c = sin(value), cos(value)
    zin = zero(eltype(ex.a_in))
    @inbounds for i in eachindex(ex.u_ia)
        ia, ib = ex.u_ia[i], ex.u_ib[i]
        aval = ia == 0 ? zin : ex.a_in.nzval[ia]
        bval = ib == 0 ? zin : ex.b_in.nzval[ib]
        Au.nzval[i] = aval * s + bval * c
    end
    return Au
end

# Preserve the original left-associative (rho0 * A') * A evaluation.
@inline function _mzi_pyx_likelihood(A, M, N, rho0)
    mul!(M, rho0, A')
    mul!(N, M, A)
    return real(tr(N))
end

function _mzi_online_scratch(ex::_MZIExec{F}) where {F}
    T = promote_type(Complex{F}, Float64)
    dim = size(ex.a_in, 1)
    Au = _mzi_au_buffer(ex)
    A0 = Matrix{T}(undef, dim, dim)
    A1 = Matrix{T}(undef, dim, dim)
    M = Matrix{T}(undef, dim, dim)
    Ns = Matrix{T}(undef, dim, dim)
    return Au, A0, A1, M, Ns
end

function _mzi_accumulate!(a_res, ex::_MZIExec, x, phi, u, Au, Tacc)
    for xi in eachindex(x)
        _mzi_fill_a_u!(Au, ex, x[xi], phi, u)
        mul!(Tacc, a_res[xi], Au)
        copyto!(a_res[xi], Tacc)
    end
    return a_res
end

"""
    calculate_online{P}

Dispatch struct for online MZI feedback calculation with target type parameter `P`.
"""
struct calculate_online{P} end
"""
    calculate_offline{P}

Dispatch struct for offline MZI feedback calculation with target type parameter `P`.
"""
struct calculate_offline{P} end

##========== online ==========##
@doc raw"""

    online(apt::Adapt_MZI; target::Symbol=:sharpness, output::String="phi")

Online adaptive phase estimation in the MZI.
- `apt`: Adaptive MZI struct which contains x, p, and rho0.
- `target`: Setting the target function for calculating the tunable phase. Options are: "sharpness" and "MI".
- `output`: Choose the output variables. Options are: "phi" and "dphi".
"""
function online(apt::Adapt_MZI; target::String = "sharpness", output::String = "phi", res=nothing)
    (; x, p, rho0) = apt
    adaptMZI_online(x, p, rho0, Symbol(output),  Symbol(target); res=res)
end

"""
    adaptMZI_online(x, p, rho0, output, target; res=nothing)

Core online MZI adaptive estimation loop. Iterates over photon-subtraction steps,
updates the feedback phase at each episode, and saves results.
"""
function adaptMZI_online(x, p, rho0, output, target; res=nothing)
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1) |> sparse
    ex = _mzi_exec_bake(a)
    Au = _mzi_au_buffer(ex)
    dim = (N + 1)^2
    Tacc = Matrix{ComplexF64}(undef, dim, dim)
    M = Matrix{ComplexF64}(undef, dim, dim)
    Ns = Matrix{ComplexF64}(undef, dim, dim)
    exp_ix = [exp(1.0im * xi) for xi in x]
    phi_span = range(-pi, stop = pi, length = length(x)) |> collect

    phi = 0.0
    a_res = [Matrix{ComplexF64}(I, dim, dim) for i in eachindex(x)]

    xout, y = [], []

    if output == :phi
        for ei = 1:N-1
            if isnothing(res)
                println("The tunable phase is $phi ($ei episodes)")
                print("Please enter the experimental result: ")
                enter = readline()
                u = parse(Int64, enter)
            else
                u = Int64(res[ei])
            end
            pyx = zeros(length(x)) |> sparse
            _mzi_accumulate!(a_res, ex, x, phi, u, Au, Tacc)
            for xi in eachindex(x)
                pyx[xi] =
                    _mzi_pyx_likelihood(a_res[xi], M, Ns, rho0) *
                    (factorial(N - ei) / factorial(N))
            end
            phi_update = calculate_online{MZI_TARGET_MAP[target]}(
                x,
                p,
                pyx,
                a_res,
                ex,
                rho0,
                N,
                ei,
                phi_span,
                exp_ix,
            )

            append!(xout, phi)
            append!(y, u)
            phi = phi_update
        end
        println("The estimator of the unknown phase is $phi ")
        append!(xout, phi)
        savefile_online(xout, y)
    else
        println("The initial tunable phase is $phi")
        for ei = 1:N-1
            if isnothing(res)
                println("The tunable phase is $phi ($ei episodes)")
                print("Please enter the experimental result: ")
                enter = readline()
                u = parse(Int64, enter)
            else
                u = Int64(res[ei])
            end

            pyx = zeros(length(x)) |> sparse
            _mzi_accumulate!(a_res, ex, x, phi, u, Au, Tacc)
            for xi in eachindex(x)
                pyx[xi] =
                    _mzi_pyx_likelihood(a_res[xi], M, Ns, rho0) *
                    (factorial(N - ei) / factorial(N))
            end

            phi_update = calculate_online{MZI_TARGET_MAP[target]}(
                x,
                p,
                pyx,
                a_res,
                ex,
                rho0,
                N,
                ei,
                phi_span,
                exp_ix,
            )

            println(
                "The adjustments of the feedback phase is $(abs(phi_update-phi)) ($ei episodes)",
            )
            append!(xout, abs(phi_update - phi))
            append!(y, u)
            phi = phi_update
        end
        savefile_online(xout, y)
    end
end

"""

    calculate_online{target}(x, p, pyx, a_res, a, rho0, N, ei, phi_span, exp_ix)

Compute the optimal tunable phase online using the specified target function (`sharpness` or `MI`).
"""
function calculate_online{sharpness}(x, p, pyx, a_res, ex::_MZIExec, rho0, N, ei, phi_span, exp_ix)
    M1_res = trapz(x, pyx .* p)
    fac = factorial(N - (ei + 1)) / factorial(N)
    M_res = zeros(length(phi_span))
    pyx0, pyx1 = zeros(length(x)), zeros(length(x))
    Au, A0, A1, M, Ns = _mzi_online_scratch(ex)
    for mj in eachindex(phi_span)
        for xj in eachindex(x)
            _mzi_fill_a_u!(Au, ex, x[xj], phi_span[mj], 0)
            mul!(A0, a_res[xj], Au)
            pyx0[xj] = _mzi_pyx_likelihood(A0, M, Ns, rho0) * fac
            _mzi_fill_a_u!(Au, ex, x[xj], phi_span[mj], 1)
            mul!(A1, a_res[xj], Au)
            pyx1[xj] = _mzi_pyx_likelihood(A1, M, Ns, rho0) * fac
        end
        # M1 and M2 are loop-invariant in the legacy formula; only the final
        # xj value of M2 was used, so evaluating it here is the same result.
        M2_res = abs(trapz(x, pyx0 .* p .* exp_ix)) + abs(trapz(x, pyx1 .* p .* exp_ix))
        M_res[mj] = M2_res / M1_res
    end
    indx_m = findmax(M_res)[2]
    phi_span[indx_m]
end

"""
    calculate_online{MI}(x, p, pyx, a_res, a, rho0, N, ei, phi_span, exp_ix)

Mutual-information variant: returns the tunable phase that maximizes the mutual information.
"""
function calculate_online{MI}(x, p, pyx, a_res, ex::_MZIExec, rho0, N, ei, phi_span, exp_ix)
    M1_res = trapz(x, pyx .* p)
    fac = factorial(N - (ei + 1)) / factorial(N)
    M_res = zeros(length(phi_span))
    pyx0, pyx1 = zeros(length(x)), zeros(length(x))
    Au, A0, A1, M, Ns = _mzi_online_scratch(ex)
    for mj in eachindex(phi_span)
        for xj in eachindex(x)
            _mzi_fill_a_u!(Au, ex, x[xj], phi_span[mj], 0)
            mul!(A0, a_res[xj], Au)
            pyx0[xj] = _mzi_pyx_likelihood(A0, M, Ns, rho0) * fac
            _mzi_fill_a_u!(Au, ex, x[xj], phi_span[mj], 1)
            mul!(A1, a_res[xj], Au)
            pyx1[xj] = _mzi_pyx_likelihood(A1, M, Ns, rho0) * fac
        end
        M2_res =
            trapz(x, pyx0 .* p .* log.(2, pyx0 ./ trapz(x, pyx0 .* p))) +
            trapz(x, pyx1 .* p .* log.(2, pyx1 ./ trapz(x, pyx1 .* p)))
        M_res[mj] = M2_res / M1_res
    end
    indx_m = findmax(M_res)[2]
    phi_span[indx_m]
end

"""
    savefile_online(xout, y)

Save online adaptive estimation results `xout` and `y` to `adaptive.dat` via JLD2.
"""
function savefile_online(xout, y)
    jldopen("adaptive.dat", "w") do f
        f["x"] = xout
        f["y"] = y
    end

    # df = DataFrame(x = xout, y = y)
    # CSV.write("adaptive.csv", df; append = true)
end

##========== offline ==========##
@doc raw"""

    offline(apt::Adapt_MZI, alg; target::Symbol=:sharpness, eps = nothing, seed=1234)

Offline adaptive phase estimation in the MZI.
- `apt`: Adaptive MZI struct which contains `x`, `p`, and `rho0`.
- `alg`: The algorithms for searching the optimal tunable phase. Here, DE and PSO are available. 
- `target`: Setting the target function for calculating the tunable phase. Options are: "sharpness" and "MI".
- `eps`: Legacy relative floor factor; when `nothing`, the relative precision policy is used; when given, it is deprecated and interpreted by the core.
- `seed`: Random seed.
"""
function offline(
    apt::Adapt_MZI,
    alg;
    target::Symbol = :sharpness,
    eps = nothing,
    seed = 1234,
)
    rng = MersenneTwister(seed)
    (; x, p, rho0) = apt
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1) |> sparse
    comb = brgd(N) |> x -> [[parse(Int, s) for s in ss] for ss in x]
    if alg isa DE
        (; p_num, ini_population, c, cr, max_episode) = alg
        if isnothing(ini_population)
            ini_population = ([apt.rho0],)
        end
        DE_deltaphiOpt(
            x,
            p,
            rho0,
            comb,
            p_num,
            ini_population[1],
            c,
            cr,
            rng,
            max_episode,
            target,
            eps,
        )
    elseif alg isa PSO
        (; p_num, ini_particle, c0, c1, c2, max_episode) = alg
        if isnothing(ini_particle)
            ini_particle = ([apt.rho0],)
        end
"""
    PSO_deltaphiOpt(x, p, rho0, comb, p_num, ini_particle, c0, c1, c2, seed::Number, max_episode, target, eps)

Convenience wrapper that converts a numeric `seed` to a `MersenneTwister` RNG and dispatches to the main `PSO_deltaphiOpt`.
"""
PSO_deltaphiOpt(
            x,
            p,
            rho0,
            comb,
            p_num,
            ini_particle[1],
            c0,
            c1,
            c2,
            rng,
            max_episode,
            target,
            eps,
        )
    end
end

"""
    DE_deltaphiOpt(x, p, rho0, comb, p_num, ini_population, c, cr, rng, max_episode, target, eps)

Optimize feedback phase adjustments using Differential Evolution (DE).
"""
function DE_deltaphiOpt(
    x,
    p,
    rho0,
    comb,
    p_num,
    ini_population,
    c,
    cr,
    rng::AbstractRNG,
    max_episode,
    target::Symbol,
    eps,
)
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1) |> sparse
    ex = _mzi_exec_bake(a)
    deltaphi = [zeros(N) for i = 1:p_num]
    # initialize
    res = logarithmic(2.0 * pi, N)
    if length(ini_population) > p_num
        ini_population = [ini_population[i] for i = 1:p_num]
    end
    for pj in eachindex(ini_population)
        deltaphi[pj] = [ini_population[pj][i] for i = 1:N]
    end
    for pk = (length(ini_population)+1):p_num
        deltaphi[pk] = [res[i] + rand(rng) for i = 1:N]
    end

    p_fit = [0.0 for i = 1:p_num]
    for pl = 1:N
        p_fit[pl] = calculate_offline{MZI_TARGET_MAP[target]}(deltaphi[pl], x, p, rho0, ex, comb, eps)
    end

    f_ini = maximum(p_fit)
    f_list = [f_ini]
    for ei = 1:(max_episode-1)
        for pm = 1:p_num
            #mutations
            mut_num = sample(rng, 1:p_num, 3, replace = false)
            deltaphi_mut = [0.0 for i = 1:N]
            for ci = 1:N
                deltaphi_mut[ci] =
                    deltaphi[mut_num[1]][ci] +
                    c * (deltaphi[mut_num[2]][ci] - deltaphi[mut_num[3]][ci])
            end
            #crossover
            deltaphi_cross = [0.0 for i = 1:N]
            cross_int = sample(rng, 1:N, 1, replace = false)[1]
            for cj = 1:N
                rand_num = rand(rng)
                if rand_num <= cr
                    deltaphi_cross[cj] = deltaphi_mut[cj]
                else
                    deltaphi_cross[cj] = deltaphi[pm][cj]
                end
                deltaphi_cross[cross_int] = deltaphi_mut[cross_int]
            end
            #selection
            for cm = 1:N
                deltaphi_cross[cm] =
                    (x -> x < 0.0 ? 0.0 : x > pi ? pi : x)(deltaphi_cross[cm])
            end
            f_cross =
                calculate_offline{MZI_TARGET_MAP[target]}(deltaphi_cross, x, p, rho0, ex, comb, eps)
            if f_cross > p_fit[pm]
                p_fit[pm] = f_cross
                for ck = 1:N
                    deltaphi[pm][ck] = deltaphi_cross[ck]
                end
            end
        end
        append!(f_list, maximum(p_fit))
    end
    savefile_offline(deltaphi[findmax(p_fit)[2]], f_list)
    return deltaphi[findmax(p_fit)[2]]
end

"""

    PSO_deltaphiOpt(x, p, rho0, comb, p_num, ini_particle, c0, c1, c2, rng, max_episode, target, eps)

Optimize feedback phase adjustments using Particle Swarm Optimization (PSO).
"""
function PSO_deltaphiOpt(
    x,
    p,
    rho0,
    comb,
    p_num,
    ini_particle,
    c0,
    c1,
    c2,
    rng::AbstractRNG,
    max_episode,
    target::Symbol,
    eps,
)
    N = Int(sqrt(size(rho0, 1))) - 1
    a = destroy(N + 1) |> sparse
    ex = _mzi_exec_bake(a)
    n = size(a)[1]

    if typeof(max_episode) == Int
        max_episode = [max_episode, max_episode]
    end

    deltaphi = [zeros(N) for i = 1:p_num]
    velocity = [zeros(N) for i = 1:p_num]
    # initialize
    res = logarithmic(2.0 * pi, N)
    if length(ini_particle) > p_num
        ini_particle = [ini_particle[i] for i = 1:p_num]
    end
    for pj in eachindex(ini_particle)
        deltaphi[pj] = [ini_particle[pj][i] for i = 1:N]
    end
    for pk = (length(ini_particle)+1):p_num
        deltaphi[pk] = [res[i] + rand(rng) for i = 1:N]
    end
    for pl = 1:p_num
        velocity[pl] = [0.1 * rand(rng) for i = 1:N]
    end

    pbest = [zeros(N) for i = 1:p_num]
    gbest = zeros(N)
    fit = 0.0
    p_fit = [0.0 for i = 1:p_num]
    f_list = []
    for ei = 1:(max_episode[1]-1)
        for pm = 1:p_num
            f_now = calculate_offline{MZI_TARGET_MAP[target]}(deltaphi[pm], x, p, rho0, ex, comb, eps)
            if f_now > p_fit[pm]
                p_fit[pm] = f_now
                for ci = 1:N
                    pbest[pm][ci] = deltaphi[pm][ci]
                end
            end
        end

        for pn = 1:p_num
            if p_fit[pn] > fit
                fit = p_fit[pn]
                for cj = 1:N
                    gbest[cj] = pbest[pn][cj]
                end
            end
        end

        for pa = 1:p_num
            deltaphi_pre = [0.0 for i = 1:N]
            for ck = 1:N
                deltaphi_pre[ck] = deltaphi[pa][ck]
                velocity[pa][ck] =
                    c0 * velocity[pa][ck] +
                    c1 * rand(rng) * (pbest[pa][ck] - deltaphi[pa][ck])
                +c2 * rand(rng) * (gbest[ck] - deltaphi[pa][ck])
                deltaphi[pa][ck] += velocity[pa][ck]
            end

            for cn = 1:N
                deltaphi[pa][cn] = (x -> x < 0.0 ? 0.0 : x > pi ? pi : x)(deltaphi[pa][cn])
                velocity[pa][cn] = deltaphi[pa][cn] - deltaphi_pre[cn]
            end
        end
        append!(f_list, fit)
        if ei % max_episode[2] == 0
            for pb = 1:p_num
                deltaphi[pb] = [gbest[i] for i = 1:N]
            end
        end
    end
    savefile_offline(gbest, f_list)
    return gbest
end

"""

    calculate_offline{target}(delta_phi, x, p, rho0, ex, comb, eps)

Compute the objective function value for offline optimization with given phase adjustments.
"""
function calculate_offline{sharpness}(delta_phi, x, p, rho0, ex::_MZIExec, comb, eps)
    N = size(ex.a, 1) - 1
    dim = (N + 1)^2
    exp_ix = [exp(1.0im * xi) for xi in x]
    Au = _mzi_au_buffer(ex)
    Tacc = Matrix{ComplexF64}(undef, dim, dim)
    M = Matrix{ComplexF64}(undef, dim, dim)
    Ns = Matrix{ComplexF64}(undef, dim, dim)

    M_res = zeros(length(comb))
    for ui in eachindex(comb)
        u = comb[ui]
        phi = 0.0

        a_res = [Matrix{ComplexF64}(I, dim, dim) for i in eachindex(x)]
        for ei = 1:N-1
            phi = phi - (-1)^u[ei] * delta_phi[ei]
            _mzi_accumulate!(a_res, ex, x, phi, u[ei], Au, Tacc)
        end

        pyx = zeros(length(x))
        for xj in eachindex(x)
            pyx[xj] = _mzi_pyx_likelihood(a_res[xj], M, Ns, rho0) * (1 / factorial(N))
        end
        M_res[ui] = abs(trapz(x, pyx .* p .* exp_ix))
    end
    return sum(M_res)
end

"""
    calculate_offline{MI}(delta_phi, x, p, rho0, ex, comb, eps)

Mutual-information variant of offline evaluation for given phase adjustments.
"""
function calculate_offline{MI}(delta_phi, x, p, rho0, ex::_MZIExec, comb, eps)
    N = size(ex.a, 1) - 1
    dim = (N + 1)^2
    exp_ix = [exp(1.0im * xi) for xi in x]
    Au = _mzi_au_buffer(ex)
    Tacc = Matrix{ComplexF64}(undef, dim, dim)
    M = Matrix{ComplexF64}(undef, dim, dim)
    Ns = Matrix{ComplexF64}(undef, dim, dim)

    M_res = zeros(length(comb))
    for ui in eachindex(comb)
        u = comb[ui]
        phi = 0.0

        a_res = [Matrix{ComplexF64}(I, dim, dim) for i in eachindex(x)]
        for ei = 1:N-1
            phi = phi - (-1)^u[ei] * delta_phi[ei]
            _mzi_accumulate!(a_res, ex, x, phi, u[ei], Au, Tacc)
        end

        pyx = zeros(length(x))
        for xj in eachindex(x)
            pyx[xj] = _mzi_pyx_likelihood(a_res[xj], M, Ns, rho0) * (1 / factorial(N))
        end
        M_res[ui] = trapz(x, pyx .* p .* log.(2, pyx ./ trapz(x, pyx .* p)))
    end
    return sum(M_res)
end

"""
    savefile_offline(deltaphi, flist)

Save offline optimization results: `deltaphi` to `deltaphi.csv` and objective values `flist` to `f.csv`.
raw"""
function savefile_offline(deltaphi, flist)
    # JLD2 save
    open("deltaphi.csv", "w") do m
        writedlm(m, deltaphi)
    end
 
    # df = DataFrame(deltaphi = deltaphi)
    # CSV.write("deltaphi.csv", df)

    # JLD2 save
    open("f.csv", "w") do n
        writedlm(n, flist)
    end

    # df = DataFrame(f = flist)
    # CSV.write("f.csv", df)
end

@doc raw"""
    a_u(a, x, phi, u)

Photon-subtraction operator for the MZI after feedback:

```math
A_u = a_{\mathrm{in}} \sin\left(\frac{x - \phi}{2} + \frac{\pi u}{2}\right) + b_{\mathrm{in}} \cos\left(\frac{x - \phi}{2} + \frac{\pi u}{2}\right)
```
"""
function a_u(a, x, phi, u)
    N = size(a)[1] - 1
    a_in = kron(a, Matrix(I, N + 1, N + 1))
    b_in = kron(Matrix(I, N + 1, N + 1), a)

    value = 0.5 * (x - phi) + 0.5 * pi * u
    return a_in * sin(value) + b_in * cos(value)
end

raw"""

    logarithmic(number, N)

Generate a logarithmic sequence of length ``N``: ``[\mathrm{number}/2, \mathrm{number}/4, \dots, \mathrm{number}/2^N]``.
"""
function logarithmic(number, N)
    res = zeros(N)
    res_tp = number
    for i = 1:N
        res_tp = res_tp / 2
        res[i] = res_tp
    end
    return res
end

"""

    brgd(n)

Generate the binary reflected Gray code of length ``2^n`` as a vector of binary strings.
"""
function brgd(n)
    if n == 1
        return ["0", "1"]
    end
    L0 = brgd(n - 1)
    L1 = deepcopy(L0)
    reverse!(L1)
    L0 = ["0" * l for l in L0]
    L1 = ["1" * l for l in L1]
    return deepcopy(vcat(L0, L1))
end


"""

    adapt!(scheme::Adapt_MZI; target=:sharpness, output="phi")

Run online adaptive estimation. Delegates to `online`.
"""
function adapt!(scheme::Adapt_MZI; target::Symbol = :sharpness, output::String = "phi")
    return online(scheme, target = target, output = output)
end

"""

    adapt!(scheme::Adapt_MZI, alg; target=:sharpness, eps=nothing, seed=1234)

Run offline adaptive estimation with a given optimization algorithm (`DE` or `PSO`). Delegates to `offline`.

`eps` is a legacy keyword; when `nothing`, the new relative precision policy is used; when given, it is deprecated and interpreted by the core.
"""
function adapt!(
    scheme::Adapt_MZI,
    alg;
    target::Symbol = :sharpness,
    eps = nothing,
    seed = 1234,
)
    return offline(scheme, alg, target = target, eps = eps, seed = seed)
end
