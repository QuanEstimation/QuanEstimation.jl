
# dynamics in Lindblad form
struct Lindblad{N,C,R,P} <: AbstractDynamics
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
    state_rep::Symbol
    para_type::Symbol
end

Lindblad(data, N, C, R) =
    para_type(data) |> P -> Lindblad{((N, C, R, P) .|> eval)...}(data, N, C, R, P)
Lindblad(data, N, C) = Lindblad(data, N, C, :dm)

struct Lindblad_noiseless_free <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
end

struct Lindblad_noisy_free <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end


struct Lindblad_noiseless_timedepend <: AbstractDynamicsData
    freeHamiltonian::AbstractVector
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
end

struct Lindblad_noisy_timedepend <: AbstractDynamicsData
    freeHamiltonian::AbstractVector
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end

struct Lindblad_noiseless_controlled <: AbstractDynamicsData
    freeHamiltonian::AbstractVecOrMat
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
end

struct Lindblad_noisy_controlled <: AbstractDynamicsData
    freeHamiltonian::AbstractVecOrMat
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
end


struct Lindblad_noiseless_free_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
end

struct Lindblad_noisy_free_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end

struct Lindblad_noiseless_timedepend_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractVector
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
end

struct Lindblad_noisy_timedepend_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractVector
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
end
struct Lindblad_noiseless_controlled_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractVecOrMat
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
end

struct Lindblad_noisy_controlled_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractVecOrMat
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
end


para_type(data) = length(data.Hamiltonian_derivative) == 1 ? :single_para : :multi_para

# Constructor of Lindblad dynamics
Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
) = Lindblad(Lindblad_noiseless_free(H0, dH, ρ0, tspan), :noiseless, :free)

Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(Lindblad_noisy_free(H0, dH, ρ0, tspan, decay_opt, γ), :noisy, :free)

Lindblad(
    H0::AbstractVector,
    dH::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
) = Lindblad(Lindblad_noiseless_timedepend(H0, dH, ρ0, tspan), :noiseless, :timedepend)

Lindblad(
    H0::AbstractVector,
    dH::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(
    Lindblad_noisy_timedepend(H0, dH, ρ0, tspan, decay_opt, γ),
    :noisy,
    :timedepend,
)

Lindblad(
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Hc::AbstractVector,
    ctrl::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
) = Lindblad(
    Lindblad_noiseless_controlled(H0, dH, ρ0, tspan, Hc, ctrl),
    :noiseless,
    :controlled,
)

Lindblad(
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Hc::AbstractVector,
    ctrl::AbstractVector,
    ρ0::AbstractMatrix,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(
    Lindblad_noisy_controlled(H0, dH, ρ0, tspan, decay_opt, γ, Hc, ctrl),
    :noisy,
    :controlled,
)

Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
) = Lindblad(Lindblad_noiseless_free(H0, dH, ψ0, tspan), :noiseless, :free, :ket)

Lindblad(
    H0::AbstractMatrix,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(Lindblad_noisy_free(H0, dH, ψ0, tspan, decay_opt, γ), :noisy, :free, :ket)

Lindblad(
    H0::AbstractVector,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
) = Lindblad(
    Lindblad_noiseless_timedepend(H0, dH, ψ0, tspan),
    :noiseless,
    :timedepend,
    :ket,
)

Lindblad(
    H0::AbstractVector,
    dH::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(
    Lindblad_noisy_timedepend(H0, dH, ψ0, tspan, decay_opt, γ),
    :noisy,
    :timedepend,
    :ket,
)

Lindblad(
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Hc::AbstractVector,
    ctrl::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
) = Lindblad(
    Lindblad_noiseless_controlled(H0, dH, ψ0, tspan, Hc, ctrl),
    :noiseless,
    :controlled,
    :ket,
)

Lindblad(
    H0::AbstractVecOrMat,
    dH::AbstractVector,
    Hc::AbstractVector,
    ctrl::AbstractVector,
    ψ0::AbstractVector,
    tspan::AbstractVector,
    decay_opt::AbstractVector,
    γ::AbstractVector,
) = Lindblad(
    Lindblad_noisy_controlled(H0, dH, ψ0, tspan, decay_opt, γ, Hc, ctrl),
    :noisy,
    :controlled,
    :ket,
)

# functions for evolve dynamics in Lindblad form
function liouville_commu(H)
    kron(one(H), H) - kron(H |> transpose, one(H))
end

function liouville_dissip(Γ)
    kron(Γ |> conj, Γ) - 0.5 * kron((Γ |> transpose) * (Γ |> conj), Γ |> one) -
    0.5 * kron(Γ |> one, Γ' * Γ)
end

function liouville_commu_py(A::Array{T}) where {T<:Complex}
    dim = size(A)[1]
    result = zeros(T, dim^2, dim^2)
    @inbounds for i = 1:dim
        @inbounds for j = 1:dim
            @inbounds for k = 1:dim
                ni = dim * (i - 1) + j
                nj = dim * (k - 1) + j
                nk = dim * (i - 1) + k

                result[ni, nj] = A[i, k]
                result[ni, nk] = -A[k, j]
                result[ni, ni] = A[i, i] - A[j, j]
            end
        end
    end
    result
end

function liouville_dissip_py(A::Array{T}) where {T<:Complex}
    dim = size(A)[1]
    result = zeros(T, dim^2, dim^2)
    @inbounds for i = 1:dim
        @inbounds for j = 1:dim
            ni = dim * (i - 1) + j
            @inbounds for k = 1:dim
                @inbounds for l = 1:dim
                    nj = dim * (k - 1) + l
                    L_temp = A[i, k] * conj(A[j, l])
                    @inbounds for p = 1:dim
                        L_temp -=
                            0.5 * float(k == i) * A[p, j] * conj(A[p, l]) +
                            0.5 * float(l == j) * A[p, k] * conj(A[p, i])
                    end
                    result[ni, nj] = L_temp
                end
            end
        end
    end
    result[findall(abs.(result) .< 1e-10)] .= 0.0
    result
end

function dissipation(
    Γ::Vector{Matrix{T}},
    γ::Vector{R},
    t::Int = 0,
) where {T<:Complex,R<:Real}
    [γ[i] * liouville_dissip(Γ[i]) for i = 1:length(Γ)] |> sum
end

function dissipation(
    Γ::Vector{Matrix{T}},
    γ::Vector{Vector{R}},
    t::Int = 0,
) where {T<:Complex,R<:Real}
    [γ[i][t] * liouville_dissip(Γ[i]) for i = 1:length(Γ)] |> sum
end

function free_evolution(H0)
    -1.0im * liouville_commu(H0)
end

function liouvillian(
    H::Matrix{T},
    decay_opt::Vector{Matrix{T}},
    γ,
    t::Real,
) where {T<:Complex}
    freepart = liouville_commu(H)
    dissp = norm(γ) + 1 ≈ 1 ? freepart |> zero : dissipation(decay_opt, γ, t)
    -1.0im * freepart + dissp
end

function Htot(H0::Matrix{T}, Hc::Vector{Matrix{T}}, ctrl) where {T<:Complex,R}
    Htot = [H0] .+ ([ctrl[i] .* [Hc[i]] for i = 1:length(ctrl)] |> sum)
end

function Htot(
    H0::Matrix{T},
    Hc::Vector{Matrix{T}},
    ctrl::Vector{R},
) where {T<:Complex,R<:Real}
    Htot = H0 + ([ctrl[i] * Hc[i] for i = 1:length(ctrl)] |> sum)
end

function Htot(H0::Vector{Matrix{T}}, Hc::Vector{Matrix{T}}, ctrl) where {T<:Complex}
    Htot = H0 + ([ctrl[i] .* [Hc[i]] for i = 1:length(ctrl)] |> sum)
end

function expL(H, decay_opt, γ, dt, tj)
    Ld = dt * liouvillian(H, decay_opt, γ, tj)
    exp(Ld)
end

function expL(H, dt)
    freepart = liouville_commu(H)
    Ld = -1.0im * dt * freepart
    exp(Ld)
end

function expm(
    H0::Matrix{T},
    ∂H_∂x::Matrix{T},
    ρ0::Matrix{T},
    decay_opt::Vector{Matrix{T}},
    γ,
    Hc::Vector{Matrix{T}},
    ctrl::Vector{Vector{R}},
    tspan,
) where {T<:Complex,R<:Real}

    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]

    H = Htot(H0, Hc, ctrl)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]

    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i = 1:length(tspan)]
    ∂ρt_∂x_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i = 1:length(tspan)]
    ρt_all[1] = ρ0 |> vec
    ∂ρt_∂x_all[1] = ρt_all[1] |> zero

    for t = 2:length(tspan)
        expL = evolve(H[t-1], decay_opt, γ, Δt, t)
        ρt_all[t] = expL * ρt_all[t-1]
        ∂ρt_∂x_all[t] = -im * Δt * ∂H_L * ρt_all[t] + expL * ∂ρt_∂x_all[t-1]
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

function expm(
    H0::Matrix{T},
    ∂H_∂x::Vector{Matrix{T}},
    ρ0::Matrix{T},
    decay_opt::Vector{Matrix{T}},
    γ,
    Hc::Vector{Matrix{T}},
    ctrl::Vector{Vector{R}},
    tspan,
) where {T<:Complex,R<:Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]

    H = Htot(H0, Hc, ctrl)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]

    Δt = tspan[2] - tspan[1]

    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i = 1:length(tspan)]
    ∂ρt_∂x_all = [
        [Vector{ComplexF64}(undef, (length(H0))^2) for j = 1:para_num] for
        i = 1:length(tspan)
    ]
    ρt_all[1] = ρ0 |> vec
    for pj = 1:para_num
        ∂ρt_∂x_all[1][pj] = ρt_all[1] |> zero
    end

    for t = 2:length(tspan)
        expL = evolve(H[t-1], decay_opt, γ, Δt, t)
        ρt_all[t] = expL * ρt_all[t-1]
        for pj = 1:para_num
            ∂ρt_∂x_all[t][pj] = -im * Δt * ∂H_L[pj] * ρt_all[t] + expL * ∂ρt_∂x_all[t-1][pj]
        end
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

function secondorder_derivative(
    H0,
    ∂H_∂x::Vector{Matrix{T}},
    ∂2H_∂x::Vector{Matrix{T}},
    ρ0::Matrix{T},
    decay_opt::Vector{Matrix{T}},
    γ,
    Hc::Vector{Matrix{T}},
    ctrl::Vector{Vector{R}},
    tspan,
) where {T<:Complex,R<:Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]

    H = Htot(H0, Hc, ctrl)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    ∂2H_L = [liouville_commu(∂2H_∂x[i]) for i = 1:para_num]

    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    ∂2ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = evolve(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
        ∂2ρt_∂x =
            [
                (-im * Δt * ∂2H_L[i] + Δt * Δt * ∂H_L[i] * ∂H_L[i]) * ρt -
                2 * im * Δt * ∂H_L[i] * ∂ρt_∂x[i] for i = 1:para_num
            ] + [expL] .* ∂2ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat, ∂2ρt_∂x |> vec2mat
end

#### evolution of pure states under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,free,ket})
    (H0, ∂H_∂x, psi0, tspan) = dynamics

    para_num = length(∂H_∂x)
    Δt = tspan[2] - tspan[1]
    U = exp(-im * H0 * Δt)
    psi_t = psi0
    ∂psi_∂x = [psi0 |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        psi_t = U * psi_t
        ∂psi_∂x = [-im * Δt * ∂H_∂x[i] * psi_t for i = 1:para_num] + [U] .* ∂psi_∂x
    end
    ρt = psi_t * psi_t'
    ∂ρt_∂x = [(∂psi_∂x[i] * psi_t' + psi_t * ∂psi_∂x[i]') for i = 1:para_num]
    ρt, ∂ρt_∂x
end

#### evolution of pure states under time-dependent Hamiltonian without noise and controls ####
function expL(dynamics::Lindblad{noiseless,timedepend,ket})
    (H0, ∂H_∂x, psi0, tspan) = dynamics

    para_num = length(∂H_∂x)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    ρt = (psi0 * psi0') |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], Δt)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,free,dm})
    (H0, ∂H_∂x, ρ0, tspan) = dynamics

    para_num = length(∂H_∂x)
    Δt = tspan[2] - tspan[1]
    expL = expL(H0, Δt)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-dependent Hamiltonian without noise and controls ####
function evolve(dynamics::Lindblad{noiseless,timedepend,dm})
    (H0, ∂H_∂x, ρ0, tspan) = dynamics

    para_num = length(∂H_∂x)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], Δt)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of pure states under time-independent Hamiltonian  
#### with noise but without controls
function evolve(dynamics::Lindblad{noisy,free,ket})
    (H0, ∂H_∂x, psi0, tspan, decay_opt, γ) = dynamics

    para_num = length(∂H_∂x)
    ρt = (psi0 * psi0') |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    Δt = tspan[2] - tspan[1]
    expL = expL(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    for t = 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian  
#### with noise but without controls
function evolve(dynamics::Lindblad{noisy,free,dm})
    (H0, ∂H_∂x, ρ0, tspan, decay_opt, γ) = dynamics

    para_num = length(∂H_∂x)
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    Δt = tspan[2] - tspan[1]
    expL = expL(H0, decay_opt, γ, Δt, 1)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    for t = 2:length(tspan)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of pure states under time-dependent Hamiltonian  
#### with noise but without controls
function evolve(dynamics::Lindblad{noisy,timedepend,ket})
    (H0, ∂H_∂x, psi0, tspan, decay_opt, γ) = dynamics

    para_num = length(∂H_∂x)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    ρt = (psi * psi') |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-dependent Hamiltonian  
#### with noise but without controls
function evolve(dynamics::Lindblad{noisy,timedepend,dm})
    (H0, ∂H_∂x, ρ0, tspan, decay_opt, γ) = dynamics

    para_num = length(∂H_∂x)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H0[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H0[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian 
#### with controls but without noise #### 
function evolve(dynamics::Lindblad{noiseless,controlled,dm})
    (H0, ∂H_∂x, ρ0, tspan, Hc, ctrl) = dynamics

    para_num = length(∂H_∂x)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]
    H = Htot(H0, Hc, ctrl)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H[t-1], Δt)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

#### evolution of density matrix under time-independent Hamiltonian with noise and controls #### 
function evolve(dynamics::Lindblad{noisy,controlled,dm})
    (H0, ∂H_∂x, ρ0, tspan, decay_opt, γ, Hc, ctrl) = dynamics

    para_num = length(∂H_∂x)
    ctrl_num = length(Hc)
    ctrl_interval = ((length(tspan) - 1) / length(ctrl[1])) |> Int
    ctrl = [repeat(ctrl[i], 1, ctrl_interval) |> transpose |> vec for i = 1:ctrl_num]
    H = Htot(H0, Hc, ctrl)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i = 1:para_num]
    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i = 1:para_num]
    for t = 2:length(tspan)
        Δt = tspan[t] - tspan[t-1] # tspan may not be equally spaced 
        expL = expL(H[t-1], decay_opt, γ, Δt, t)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i = 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end
