
# dynamics in Lindblad form
struct Lindblad{N, C, R} <: AbstractDynamics 
    data::AbstractDynamicsData
    noise_type::Symbol
    ctrl_type::Symbol
    state_rep::Symbol
end

Lindblad{n,c}(data::AbstractDynamicsData,N::Symbol,C::Symbol) where {n,c} =Lindblad{n,c,dm}(data,N,C,:dm)

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

struct Lindblad_noiseless_controlled <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
end

struct Lindblad_noisy_controlled <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ρ0::AbstractMatrix
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
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

struct Lindblad_noiseless_controlled_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
end

struct Lindblad_noisy_controlled_pure <: AbstractDynamicsData
    freeHamiltonian::AbstractMatrix
    Hamiltonian_derivative::AbstractVector
    ψ0::AbstractVector
    tspan::AbstractVector
    decay_opt::AbstractVector
    γ::AbstractVector
    control_Hamiltonian::AbstractVector
    control_coefficients::AbstractVector
    ctrl_bound::AbstractVector
end


# Constructor of Lindblad dynamics
Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector) = Lindblad{noiseless, free}(Lindblad_noiseless_free(H0, dH, ρ0, tspan), :noiseless, :free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector) = Lindblad{noisy,free}(Lindblad_noisy_free(H0, dH, ρ0, tspan, decay_opt, γ), :noisy, :free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad{noiseless,controlled}(Lindblad_noiseless_controlled(H0, dH, ρ0, tspan, Hc, cc, cb), :noiseless, :controlled)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ρ0::AbstractMatrix, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector,Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad{noisy, controlled}(Lindblad_noisy_controlled(H0, dH, ρ0, tspan, decay_opt, γ, Hc, cc, cb), :noisy, :controlled)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ψ0::AbstractVector, tspan::AbstractVector) = Lindblad{noiseless, free, ket}(Lindblad_noiseless_free(H0, dH, ψ0, tspan), noiseless, free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ψ0::AbstractVector, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector) = Lindblad{noisy,free,ket}(Lindblad_noisy_free(H0, dH, ψ0, tspan, decay_opt, γ), noisy, free)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ψ0::AbstractVector, tspan::AbstractVector, Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad{noiseless,controlled, ket}(Lindblad_noiseless_controlled(H0, dH, ψ0, tspan, Hc, cc, cb), noiseless, controlled)

Lindblad(H0::AbstractMatrix,dH::AbstractVector, ψ0::AbstractVector, tspan::AbstractVector, decay_opt::AbstractVector,γ::AbstractVector,Hc::AbstractVector, cc::AbstractVector, cb::AbstractVector ) = Lindblad{noisy, controlled, ket}(Lindblad_noisy_controlled(H0, dH, ψ0, tspan, decay_opt, γ, Hc, cc, cb), noisy, controlled)


# functions for evolute dynamics in Lindblad form
function liouville_commu(H) 
    kron(one(H), H) - kron(H |> transpose, one(H))
end

function liouville_dissip(Γ)
    kron(Γ |> conj, Γ) - 0.5 * kron((Γ |> transpose) * (Γ |> conj), Γ |> one) - 0.5 * kron(Γ |> one, Γ' * Γ)
end

function liouville_commu_py(A::Array{T}) where {T <: Complex}
    dim = size(A)[1]
    result = zeros(T, dim^2, dim^2)
    @inbounds for i in 1:dim
        @inbounds for j in 1:dim
            @inbounds for k in 1:dim
                ni = dim * (i - 1) + j
                nj = dim * (k - 1) + j
                nk = dim * (i - 1) + k

                result[ni,nj] = A[i,k]
                result[ni,nk] = -A[k,j]
                result[ni,ni] = A[i,i] - A[j,j]
            end
        end
    end
    result
end

function liouville_dissip_py(A::Array{T}) where {T <: Complex}
    dim = size(A)[1]
    result =  zeros(T, dim^2, dim^2)
    @inbounds for i = 1:dim
        @inbounds for j in 1:dim
            ni = dim * (i - 1) + j
            @inbounds for k in 1:dim
                @inbounds for l in 1:dim 
                    nj = dim * (k - 1) + l
                    L_temp = A[i,k] * conj(A[j,l])
                    @inbounds for p in 1:dim
                        L_temp -= 0.5 * float(k == i) * A[p,j] * conj(A[p,l]) + 0.5 * float(l == j) * A[p,k] * conj(A[p,i])
                    end
                    result[ni,nj] = L_temp
                end
            end 
        end
    end
    result[findall(abs.(result) .< 1e-10)] .= 0.
    result
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{R}, t::Int=0) where {T <: Complex,R <: Real}
    [γ[i] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{Vector{R}}, t::Int=0) where {T <: Complex,R <: Real}
    [γ[i][t] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

function free_evolution(H0)
    -1.0im * liouville_commu(H0)
end

function liouvillian(H::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, t::Real) where {T <: Complex} 
    freepart = liouville_commu(H)
    dissp = norm(γ) +1 ≈ 1 ? freepart|>zero : dissipation(decay_opt, γ, t)
    -1.0im * freepart + dissp
end

function Htot(H0::Matrix{T}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients) where {T <: Complex, R}
    Htot = [H0] .+ ([control_coefficients[i] .* [control_Hamiltonian[i]] for i in 1:length(control_coefficients)] |> sum )
end

function Htot(H0::Matrix{T}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{R}) where {T <: Complex, R<:Real}
    Htot = H0 + ([control_coefficients[i] * control_Hamiltonian[i] for i in 1:length(control_coefficients)] |> sum )
end

function Htot(H0::Vector{Matrix{T}}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients) where {T <: Complex}
    Htot = H0 + ([control_coefficients[i] .* [control_Hamiltonian[i]] for i in 1:length(control_coefficients)] |> sum )
end

function evolute(H, decay_opt, γ, dt, tj)
    Ld = dt * liouvillian(H, decay_opt, γ, tj)
    exp(Ld)
end

function evolute(H, dt)
    freepart = liouville_commu(H)
    Ld = -1.0im * dt * freepart
    exp(Ld)
end

function expm(H0::Matrix{T}, ∂H_∂x::Matrix{T}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}

    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = liouville_commu(∂H_∂x)

    Δt = tspan[2] - tspan[1]

    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(tspan)]
    ∂ρt_∂x_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(tspan)]
    ρt_all[1] = ρ0 |> vec
    ∂ρt_∂x_all[1] = ρt_all[1] |> zero
    
    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt_all[t] = expL * ρt_all[t-1]
        ∂ρt_∂x_all[t] = -im * Δt * ∂H_L * ρt_all[t] + expL * ∂ρt_∂x_all[t-1]
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

function expm(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}}, ρ0::Matrix{T}, decay_opt::Vector{Matrix{T}}, γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, tspan) where {T <: Complex,R <: Real}

    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    ctrl_interval = ((length(tspan)-1)/length(control_coefficients[1])) |> Int
    control_coefficients = [repeat(control_coefficients[i], 1, ctrl_interval) |>transpose |>vec for i in 1:ctrl_num]

    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    Δt = tspan[2] - tspan[1]
    
    ρt_all = [Vector{ComplexF64}(undef, (length(H0))^2) for i in 1:length(tspan)]
    ∂ρt_∂x_all = [[Vector{ComplexF64}(undef, (length(H0))^2) for j in 1:para_num] for i in 1:length(tspan)]
    ρt_all[1] = ρ0 |> vec
    for pj in 1:para_num
        ∂ρt_∂x_all[1][pj] = ρt_all[1] |> zero
    end

    for t in 2:length(tspan)
        expL = evolute(H[t-1], decay_opt, γ, Δt, t)
        ρt_all[t] = expL * ρt_all[t-1]
        for pj in 1:para_num
            ∂ρt_∂x_all[t][pj] = -im * Δt * ∂H_L[pj] * ρt_all[t] + expL* ∂ρt_∂x_all[t-1][pj]
        end
    end
    ρt_all |> vec2mat, ∂ρt_∂x_all |> vec2mat
end

######## evolute dynamics in Lindblad form#######
function evolute(dynamics::Lindblad{noiseless, free, dm})
    (H0, ∂H_∂x, ρ0, tspan) = dynamics
    para_num = length(∂H_∂x)
    tnum = length(tspan)

    H = [H0 for i in 1:tnum]
    ∂H_L = [liouville_commu(∂H_∂x[i]) for i in 1:para_num]

    ρt = ρ0 |> vec
    ∂ρt_∂x = [ρt |> zero for i in 1:para_num]
    for t in 2:length(tspan)
        Δt = tspan[t] - tspan[t-1]
        expL = evolute(H[t-1], Δt)
        ρt = expL * ρt
        ∂ρt_∂x = [-im * Δt * ∂H_L[i] * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    ρt = exp(vec(H[end])' * zero(ρt)) * ρt
    ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end