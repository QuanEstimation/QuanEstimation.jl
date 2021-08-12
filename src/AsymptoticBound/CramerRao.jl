function CFI(ρ, dρ, M)
    m_num = length(M)
    p = zero(ComplexF64)
    dp = zero(ComplexF64)
    F = 0.
    for i in 1:m_num
        mp = M[i]
        p += tr(ρ * mp)
        dp = tr(dρ * mp)
        cadd = 0.
        if p != 0
            cadd = (dp^2) / p
        end
        F += cadd
    end 
    real(F)
end
function CFI(M::Vector{Matrix{T}}, H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    ρt = evolute(H[1], Liouville_operator, γ, times, times[1]) * (ρ_initial |> vec)
    ∂ρt_∂x = -im * Δt * liouville_commu(∂H_∂x) * ρt
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt=  expL * ρt
        ∂ρt_∂x= -im * Δt * liouville_commu(∂H_∂x) * ρt + expL * ∂ρt_∂x
    end
    CFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end

function CFIM(ρ, dρ, M)
    m_num = length(M)
    cfim = [tr.(kron(dρ', dρ).*M[i]) / tr(ρ*M[i])  for i in 1:m_num] |> sum
end
function CFIM(M::Vector{Matrix{T}}, H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    para_num = length(∂H_∂x)
    ρt = evolute(H[1], Liouville_operator, γ, times, times[1]) * ρ_initial[:]
    ∂ρt_∂x = [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num]
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt=  expL * ρt
        ∂ρt_∂x= [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    CFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat, M)
end


function SLD(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    2 * pinv(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ)) * vec(∂ρ_∂x) |> vec2mat
end
function SLD(ρ::Vector{T},∂ρ_∂x::Vector{T}) where {T <: Complex}
    SLD(ρ |> vec2mat, ∂ρ_∂x |> vec2mat)
end
function SLD(ρ::Matrix{T}, ∂ρ_∂x::Vector{Matrix{T}}) where {T <: Complex}
    (x->SLD(ρ, x)).(∂ρ_∂x)
end
function SLD_qr(ρ::Matrix{T}, ∂ρ_∂x::Matrix{T}) where {T <: Complex}
    2 * (qr(kron(ρ |> transpose, ρ |> one) + kron(ρ |> one, ρ), Val(true)) \ vec(∂ρ_∂x)) |> vec2mat
end
function SLD_eig(ρ::Array{T}, dρ::Array{T})::Array{T} where {T <: Complex}
    dim = size(ρ)[1]
    if typeof(dρ) == Array{T,2}
        purity = tr(ρ * ρ)
        SLD_res = zeros(T, dim, dim)
        if abs(1 - purity) < 1e-8
            SLD_res = 2 * dρ
        else
            val, vec_mat = eigen(ρ)
            for fi in 1:dim
                for fj in 1:dim
                    coeff = 2 / (val[fi] + val[fj])
                    SLD_res[fi, fj] = coeff * (vec_mat[:,fi]' * (dρ * vec_mat[:, fj]))
                end
            end
            SLD_res[findall(SLD_res == Inf)] .= 0.
            SLD_res = vec_mat * (SLD_res * vec_mat')
        end
    else
        # multi-parameter scenario
        purity = tr(ρ * ρ)
        if abs(1 - purity) < 1e-8
            SLD_res = [2 * dρ[i] for i in 1:length(dρ)]
        else
            # SLD_res = [zeros(T,dim,dim) for i in 1:length(dρ)]
            dim = ndims(ρ)
            val, vec_mat = eigens(ρ)
            for para_i in 1:length(dρ)
                SLD_tp = zeros(T, dim, dim)
                for fi in 1:dim
                    for fj in 1:dim
                        coeff = 2. / (val[fi] + val[fj])
                        SLD_tp[fi][fj] = coeff * (vec[fi]' * (dρ[para_i] * vec[fj]))
                    end
                end
                SLD_tp[findall(SLD_rp == Inf)] .= 0.
                SLD_res[para_i] = vec_mat * (SLD_tp * vec_mat')
            end
        end
    end
    SLD_res
end
function RLD(ρ::Matrix{T}, dρ::Matrix{T}) where {T <: Complex}
    dρ * pinv(ρ)
end
function QFI_RLD(ρ, dρ)
    RLD_tp = RLD(ρ, dρ)
    F = tr(ρ * RLD_tp * RLD_tp')
    F |> real
end
function QFI(ρ, dρ)
    SLD_tp = SLD(ρ, dρ)
    SLD2_tp = SLD_tp * SLD_tp
    F = tr(ρ * SLD2_tp)
    F |> real
end
function QFIM(ρ, dρ)
    SLD_tp = SLD(ρ, dρ)
    [0.5*ρ] .* (kron(SLD_tp, SLD_tp') + kron(SLD_tp', SLD_tp)).|> tr .|>real 
end
function QFI(H::Vector{Matrix{T}}, ∂H_∂x::Matrix{T},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    ρt = evolute(H[1], Liouville_operator, γ, times, times[1]) * (ρ_initial |> vec)
    ∂ρt_∂x = -im * Δt * liouville_commu(∂H_∂x) * ρt
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt=  expL * ρt
        ∂ρt_∂x= -im * Δt * liouville_commu(∂H_∂x) * ρt + expL * ∂ρt_∂x
    end
    QFI(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end
function QFIM(H::Vector{Matrix{T}}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ,  times) where {T <: Complex,R <: Real}
    dim = size(H[1])[1]
    Δt = times[2] - times[1]
    para_num = length(∂H_∂x)
    ρt = evolute(H[1], Liouville_operator, γ, times, times[1]) * ρ_initial[:]
    ∂ρt_∂x = [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num]
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt=  expL * ρt
        ∂ρt_∂x= [-im * Δt * liouville_commu(∂H_∂x[i]) * ρt for i in 1:para_num] + [expL] .* ∂ρt_∂x
    end
    QFIM(ρt|> vec2mat, ∂ρt_∂x|> vec2mat)
end
function CFI(M, system)
    CFI(M,Htot(system.freeHamiltonian, system.control_Hamiltonian, system.control_coefficients), system.Hamiltonian_derivative[1], system.ρ_initial, system.Liouville_operator, system.γ, system.times)
end
function CFIM(M, system)
    CFIM(M,Htot(system.freeHamiltonian, system.control_Hamiltonian, system.control_coefficients), system.Hamiltonian_derivative, system.ρ_initial, system.Liouville_operator, system.γ, system.times)
end
function QFI(system)
    QFI(Htot(system.freeHamiltonian, system.control_Hamiltonian, system.control_coefficients), system.Hamiltonian_derivative[1], system.ρ_initial, system.Liouville_operator, system.γ, system.times)
end
function QFIM(system)
    QFIM(Htot(system.freeHamiltonian, system.control_Hamiltonian, system.control_coefficients), system.Hamiltonian_derivative, system.ρ_initial, system.Liouville_operator, system.γ, system.times)
end