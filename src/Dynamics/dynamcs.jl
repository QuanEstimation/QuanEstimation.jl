
"""
    liouville_commu()

"""
function liouville_commu(H) 
    kron(one(H), H) - kron(H |> transpose, one(H))
end

"""
    liouville_commu_dissip()

"""
function liouville_dissip(Γ)
    kron(Γ |> conj, Γ) - 0.5 * kron((Γ |> transpose) * Γ |> conj, Γ |> one) - 0.5 * kron(Γ |> one, Γ' * Γ)
end

"""
    dissipation()

"""
function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{R}, t::Real) where {T <: Complex,R <: Real}
    [γ[i] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end
function dissipation(Γ::Vector{Matrix{T}}, γ::Vector{Vector{R}}, t::Real) where {T <: Complex,R <: Real}
    [γ[i][t] * liouville_dissip(Γ[i]) for i in 1:length(Γ)] |> sum
end

"""
    free_evolution()

"""
function free_evolution(H0)
    -1.0im * liouville_commu(H0)
end

"""
    liouvillian()

"""
function liouvillian(H::Matrix{T}, Liouville_operator::Vector{Matrix{T}}, γ, t::Real) where {T <: Complex} 
    freepart = liouville_commu(H)
    dissp = norm(γ) +1 ≈ 1 ? freepart|>zero : dissipation(Liouville_operator, γ, t)
    -1.0im * freepart + dissp
end

"""
    Htot()
"""
function Htot(H0::Matrix{T}, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients) where {T <: Complex}
    Htot = [H0] .+  ([control_coefficients[i] .* [control_Hamiltonian[i]] for i in 1:length(control_coefficients)] |> sum )
end

"""
    evolute()

"""
function evolute(H, Liouville_operator, γ, times, t)
    tj = Int(round((t - times[1]) / (times[2] - times[1]))) + 1 
    dt = times[2] - times[1]
    Ld = dt * liouvillian(H, Liouville_operator, γ, tj)
    exp(Ld)
end

"""
    evolute_ODE!()

"""
function evolute_ODE!(grape::GrapeControl)
    H(p) = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, p)
    dt = grape.times[2] - grape.times[1]    
    tspan = (grape.times[1], grape.times[end])
    u0 = grape.ρ_initial
    Γ = grape.Liouville_operator
    f(u, p, t) = -im * (H(p)[t2Num(tspan[1], dt, t)] * u + u * H(p)[t2Num(tspan[1], dt, t)]) + 
                 ([grape.γ[i] * (Γ[i] * u * Γ[i]' - (Γ[i]' * Γ[i] * u + u * Γ[i]' * Γ[i] )) for i in 1:length(Γ)] |> sum)
    prob = ODEProblem(f, u0, tspan, grape.control_coefficients, saveat=dt)
    sol = solve(prob)
    sol.u
end

"""
    propagate()

"""
function propagate(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}},
                   γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}
    dim = size(H0)[1]
    para_num = length(∂H_∂x)
    H = Htot(H0, control_Hamiltonian, control_coefficients)
    ρt = [Vector{ComplexF64}(undef, dim^2)  for i in 1:length(times)]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for i in 1:length(times)] for para in 1:para_num]
    Δt = times[2] - times[1]
    ρt[1] = evolute(H[1], Liouville_operator, γ, times, times[1]) * (ρ_initial |> vec)
    for para in  1:para_num
        ∂ρt_∂x[para][1] = -im * Δt * liouville_commu(∂H_∂x[para]) * ρt[1]
    end
    for t in 2:length(times)
        expL = evolute(H[t], Liouville_operator, γ, times, times[t])
        ρt[t] =  expL * ρt[t - 1]
        for para in para_num
            ∂ρt_∂x[para][t] = -im * Δt * liouville_commu(∂H_∂x[para]) * ρt[t] + expL * ∂ρt_∂x[para][t - 1]
        end
    end
    ρt .|> vec2mat, ∂ρt_∂x .|> vec2mat
end

"""
    propagate!()

"""
function propagate!(grape::GrapeControl)
    grape.ρ, grape.∂ρ_∂x = propagate(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial,
                                                grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, 
                                                grape.control_coefficients, grape.times )
end

"""
    propagate_analytical()
"""
function propagate_analytical(H0::Matrix{T}, ∂H_∂x::Vector{Matrix{T}},  ρ_initial::Matrix{T}, Liouville_operator::Vector{Matrix{T}},
    γ, control_Hamiltonian::Vector{Matrix{T}}, control_coefficients::Vector{Vector{R}}, times) where {T <: Complex,R <: Real}

    dim = size(H0)[1]
    tnum = length(times)
    para_num = length(∂H_∂x)
    ctrl_num = length(control_Hamiltonian)
    Δt = times[2] - times[1]

    H = Htot(H0, control_Hamiltonian, control_coefficients)

    ρt = [Vector{ComplexF64}(undef, dim^2)  for i in 1:tnum]
    ∂ρt_∂x = [[Vector{ComplexF64}(undef, dim^2) for para in 1:para_num] for i in 1:tnum]
    δρt_δV = [[] for ctrl in 1:ctrl_num]
    ∂xδρt_δV = [[[] for ctrl in 1:ctrl_num] for i in 1:para_num]
    ∂H_L = [Matrix{ComplexF64}(undef, dim^2,dim^2)  for i in 1:para_num]
    Hc_L = [Matrix{ComplexF64}(undef, dim^2,dim^2)  for i in 1:ctrl_num]

    ρt[1] =  evolute(H[1], Liouville_operator, γ, times, times[1]) * (ρ_initial |> vec)
    for pi in 1:para_num
        ∂ρt_∂x[1][pi] = -im * Δt * ∂H_L[pi] * (ρ_initial |> vec)
        ∂H_L[pi] = liouville_commu(∂H_∂x[pi])
        for ci in 1:ctrl_num
            append!(δρt_δV[ci], [-im*Δt*Hc_L[ci]*ρt[1]])
            append!(∂xδρt_δV[pi][ci], [-im*Δt*Hc_L[ci]*∂ρt_∂x[1][pi]])
        end 
    end

    for cj in 1:ctrl_num
        Hc_L[cj] = liouville_commu(control_Hamiltonian[cj])
    end

    for ti in 2:tnum
        expL = evolute(H[ti], Liouville_operator, γ, times, times[ti])
        ρt[ti] =  expL * ρt[ti-1]
        for pk in 1:para_num
            ∂ρt_∂x[ti][pk] = -im * Δt * ∂H_L[pk] * ρt[ti] + expL * ∂ρt_∂x[ti-1][pk]
            for ck in 1:ctrl_num
                for tk in 1:ti
                    δρt_δV_first = popfirst!(δρt_δV[ck])
                    ∂xδρt_δV_first = popfirst!(∂xδρt_δV[pk][ck])
                    δρt_δV_tp = expL * δρt_δV_first
                    ∂xδρt_δV_tp = -im * Δt * ∂H_L[pk] * expL * δρt_δV_first + expL * ∂xδρt_δV_first
                    append!(δρt_δV[ck], [δρt_δV_tp])
                    append!(∂xδρt_δV[pk][ck], [∂xδρt_δV_tp])
                end
                δρt_δV_last = -im * Δt * Hc_L[ck] * ρt[ti]
                ∂xδρt_δV_last = -im * Δt * Hc_L[ck] * ∂ρt_∂x[ti][pk]
                append!(δρt_δV[ck], [δρt_δV_last])
                append!(∂xδρt_δV[pk][ck], [∂xδρt_δV_last])
                end
            end
        end

        ρt_T = ρt[end] |> vec2mat
        ∂ρt_T = [(∂ρt_∂x[end][para] |> vec2mat) for para in 1:para_num]
        F_T = QFIM(ρt_T, ∂ρt_T)

        if para_num == 1
            Lx = SLD_eig(ρt_T, ∂ρt_T[1])
            anti_commu = 2*Lx[1]*Lx[1]
            δF = [[0.0 for i in 1:tnum] for ctrl in 1:ctrl_num]
            for tm in 1:tnum
                for cm in 1:ctrl_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[1][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV*Lx[1])
                    term2 = tr(∂ρt_T_δV*anti_commu)
                    δF[cm][tm] = ((2*term1-0.5*term2) |> real)
                end
            end

        elseif para_num == 2
            F_det = F_T[1,1] * F_T[2,2] - F_T[1,2] * F_T[2,1]
            δF = [[0.0 for i in 1:tnum] for ctrl in 1:ctrl_num]
            for tm in 1:tnum
                for cm in 1:ctrl_num
                    for pm in 1:para_num
                        ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                        ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                        term1 = tr(∂xδρt_T_δV * Lx[pm])
                        anti_commu = 2 * Lx[pm] * Lx[pm]
                        term2 = tr(∂ρt_T_δV * anti_commu)
                        δF[cm][tm] = δF[cm][tm] - ((2*term1-0.5*term2) |> real)
                    end
                    δF[cm][tm] = δF[cm][tm] / F_det
                end
            end

        else
        δF = [[0.0 for i in 1:tnum] for ctrl in 1:ctrl_num]
        for tm in 1:tnum
            for cm in 1:ctrl_num
                for pm in 1:para_num
                    ∂ρt_T_δV = δρt_δV[cm][tm] |> vec2mat
                    ∂xδρt_T_δV = ∂xδρt_δV[pm][cm][tm] |> vec2mat
                    term1 = tr(∂xδρt_T_δV * Lx[pm])
                    anti_commu = 2 * Lx[pm] * Lx[pm]
                    term2 = tr(∂ρt_T_δV * anti_commu)
                    δF[cm][tm] = δF[cm][tm] - ((2*term1-0.5*term2) |> real) / (F_T[pm][pm] * F_T[pm][pm])
                end
            end
        end
    end
    ρt |> vec2mat |> filterZeros!, ∂ρt_∂x |> vec2mat |> filterZeros!, δF
end

"""
    propagate_analitical!()
"""
function propagate_analitical!(grape::GrapeControl)
    grape.ρ, grape.∂ρ_∂x, δF = propagate_analitical(grape.freeHamiltonian, grape.Hamiltonian_derivative, grape.ρ_initial,
                                                grape.Liouville_operator, grape.γ, grape.control_Hamiltonian, 
                                                grape.control_coefficients, grape.times )
    δF
end

"""
    propagate_ODEAD!()
"""
function propagate_ODEAD!(grape::GrapeControl)
    H(p) = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, p)
    dt = grape.times[2] - grape.times[1]    
    tspan = (grape.times[1], grape.times[end])
    u0 = grape.ρ_initial
    Γ = grape.Liouville_operator
    f(u, p, t) = -im * (H(p)[t2Num(tspan[1], dt, t)] * u + u * H(p)[t2Num(tspan[1], dt, t)]) + 
                 ([grape.γ[i] * (Γ[i] * u * Γ[i]' - (Γ[i]' * Γ[i] * u + u * Γ[i]' * Γ[i] )) for i in 1:length(Γ)] |> sum)
    p = grape.control_coefficients
    prob = ODEProblem(f, u0, tspan, p, saveat=dt)
    u = solve(prob).u
    du = Zygote.jacobian(solve(remake(prob, u0=u, p), sensealg=QuadratureAdjoint()))
    u, du
end

"""
    propagate_L_ODE!()
"""
function propagate_L_ODE!(grape::GrapeControl)
    H = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients)
    Δt = grape.times[2] - grape.times[1]    
    tspan = (grape.times[1], grape.times[end])
    u0 = grape.ρ_initial |> vec
    evo(p, t) = evolute(p[t2Num(tspan[1], Δt,  t)], grape.Liouville_operator, grape.γ, grape.times, t2Num(tspan[1], Δt, t)) 
    f(u, p, t) = evo(p, t) * u
    prob = DiscreteProblem(f, u0, tspan, H,dt=Δt)
    ρt = solve(prob).u 
    ∂ρt_∂x = Vector{Vector{Vector{eltype(u0)}}}(undef, 1)
    for para in 1:length(grape.Hamiltonian_derivative)
        devo(p, t) = -1.0im * Δt * liouville_commu(grape.Hamiltonian_derivative[para]) * evo(p, t) 
        du0 = devo(H, tspan[1]) * u0
        g(du, p, t) = evo(p, t) * du + devo(p, t) * ρt[t2Num(tspan[1], Δt,  t)] 
        dprob = DiscreteProblem(g, du0, tspan, H,dt=Δt) 
        ∂ρt_∂x[para] = solve(dprob).u
    end

    grape.ρ, grape.∂ρ_∂x = ρt |> vec2mat, ∂ρt_∂x |> vec2mat
end

