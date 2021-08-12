abstract type ControlSystem end
mutable struct GrapeControl{T <: Complex,M <: Real} <: ControlSystem
    freeHamiltonian::Matrix{T}
    Hamiltonian_derivative::Vector{Matrix{T}}
    ρ_initial::Matrix{T}
    times::StepRangeLen{M,Base.TwicePrecision{M},Base.TwicePrecision{M}}
    Liouville_operator::Vector{Matrix{T}}
    γ::Vector{M}
    control_Hamiltonian::Vector{Matrix{T}}
    control_coefficients::Vector{Vector{M}}
    ϵ::M
    ρ::Vector{Matrix{T}}
    ∂ρ_∂x::Vector{Vector{Matrix{T}}}
    GrapeControl(freeHamiltonian::Matrix{T}, Hamiltonian_derivative::Vector{Matrix{T}}, ρ_initial::Matrix{T},
                 times::StepRangeLen{M,Base.TwicePrecision{M},Base.TwicePrecision{M}},
                 Liouville_operator::Vector{Matrix{T}},γ::Vector{M}, control_Hamiltonian::Vector{Matrix{T}},
                 control_coefficients::Vector{Vector{M}}, ϵ=0.01, ρ=Vector{Matrix{T}}(undef, 1), 
                 ∂ρ_∂x=Vector{Vector{Matrix{T}}}(undef, 1),∂ρ_∂V=Vector{Vector{Matrix{T}}}(undef, 1)) where {T <: Complex,M <: Real} = 
                 new{T,M}(freeHamiltonian, Hamiltonian_derivative, ρ_initial, times, Liouville_operator, γ, control_Hamiltonian,
                          control_coefficients, ϵ, ρ, ∂ρ_∂x) 
end

function gradient_CFI!(grape::GrapeControl{T}, M) where {T <: Complex}
    δI = gradient(x->CFI(M, Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δI
end
function gradient_CFIM!(grape::GrapeControl{T}, M) where {T <: Complex}
    δI = gradient(x->CFIM(M, Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients).|>real
    grape.control_coefficients += grape.ϵ*δI
end
function gradient_CFI_ADAM!(grape::GrapeControl{T}, M) where {T <: Complex}
    δI = gradient(x->CFI(M, Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients)[1].|>real
    Adam!(grape, δI)
end
function gradient_CFIM_ADAM!(grape::GrapeControl{T}, M, mt, vt) where {T <: Complex}
    δI = gradient(x->CFIM(M, Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients).|>real
    Adam!(grape, δI)
end

function gradient_QFI!(grape::GrapeControl{T}) where {T <: Complex}
    δF = gradient(x->QFI(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients)[1].|>real
    grape.control_coefficients += grape.ϵ*δF
end
function gradient_QFIM!(grape::GrapeControl{T}) where {T <: Complex}
    δF = gradient(x->1/(QFIM(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times) |> pinv |> tr |>real), grape.control_coefficients).|>real |>sum
    grape.control_coefficients += grape.ϵ*δF
end
function gradient_QFI_ADAM!(grape::GrapeControl{T}) where {T <: Complex}
    δF = gradient(x->QFI(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative[1], grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times), grape.control_coefficients)[1].|>real
    Adam!(grape, δF)
end
function gradient_QFIM_ADAM!(grape::GrapeControl{T}) where {T <: Complex}
    δF = gradient(x->1/(QFIM(Htot(grape.freeHamiltonian, grape.control_Hamiltonian, x), grape.Hamiltonian_derivative, grape.ρ_initial, grape.Liouville_operator, grape.γ, grape.times) |> pinv |> tr |>real), grape.control_coefficients).|>real |>sum
    Adam!(grape, δF)
end
function gradient_QFI_analitical_ADAM!(grape::GrapeControl{T}) where {T <: Complex}
    δF = propagate_analitical!(grape)
    Adam!(grape, δF)
end
function gradient_QFIM_ODE(grape::GrapeControl)
    H = Htot(grape.freeHamiltonian, grape.control_Hamiltonian, grape.control_coefficients)
    Δt = grape.times[2] - grape.times[1]
    t_num = length(grape.times)
    para_num = length(grape.Hamiltonian_derivative)    
    ctrl_num = length(grape.control_Hamiltonian)
    tspan(j) = (grape.times[1], grape.times[j])
    tspan() = (grape.times[1], grape.times[end])
    u0 = grape.ρ_initial |> vec
    evo(p, t) = evolute(p[t2Num(tspan()[1], Δt,  t)], grape.Liouville_operator, grape.γ, grape.times, t2Num(tspan()[1], Δt, t)) 
    f(u, p, t) = evo(p, t) * u
    prob = DiscreteProblem(f, u0, tspan(), H,dt=Δt)
    ρt = solve(prob).u 
    ∂ρt_∂x = Vector{Vector{Vector{eltype(u0)}}}(undef, 1)
    for para in 1:para_num
        devo(p, t) = -1.0im * Δt * liouville_commu(grape.Hamiltonian_derivative[para]) * evo(p, t) 
        du0 = devo(H, tspan()[1]) * u0
        g(du, p, t) = evo(p, t) * du + devo(p, t) * ρt[t2Num(tspan()[1], Δt,  t)] 
        dprob = DiscreteProblem(g, du0, tspan(), H,dt=Δt) 
        ∂ρt_∂x[para] = solve(dprob).u
    end
    δρt_δV = Matrix{Vector{Vector{eltype(u0)}}}(undef,ctrl_num,length(grape.times))
    for ctrl in 1:ctrl_num
        for j in 1:t_num
            devo(p, t) = -1.0im * Δt * liouville_commu(grape.control_Hamiltonian[ctrl]) * evo(p, t) 
            du0 = devo(H, tspan()[1]) * u0
            g(du, p, t) = evo(p, t) * du + devo(p, t) * ρt[t2Num(tspan()[1], Δt,  t)] 
            dprob = DiscreteProblem(g, du0, tspan(j), H,dt=Δt) 
            δρt_δV[ctrl,j] = solve(dprob).u
        end
    end
    ∂xδρt_δV = Array{Vector{eltype(u0)}, 3}(undef,para_num, ctrl_num,length(grape.times))
    for para in 1:para_num
        for ctrl in 1:ctrl_num
            dxevo = -1.0im * Δt * liouville_commu(grape.Hamiltonian_derivative[para]) 
            dkevo = -1.0im * Δt * liouville_commu(grape.control_Hamiltonian[ctrl])
            for j in 1:t_num
                g(du, p, t) = dxevo * dkevo  * evo(p, t) * ρt[t2Num(tspan()[1], Δt,  t)] +
                              dxevo * evo(p, t) * δρt_δV[ctrl, j][t2Num(tspan()[1], Δt,  t)] +
                              dkevo * evo(p, t) * ∂ρt_∂x[para][t2Num(tspan()[1], Δt,  t)] + 
                              evo(p, t) * du
                du0 = dxevo * dkevo  * evo(H,tspan()[1]) * ρt[t2Num(tspan()[1], Δt, tspan()[1])]
                dprob = DiscreteProblem(g, du0, tspan(j), H, dt=Δt)
                ∂xδρt_δV[para, ctrl, j] = solve(dprob).u[end]
            end
        end
    end
    δF = grape.control_coefficients .|> zero
    for para in 1:para_num
        SLD_tp = SLD(ρt[end], ∂ρt_∂x[para][end])
        for ctrl in 1:ctrl_num
            for j in 1:t_num   
                δF[ctrl][j] -= 2 * tr((∂xδρt_δV[para,ctrl,j]|> vec2mat) * SLD_tp) - 
                                   tr((δρt_δV[ctrl, j][end] |> vec2mat) * SLD_tp^2) |> real
            end
        end
    end
    δF
end
function gradient_QFIM_ODE!(grape::GrapeControl{T}) where {T <: Complex}
    grape.control_coefficients += grape.ϵ * gradient_QFIM_ODE(grape)
end

function Run(M, grape::GrapeControl{T}) where {T<: Complex}
    println("classical parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        cfi_ini = CFI(M, grape)
        cfi_list = [cfi_ini]
        println("initial CFI is $(cfi_ini)")
        gradient_CFI!(M, grape)
        while true
            cfi_now = CFI(M, grape)
            gradient_CFI!(M, grape)
            if  0 < (cfi_now - cfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final CFI is ", cfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "cfi", cfi_list)
                break
            else
                cfi_ini = cfi_now
                append!(cfi_list,cfi_now)
                print("current CFI is ", cfi_now, " ($(cfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> CFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_CFIM!(M, grape)
        while true
            f_now = 1/(grape |> CFIM |> inv |> tr)
            gradient_CFIM!(M, grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(I^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(I^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end
function RunADAM(grape)
    println("AutoGrape strategies")
    println("quantum parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        qfi_ini = QFI(grape)
        qfi_list = [qfi_ini]
        println("initial QFI is $(qfi_ini)")
        gradient_QFI_ADAM!(grape)
        while true
            qfi_now = QFI(grape)
            gradient_QFI_ADAM!(grape)
            if  0 < (qfi_now - qfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final QFI is ", qfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "qfi", qfi_list)
                break
            else
                qfi_ini = qfi_now
                append!(qfi_list,qfi_now)
                print("current QFI is ", qfi_now, " ($(qfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> QFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_QFIM_ADAM!(grape)
        while true
            f_now = 1/(grape |> QFIM |> inv |> tr)
            gradient_QFIM_ADAM!(grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(F^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(F^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end
function RunAnaliticalADAM(grape)
    println("Analitical strategies")
    println("quantum parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        qfi_ini = QFI(grape)
        qfi_list = [qfi_ini]
        println("initial QFI is $(qfi_ini)")
        gradient_QFI_analitical_ADAM!(grape)
        while true
            qfi_now = QFI(grape)
            gradient_QFI_analitical_ADAM!(grape)
            if  0 < (qfi_now - qfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final QFI is ", qfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "qfi", qfi_list)
                break
            else
                qfi_ini = qfi_now
                append!(qfi_list,qfi_now)
                print("current QFI is ", qfi_now, " ($(qfi_list|>length) epochs)    \r")
            end
        end
    end
end
function RunADAM(M, grape::GrapeControl{T}) where {T<: Complex}
    println("classical parameter estimation")
    if length(grape.Hamiltonian_derivative) == 1
        println("single parameter estimation scenario")
        cfi_ini = CFI(M, grape)
        cfi_list = [cfi_ini]
        println("initial CFI is $(cfi_ini)")
        gradient_CFI_ADAM!(M, grape)
        while true
            cfi_now = CFI(M, grape)
            gradient_CFI_ADAM!(M, grape)
            if  0 < (cfi_now - cfi_ini) < 1e-4
                println("\n Iteration over, data saved.")
                println("Final CFI is ", cfi_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "cfi", cfi_list)
                break
            else
                cfi_ini = cfi_now
                append!(cfi_list,cfi_now)
                print("current CFI is ", cfi_now, " ($(cfi_list|>length) epochs)    \r")
            end
        end
    else
        println("multiple parameters estimation scenario")
        f_ini =1/(grape |> CFIM |> inv |> tr)
        f_list = [f_ini]
        println("initial 1/tr(F^-1) is $(f_ini)")
        gradient_CFIM!(M, grape)
        while true
            f_now = 1/(grape |> CFIM |> inv |> tr)
            gradient_CFIM!(M, grape)
            if  0< f_now - f_ini < 1e-4
                println("\n Iteration over, data saved.")
                println("Final 1/tr(I^-1) is ", f_now)
                save("controls.jld", "controls", grape.control_coefficients, "time_span", grape.times, "f", f_list)
                break
            else
                f_ini = f_now
                append!(f_list,f_now)
                print("current 1/tr(I^-1) is ", f_now, " ($(f_list|>length) epochs)    \r")
            end
        end
    end
end