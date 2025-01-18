function QubitDephasing(r::Vector{Float64}, 
                        para_est::String,
                        gamma::Float64,  
                        tspan::Union{Vector{Float64}, StepRangeLen})
    H0 = r[1]*SigmaX()+r[2]*SigmaY()+r[3]*SigmaZ()
    if para_est == "x"
        dH = [SigmaX()]
    elseif para_est == "y"
        dH = [SigmaY()]
    elseif para_est == "z"
        dH = [SigmaZ()]
    end
    decay = [[SigmaZ()], [gamma]]
    return Lindblad(H0, dH, tspan, decay, dyn_method=:Ode)
end