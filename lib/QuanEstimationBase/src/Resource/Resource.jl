using SparseArrays

function J₊(j::Number)
    spdiagm(1 => [sqrt(j * (j + 1) - m * (m + 1)) for m = j:-1:-j][2:end])
end

function Jp_full(N)
    sp = [0.0 1.0; 0.0 0.0]
    Jp, jp_tp = zeros(2^N, 2^N), zeros(2^N, 2^N)
    for i = 0:N-1
        if i == 0
            jp_tp = kron(sp, Matrix(I, 2^(N - 1), 2^(N - 1)))
        elseif i == N - 1
            jp_tp = kron(Matrix(I, 2^(N - 1), 2^(N - 1)), sp)
        else
            jp_tp =
                kron(Matrix(I, 2^i, 2^i), kron(sp, Matrix(I, 2^(N - 1 - i), 2^(N - 1 - i))))
        end
        Jp += jp_tp
    end
    return Jp
end

function Jz_full(N)
    sz = [1.0 0.0; 0.0 -1.0]
    Jz, jz_tp = zeros(2^N, 2^N), zeros(2^N, 2^N)
    for i = 0:N-1
        if i == 0
            jz_tp = kron(sz, Matrix(I, 2^(N - 1), 2^(N - 1)))
        elseif i == N - 1
            jz_tp = kron(Matrix(I, 2^(N - 1), 2^(N - 1)), sz)
        else
            jz_tp =
                kron(Matrix(I, 2^i, 2^i), kron(sz, Matrix(I, 2^(N - 1 - i), 2^(N - 1 - i))))
        end
        Jz += jz_tp
    end
    return 0.5 * Jz
end

"""

    SpinSqueezing(ρ::AbstractMatrix; basis="Dicke", output="KU")
    
Calculate the spin squeezing parameter for the input density matrix. The `basis` can be `"Dicke"` for the Dicke basis, or `"Pauli"` for the Pauli basis. The `output` can be both `"KU"`(for spin squeezing defined by Kitagawa and Ueda) and `"WBIMH"`(for spin squeezing defined by Wineland et al.).

"""
function SpinSqueezing(ρ::AbstractMatrix; basis = "Dicke", output = "KU")

    if basis == "Pauli"
        # For Pauli basis, the density matrix size should be 2^N
        N = Int(log2(size(ρ, 1)))
        j = N / 2
        Jp = Jp_full(N)
        Jz = Jz_full(N)
        # Precompute Jx and Jy for Pauli basis
        Jx = 0.5 * (Jp + Jp')
        Jy = -0.5im * (Jp - Jp')
    elseif basis == "Dicke"
        j = (size(ρ, 1) - 1) / 2
        N = 2 * j
        Jp = J₊(j)
        Jz = spdiagm(j:-1:-j)
        # Precompute Jx and Jy for Dicke basis
        Jx = 0.5 * (Jp + Jp')
        Jy = -0.5im * (Jp - Jp')
    else
        throw(ErrorException("Invalid basis type. Valid options are: Dicke, Pauli"))
    end

    coef = 4.0 / N
        
    Jx_mean = tr(ρ * Jx) |> real
    Jy_mean = tr(ρ * Jy) |> real
    Jz_mean = tr(ρ * Jz) |> real

    if Jx_mean == 0 && Jy_mean == 0
        if Jz_mean == 0
            throw(ErrorException("The density matrix does not have a valid spin squeezing."))
        else
            A = tr(ρ * (Jx * Jx - Jy * Jy))
            B = tr(ρ * (Jx * Jy + Jy * Jx))
            C = tr(ρ * (Jx * Jx + Jy * Jy))
        end
    else    
        cosθ = Jz_mean / sqrt(Jx_mean^2 + Jy_mean^2 + Jz_mean^2)
        sinθ = sin(acos(cosθ))
        cosϕ = Jx_mean / sqrt(Jx_mean^2 + Jy_mean^2)
        sinϕ = Jy_mean > 0 ? sin(acos(cosϕ)) : sin(2pi - acos(cosϕ))

        Jn1 = -Jx * sinϕ + Jy * cosϕ
        Jn2 = -Jx * cosθ * cosϕ - Jy * cosθ * sinϕ + Jz * sinθ
        A = tr(ρ * (Jn1 * Jn1 - Jn2 * Jn2))
        B = tr(ρ * (Jn1 * Jn2 + Jn2 * Jn1))
        C = tr(ρ * (Jn1 * Jn1 + Jn2 * Jn2))
    end

    V₋ = 0.5 * (C - sqrt(A^2 + B^2)) |> real
    ξ = coef * V₋
    ξ = ξ > 1 ? 1.0 : ξ

    if output == "KU"
        return ξ
    elseif output == "WBIMH"
        return (N / 2)^2 * ξ / (Jx_mean^2 + Jy_mean^2 + Jz_mean^2)
    else
        throw(ErrorException("Invalid output type. Valid options are: KU, WBIMH"))
    end
end

"""
    TargetTime(f::Number, tspan::AbstractVector, func::Function, args...; kwargs...)

Calculate the minimum time to reach a precision limit of given level. The `func` can be any objective function during the control optimization, e.g. QFIM, CFIM, HCRB, etc.

"""
function TargetTime(f::Number, tspan::AbstractVector, func::Function, args...; kwargs...)
    
    # Find the first index where func(t) crosses the target f
    for i in eachindex(tspan)
        f_val = func(tspan[i], args...; kwargs...)
        
        if i == 1
            # Check if we're already at the target at the first time point
            if f_val ≈ f
                return tspan[1]
            end
        else
            # Get the previous value
            f_prev = func(tspan[i-1], args...; kwargs...)
            
            # Check if we've crossed the target
            if (f_prev ≤ f && f_val ≥ f) || (f_prev ≥ f && f_val ≤ f)
                # Linear interpolation for more accurate crossing time
                t1 = tspan[i-1]
                t2 = tspan[i]
                y1 = f_prev
                y2 = f_val
                
                # Avoid division by zero
                if y2 ≈ y1
                    return t1
                end
                
                # Calculate the crossing time
                t_cross = t1 + (f - y1) * (t2 - t1) / (y2 - y1)
                return t_cross
            end
        end
    end
    
    # If we never cross the target, return the last time point
    println("No time is found in the given time span to reach the target.")
    return nothing
end
