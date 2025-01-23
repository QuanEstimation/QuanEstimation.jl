function evaluate_kraus(scheme::Scheme{S,Kraus{KT,DKT,NK,NP},M,E}) where {S,KT,DKT,NK,NP,M,E}
    (; K, dK) = param_data(scheme)
    return K, dK
end

function evaluate_kraus(
    scheme::Scheme{S,Kraus{F1,F2,NK,NP},M,E},
) where {S,M,E,F1<:Function,F2<:Function,NK,NP}
    (; K, dK) = param_data(scheme)
    params = scheme.Parameterization.params
    KM = K(params...)
    dKM = dK(params...)
    return KM, dKM
end

#### evolution of pure states under time-independent Hamiltonian without noise and controls ####
"""

    evolve(dynamics::Kraus{ket})

Evolution of pure states under time-independent Hamiltonian without noise and controls
"""
function evolve(scheme::Scheme{Ket,Kraus{KT,DKT,NK,NP},M,E}) where {KT,DKT,NK,NP,M,E}
    ψ0 = state_data(scheme)
    K, dK = evaluate_kraus(scheme)
    ρ0 = ψ0 * ψ0'

    K_num = length(K)
    para_num = length(dK[1])
    ρ = [K[i] * ρ0 * K[i]' for i = 1:K_num] |> sum
    dρ = [
        [dK[i][j] * ρ0 * K[i]' + K[i] * ρ0 * dK[i][j]' for i = 1:K_num] |> sum for
        j = 1:para_num
    ]

    ρ, dρ
end

#### evolution of density matrix under time-independent Hamiltonian without noise and controls ####
"""

    evolve(dynamics::Kraus{dm})

Evolution of density matrix under time-independent Hamiltonian without noise and controls.
"""
function evolve(scheme::Scheme{DensityMatrix,Kraus{KT,DKT,NK,NP},M,E}) where {KT,DKT,NK,NP,M,E}
    ρ0 = state_data(scheme)
    K, dK = evaluate_kraus(scheme)
    K_num = length(K)
    para_num = length(dK[1])
    ρ = [K[i] * ρ0 * K[i]' for i = 1:K_num] |> sum
    dρ = [
        [dK[i][j] * ρ0 * K[i]' + K[i] * ρ0 * dK[i][j]' for i = 1:K_num] |> sum for
        j = 1:para_num
    ]

    ρ, dρ
end
