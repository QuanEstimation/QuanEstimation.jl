"""
    optimize!(opt::StateOpt, alg::RI, obj, scheme, output)

Run the Rayleigh-iteration (RI) optimization over quantum probe states.
At each episode, computes the QFIM with the SLD, constructs the dual-map matrix, and selects the
dominant eigenvector as the new probe state.
raw"""
function optimize!(opt::StateOpt, alg::RI, obj, scheme, output)
    (; max_episode) = alg

    rho, drho = evolve(scheme)
    f = QFIM(rho, drho)

    set_f!(output, f[1, 1])
    set_buffer!(output, state_data(scheme))
    set_io!(output, f[1, 1])
    show(opt, output, obj, alg)

    idx = 0
    ## single-parameter scenario
    for ei = 1:(max_episode-1)
        rho, drho = evolve(scheme)
        f, LD = QFIM(rho, drho, exportLD = true)
        M1 = d_DualMap(LD[1], param_data(scheme).K, param_data(scheme).dK)
        M2 = DualMap(LD[1] * LD[1], param_data(scheme).K)
        M = 2 * M1[1] - M2
        value, vec = eigen(M)
        val, idx = findmax(real(value))
        psi0 = vec[:, idx]
        scheme.StatePreparation.data = psi0

        set_f!(output, f[1, 1])
        set_buffer!(output, state_data(scheme))
        set_io!(output, f[1, 1], ei)
        show(output, obj)
    end
    set_io!(output, f[1, 1])
end

raw"""
    DualMap(L, K)

Compute the dual map ``\sum_i K_i^\dagger L K_i`` for a given operator ``L`` and a set of Kraus operators ``K``.
raw"""
function DualMap(L, K)
    return [Ki' * L * Ki for Ki in K] |> sum
end

raw"""
    d_DualMap(L, K, dK)

Compute the parameter derivatives of the dual map:
``\Lambda_j = \sum_i (\partial_j K_i^\dagger L K_i + K_i^\dagger L \partial_j K_i)``
for each parameter index ``j``.
"""
function d_DualMap(L, K, dK)
    K_num = length(K)
    para_num = length(dK[1])
    Lambda = [
        [dK[i][j]' * L * K[i] + K[i]' * L * dK[i][j] for i = 1:K_num] |> sum for
        j = 1:para_num
    ]
    return Lambda
end
