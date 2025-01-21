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

function DualMap(L, K)
    return [Ki' * L * Ki for Ki in K] |> sum
end

function d_DualMap(L, K, dK)
    K_num = length(K)
    para_num = length(dK[1])
    Lambda = [
        [dK[i][j]' * L * K[i] + K[i]' * L * dK[i][j] for i = 1:K_num] |> sum for
        j = 1:para_num
    ]
    return Lambda
end
