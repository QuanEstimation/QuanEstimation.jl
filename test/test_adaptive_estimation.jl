function test_adaptive_estimation_MZI()
    N = 3
    # probe state
    psi =
        sum([
            sin(k * pi / (N + 2)) * kron(basis(N + 1, k), basis(N + 1, N - k + 2)) for
            k = 1:(N+1)
        ]) |> sparse
    psi = psi * sqrt(2 / (2 + N))
    rho0 = psi * psi'
    # prior distribution
    x = range(-pi, pi, length = 100)
    p = (1.0 / (x[end] - x[1])) * ones(length(x))
    apt = Adapt_MZI(x, p, rho0)

    #================online strategy=========================#
    online(apt; target=:sharpness, output="phi", res=zeros(2))

    #================offline strategy=========================#
    # algorithm: DE
    alg = DE(p_num = 3, ini_population = nothing, max_episode = 10, c = 1.0, cr = 0.5)
    offline(apt, alg, target = :sharpness, seed = 1234)
    offline(apt, alg, target = :MI, seed = 1234)

    # # algorithm: PSO
    PSO(p_num=3, ini_particle=nothing, max_episode=[10,10], c0=1.0, c1=2.0, c2=2.0)
    offline(apt, alg, target=:sharpness, seed=1234)


    rm("f.csv")
    rm("deltaphi.csv")
    rm("adaptive.dat")
    return true
end

function test_adaptive_estimation()
    scheme=generate_scheme_adaptive()

    @suppress adapt!(scheme; res=zeros(10), method="FOP", max_episode=10)
    @suppress adapt!(scheme; res=zeros(10), method="MI", max_episode=10)

    rm("adaptive.dat")
    return true
end


@test test_adaptive_estimation_MZI()
@test test_adaptive_estimation()