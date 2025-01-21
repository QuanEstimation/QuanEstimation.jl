function test_bayes()
    function H0_func(x)
        return 0.5 * pi/2 * (σx() * cos(x) + σz() * sin(x))
    end
    function dH_func(x)
        return [0.5 * pi/2 * (-σx() * sin(x) + σz() * cos(x))]
    end

    rho0 = 0.5 * ones(2, 2)
    x = range(0.0, stop = 0.5 * pi, length = 100) |> Vector
    p = (1.0 / (x[end] - x[1])) * ones(length(x))
    tspan = range(0.0, stop = 1.0, length = 1000)
    rho = Vector{Matrix{ComplexF64}}(undef, length(x))
    for i in eachindex(x)
        H0_tp = H0_func(x[i])
        dH_tp = dH_func(x[i])
        rho_tp, _ = expm(tspan, rho0, H0_tp, dH_tp)
        rho[i] = rho_tp[end]
    end

    # Generation of the experimental results
    Random.seed!(1234)
    y = [rand() > 0.7 ? 1 : 0 for _ = 1:500]

    #===============Maximum a posteriori estimation===============#
    pout, xout = Bayes([x], p, rho, y; M = M, estimator = "MAP", savefile = false)

    #===============Maximum likelihood estimation===============#
    Lout, xout = MLE([x], rho, y, M = M; savefile = false)

    rm("bayes.dat")
    rm("MLE.dat")
end

test_bayes()