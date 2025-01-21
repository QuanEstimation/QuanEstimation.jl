using QuanEstimationBase:BCB
function test_bayes()
    (; rho0, x, p, dp, H0_func, dH_func) = generate_bayes()
    M = SIC(2)
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
    pout, xout = Bayes([x], p, rho, y; M = M, savefile = false)
    pout, xout = Bayes([x], p, rho, y; M = M, savefile = true)
    pout, xout = Bayes([x], p, rho, y; M = M, estimator = "MAP", savefile = false)
    pout, xout = Bayes([x], p, rho, y; M = M, estimator = "MAP", savefile = true)

    #===============Maximum likelihood estimation===============#
    Lout, xout = MLE([x], rho, y, M = M; savefile = false)
    Lout, xout = MLE([x], rho, y, M = M; savefile = true)

    BCB([x], p, rho)
    rm("bayes.dat")
    rm("MLE.dat")
end

test_bayes()