using QuanEstimation
using Test
using Trapz


using StableRNGs
using LinearAlgebra
using SparseArrays

# CramerRao_bounds
@testset "CramerRao_bounds" begin
    # initial state
    rho0 = 0.5*ones(2, 2)
    # free Hamiltonian
    omega = 1.0
    sx = [0. 1.; 1. 0.0im]
    sy = [0. -im; im 0.]
    sz = [1. 0.0im; 0. -1.]
    H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5*sz]
    # dissipation
    sp = [0. 1.; 0. 0.0im]
    sm = [0. 0.; 1. 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    # measurement
    M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
    M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
    M = [M1, M2]
    # time length for the evolution
    tspan = range(0., 50., length=2000)
    # dynamics
    rho, drho = QuanEstimation.expm(tspan, rho0, H0, dH, decay)
    # calculation of the CFI and QFI
    Im, F = Float64[], Float64[]
    for ti in 2:length(tspan)
        # CFI
        I_tp = QuanEstimation.CFIM(rho[ti], drho[ti], M)
        append!(Im, I_tp)
        # QFI
        F_tp = QuanEstimation.QFIM(rho[ti], drho[ti])
        append!(F, F_tp)
    end
#    CramerRao_bounds_test = CSV.read("test/CramerRao_bounds_test.csv", DataFrame)
    @test Im[length(Im)] == 1.1669316257550693
    @test F[length(F)] == 16.844867497710958
end

# Bayesian_CramerRao_bounds
@testset "Bayesian_CramerRao_bounds" begin
    function H0_func(x)
        return 0.5*B*omega0*(sx*cos(x)+sz*sin(x))
    end
    # derivative of the free Hamiltonian on x
    function dH_func(x)
        return [0.5*B*omega0*(-sx*sin(x)+sz*cos(x))]
    end
    # prior distribution
    function p_func(x, mu, eta)
        return exp(-(x-mu)^2/(2*eta^2))/(eta*sqrt(2*pi))
    end
    function dp_func(x, mu, eta)
        return -(x-mu)*exp(-(x-mu)^2/(2*eta^2))/(eta^3*sqrt(2*pi))
    end
    
    B, omega0 = 0.5*pi, 1.0
    sx = [0. 1.; 1. 0.0im]
    sy = [0. -im; im 0.]
    sz = [1. 0.0im; 0. -1.]
    # initial state
    rho0 = 0.5*ones(2, 2)
    # prior distribution
    x = range(-0.5*pi, stop=0.5*pi, length=100) |>Vector
    mu, eta = 0.0, 0.2
    p_tp = [p_func(x[i], mu, eta) for i = eachindex(x)]
    dp_tp = [dp_func(x[i], mu, eta) for i = eachindex(x)]
    # normalization of the distribution
    c = trapz(x, p_tp)
    p = p_tp/c
    dp = dp_tp/c
    # time length for the evolution
    tspan = range(0., stop=1., length=1000)
    # dynamics
    rho = Vector{Matrix{ComplexF64}}(undef, length(x))
    drho = Vector{Vector{Matrix{ComplexF64}}}(undef, length(x))
    for i = eachindex(x)
        H0_tp = H0_func(x[i])
        dH_tp = dH_func(x[i])
        rho_tp, drho_tp = QuanEstimation.expm(tspan, rho0, H0_tp, dH_tp)
        rho[i], drho[i] = rho_tp[end], drho_tp[end]
    end
    
    # Classical Bayesian bounds
    f_BCRB1 = QuanEstimation.BCRB([x], p, [], rho, drho, btype=1)
    @test f_BCRB1 == 0.6544568264060856
    f_BCRB2 = QuanEstimation.BCRB([x], p, [], rho, drho, btype=2)
    @test f_BCRB2 == 0.6516067755941157
    f_BCRB3 = QuanEstimation.BCRB([x], p, dp, rho, drho, btype=3)
    @test f_BCRB3 == 0.16521231566690223
    f_VTB = QuanEstimation.VTB([x], p, dp, rho, drho)
    @test f_VTB == 0.03768654666718875
    # Quantum Bayesian bounds
    f_BQCRB1 = QuanEstimation.BQCRB([x], p, [], rho, drho, btype=1)
    @test f_BQCRB1 == 0.5101715332365327
    f_BQCRB2 = QuanEstimation.BQCRB([x], p, [], rho, drho, btype=2)
    @test f_BQCRB2 == 0.5097829847502268
    f_BQCRB3 = QuanEstimation.BQCRB([x], p, dp, rho, drho, btype=3)
    @test f_BQCRB3 == 0.14348129346309374
    f_QVTB = QuanEstimation.QVTB([x], p, dp, rho, drho)
    @test f_QVTB == 0.03708976078857451
    f_QZZB = QuanEstimation.QZZB([x], p, rho)
    @test f_QZZB == 0.02845560767593649
end

# CMopt

@testset "CMopt" begin    
    # initial state
    rho0 = 0.5*ones(2, 2)
    # free Hamiltonian
    omega = 1.0
    sx = [0. 1.; 1. 0.0im]
    sy = [0. -im; im 0.]
    sz = [1. 0.0im; 0. -1.]
    H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5*sz]
    # control Hamiltonians 
    Hc = [sx, sy, sz]
    # dissipation
    sp = [0. 1.; 0. 0.0im]
    sm = [0. 0.; 1. 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    # measurement
    M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
    M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
    M = [M1, M2]
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    # control and measurement optimization
    opt = QuanEstimation.CMopt(ctrl_bound=[-2.0,2.0], seed=1234)

    ##==========choose comprehensive optimization algorithm==========##
    ##-------------algorithm: DE---------------------##
    alg = QuanEstimation.DE(p_num=10, max_episode=10, c=1.0, cr=0.5)
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc, 
                                    decay=decay, dyn_method=:Expm)   
    # objective function: CFI
    obj = QuanEstimation.CFIM_obj() 
    # run the comprehensive optimization problem
    @test nothing === QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)

    ##-------------algorithm: PSO---------------------##
    # alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
    #                          c1=2.0, c2=2.0)
    # # input the dynamics data
    # dynamics = QuanEstimation.Lindblad(opt, tspan, rho0, H0, dH, Hc, 
    #                                    decay=decay, dyn_method=:Expm)   
    # # objective function: CFI
    # obj = QuanEstimation.CFIM_obj() 
    # # run the comprehensive optimization problem
    # QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
end

# SCMopt
@testset "SCMopt" begin
    # initial state
    rho0 = 0.5*ones(2, 2)
    # free Hamiltonian
    omega = 1.0
    sx = [0. 1.; 1. 0.0im]
    sy = [0. -im; im 0.]
    sz = [1. 0.0im; 0. -1.]
    H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5*sz]
    # control Hamiltonians 
    Hc = [sx, sy, sz]
    # dissipation
    sp = [0. 1.; 0. 0.0im]
    sm = [0. 0.; 1. 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    # measurement
    M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
    M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
    M = [M1, M2]
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    # choose the optimization type
    opt = QuanEstimation.SCMopt(ctrl_bound=[-2.0,2.0], seed=1234)

    ##==========choose comprehensive optimization algorithm==========##
    ##-------------algorithm: DE---------------------##
    alg = QuanEstimation.DE(p_num=10, max_episode=20, c=1.0, cr=0.5)
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, dyn_method=:Expm)   
    # objective function: CFI
    obj = QuanEstimation.CFIM_obj() 
    # run the comprehensive optimization problem
    @test nothing === QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
end

@testset "SCopt" begin    
    # initial state
    rho0 = 0.5*ones(2, 2)
    # free Hamiltonian
    omega = 1.0
    sx = [0. 1.; 1. 0.0im]
    sy = [0. -im; im 0.]
    sz = [1. 0.0im; 0. -1.]
    H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5*sz]
    # control Hamiltonians 
    Hc = [sx, sy, sz]
    # dissipation
    sp = [0. 1.; 0. 0.0im]
    sm = [0. 0.; 1. 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    # measurement
    M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
    M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
    M = [M1, M2]
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    # choose the optimization type
    opt = QuanEstimation.SCopt(ctrl_bound=[-2.0,2.0], seed=1234)

    ##==========choose comprehensive optimization algorithm==========##
    ##-------------algorithm: DE---------------------##
    alg = QuanEstimation.DE(p_num=10, max_episode=10, c=1.0, cr=0.5)
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, dyn_method=:Expm) 

    ##-------------algorithm: PSO---------------------##
    # alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
    #                          c1=2.0, c2=2.0)
    # # input the dynamics data
    # dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, dyn_method=:Expm) 

    ##-------------algorithm: AD---------------------##
    # alg = QuanEstimation.AD(Adam=true, max_episode=300, epsilon=0.01, 
    #                         beta1=0.90, beta2=0.99)
    # # input the dynamics data
    # dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, Hc, decay=decay, dyn_method=:Expm) 

    ##===================choose objective function===================##
    ##-------------objective function: QFI---------------------##
    obj = QuanEstimation.QFIM_obj()
    # run the comprehensive optimization problem
    @test nothing === QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
end

# SMopt
@testset "SMopt" begin
    # initial state
    rho0 = 0.5*ones(2, 2)
    # free Hamiltonian
    omega = 1.0
    sx = [0. 1.; 1. 0.0im]
    sy = [0. -im; im 0.]
    sz = [1. 0.0im; 0. -1.]
    H0 = 0.5*omega*sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5*sz]
    # control Hamiltonians 
    Hc = [sx, sy, sz]
    # dissipation
    sp = [0. 1.; 0. 0.0im]
    sm = [0. 0.; 1. 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    # measurement
    M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
    M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
    M = [M1, M2]
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    # choose the optimization type
    opt = QuanEstimation.SMopt(seed=1234)

    ##==========choose comprehensive optimization algorithm==========##
    ##-------------algorithm: DE---------------------##
    alg = QuanEstimation.DE(p_num=10, max_episode=10, c=1.0, cr=0.5)
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, dyn_method=:Expm)   
    # objective function: CFI
    obj = QuanEstimation.CFIM_obj() 
    # run the comprehensive optimization problem
    @test nothing === QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)

    ##-------------algorithm: PSO---------------------##
    # alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
    #                          c1=2.0, c2=2.0)
    # # input the dynamics data
    # dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, dyn_method=:Expm)   
    # # objective function: CFI
    # obj = QuanEstimation.CFIM_obj() 
    # # run the comprehensive optimization problem
    # QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
end

# state_optimization_LMG
@testset "state_optimization_LMG" begin
    # dimensions of the system
    N = 8
    # generation of the coherent spin state
    j, theta, phi = N÷2, 0.5pi, 0.5pi
    Jp = Matrix(spdiagm(1=>[sqrt(j*(j+1)-m*(m+1)) for m in j:-1:-j][2:end]))
    Jm = Jp'
    psi0 = exp(0.5*theta*exp(im*phi)*Jm - 0.5*theta*exp(-im*phi)*Jp)*
        QuanEstimation.basis(Int(2*j+1), 1)
    dim = length(psi0)
    # free Hamiltonian
    lambda, g, h = 1.0, 0.5, 0.1
    Jx = 0.5*(Jp + Jm)
    Jy = -0.5im*(Jp - Jm)
    Jz = spdiagm(j:-1:-j)
    H0 = -lambda*(Jx*Jx + g*Jy*Jy) / N - h*Jz
    # derivative of the free Hamiltonian on g
    dH = [-lambda*Jy*Jy/N]
    # dissipation
    decay = [[Jz, 0.1]]
    # time length for the evolution
    tspan = range(0., 10., length=2500)
    # set the optimization type
    opt = QuanEstimation.StateOpt(psi=psi0, seed=1234) 

    ##================choose the state optimization algorithm===============##
    # state optimization algorithm: AD
    alg = QuanEstimation.AD(Adam=false, max_episode=300, epsilon=0.01, 
                            beta1=0.90, beta2=0.99)

    # # state optimization algorithm: PSO
    # alg = QuanEstimation.PSO(p_num=10, max_episode=[1000,100], c0=1.0, 
    #                          c1=2.0, c2=2.0)

    # # state optimization algorithm: DE
    # alg = QuanEstimation.DE(p_num=10, max_episode=1000, c=1.0, cr=0.5)

    # # state optimization algorithm: NM
    # alg = QuanEstimation.NM(p_num=10, max_episode=1000, ar=1.0, 
    #                         ae=2.0, ac=0.5, as0=0.5)

    # # state optimization algorithm: DDPG
    # alg = QuanEstimation.DDPG(max_episode=500, layer_num=3, layer_dim=200)

    ##====================choose the objective function==================##
    ##-------------objective function: QFI---------------------##
    # objective function: QFI
    obj = QuanEstimation.QFIM_obj()
    # input the dynamics data
    dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, dyn_method=:Expm) 
    # run the state optimization problem
    QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)

    ##-------------objective function: CFI---------------------##
    # # objective function: CFI
    # obj = QuanEstimation.CFIM_obj()
    # # input the dynamics data
    # dynamics = QuanEstimation.Lindblad(opt, tspan, H0, dH, decay=decay, dyn_method=:Expm) 
    # # run the state optimization problem
    # QuanEstimation.run(opt, alg, obj, dynamics; savefile=false)
end