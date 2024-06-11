using QuanEstimationBase
using Random

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
decay = [[sp, 0.], [sm, 0.1]]
# measurement
M1 = 0.5*[1.0+0.0im  1.; 1.  1.]
M2 = 0.5*[1.0+0.0im -1.; -1.  1.]
M = [M1, M2]
# time length for the evolution
tspan = range(0., 10., length=100)
# guessed control coefficients
cnum = length(tspan)-1
ctrl = [zeros(cnum) for _ in eachindex(Hc)]
ctrl_bound = [-2., 2.]
# choose the optimization type
opt = QuanEstimationBase.ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)

# test single parameter noisy controled
dyn_sg_dm_decay = QuanEstimationBase.Lindblad(opt, tspan, rho0, H0, dH, Hc, decay, dyn_method=:Expm)

@test dyn_sg_dm_decay.para_type == :single_para 
@test dyn_sg_dm_decay.noise_type == :noisy
@test dyn_sg_dm_decay.ctrl_type == :controlled
@test dyn_sg_dm_decay.state_rep == :dm
# @test isCtrl(dyn_sg_dm_decay)
# @test isNoisy(dyn_sg_dm_decay)

