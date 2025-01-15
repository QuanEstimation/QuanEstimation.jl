using QuanEstimationBase: ControlEnv, StateEnv, QFIM_obj
using StableRNGs, IntervalSets, Random
using ReinforcementLearning: Space

# Test for ControlEnv
function test_ControlEnv()
    # initial state
    rho0 = 0.5 * ones(2, 2)
    # free Hamiltonian
    omega = 1.0
    sx = [0.0 1.0; 1.0 0.0im]
    sy = [0.0 -im; im 0.0]
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = 0.5 * omega * sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5 * sz]
    # control Hamiltonians 
    Hc = [sx, sy, sz]
    # dissipation
    sp = [0.0 1.0; 0.0 0.0im]
    sm = [0.0 0.0; 1.0 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    # time length for the evolution
    tspan = range(0.0, 10.0, length = 2500)
    # guessed control coefficients
    cnum = length(tspan) - 1
    ctrl = [zeros(cnum) for _ in eachindex(Hc)]
    ctrl_bound = [-2.0, 2.0]

    obj = QFIM_obj()
    opt = QuanEstimationBase.ControlOpt(ctrl = ctrl, ctrl_bound = ctrl_bound, seed = 1234)
    dynamics =
        QuanEstimationBase.Lindblad(opt, tspan, rho0, H0, dH, Hc, decay, dyn_method = :Expm)
    output = QuanEstimationBase.Output(opt)
    state = rho0 |> vec
    dstate = [zero(state)]
    action_space = Space([opt.ctrl_bound[1] .. opt.ctrl_bound[2] for _ = 1:cnum])
    state_space = Space(fill(-1.0e35 .. 1.0e35, length(state)))
    done = false
    rng = MersenneTwister(1234)
    reward = 0.0
    total_reward = 0.0
    t = 0
    tspan = [0, 1, 2]
    tnum = 3
    ctrl_length = 3
    ctrl_num = 1
    para_num = 1
    f_noctrl = [0.0, 0.0, 0.0]
    f_final = [0.0]
    ctrl_list = [[]]
    ctrl_bound = [-1, 1]
    total_reward_all = [0.0]
    episode = 1

    env = ControlEnv(
        obj,
        dynamics,
        output,
        action_space,
        state_space,
        state,
        dstate,
        done,
        rng,
        reward,
        total_reward,
        t,
        tspan,
        tnum,
        ctrl_length,
        ctrl_num,
        para_num,
        f_noctrl,
        f_final,
        ctrl_list,
        ctrl_bound,
        total_reward_all,
        episode,
    )

    @test env.obj == obj
    @test env.dynamics == dynamics
    @test env.output == output
    @test env.action_space == action_space
    @test env.state_space == state_space
    @test env.state == state
    @test env.dstate == dstate
    @test env.done == done
    @test env.rng == rng
    @test env.reward == reward
    @test env.total_reward == total_reward
    @test env.t == t
    @test env.tspan == tspan
    @test env.tnum == tnum
    @test env.ctrl_length == ctrl_length
    @test env.ctrl_num == ctrl_num
    @test env.para_num == para_num
    @test env.f_noctrl == f_noctrl
    @test env.f_final == f_final
    @test env.ctrl_list == ctrl_list
    @test env.ctrl_bound == ctrl_bound
    @test env.total_reward_all == total_reward_all
    @test env.episode == episode
end

# Test for StateEnv
function test_StateEnv()
    # initial state
    rho0 = 0.5 * ones(2, 2)
    # free Hamiltonian
    omega = 1.0
    sx = [0.0 1.0; 1.0 0.0im]
    sy = [0.0 -im; im 0.0]
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = 0.5 * omega * sz
    # derivative of the free Hamiltonian on omega
    dH = [0.5 * sz]
    # control Hamiltonians 
    Hc = [sx, sy, sz]
    # dissipation
    sp = [0.0 1.0; 0.0 0.0im]
    sm = [0.0 0.0; 1.0 0.0im]
    decay = [[sp, 0.0], [sm, 0.1]]
    # time length for the evolution
    tspan = range(0.0, 10.0, length = 10)
    # guessed control coefficients
    cnum = length(tspan) - 1
    ctrl = [zeros(cnum) for _ in eachindex(Hc)]
    ctrl_bound = [-2.0, 2.0]

    obj = QFIM_obj()
    opt = QuanEstimationBase.ControlOpt(ctrl = ctrl, ctrl_bound = ctrl_bound, seed = 1234)
    dynamics =
        QuanEstimationBase.Lindblad(opt, tspan, rho0, H0, dH, Hc, decay, dyn_method = :Expm)
    output = QuanEstimationBase.Output(opt)
    state = rho0 |> vec
    action_space = Space([opt.ctrl_bound[1] .. opt.ctrl_bound[2] for _ = 1:cnum])
    state_space = Space(fill(-1.0e35 .. 1.0e35, length(state)))
    done = false
    rng = MersenneTwister(1234)
    reward = 0.0
    total_reward = 0.0
    ctrl_num = 1
    para_num = 1
    f_ini = 0.0
    f_list = [0.0]
    reward_all = [0.0]
    episode = 1

    env = StateEnv(
        obj,
        dynamics,
        output,
        action_space,
        state_space,
        state,
        done,
        rng,
        reward,
        total_reward,
        ctrl_num,
        para_num,
        f_ini,
        f_list,
        reward_all,
        episode,
    )

    @test env.obj == obj
    @test env.dynamics == dynamics
    @test env.output == output
    @test env.action_space == action_space
    @test env.state_space == state_space
    @test env.state == state
    @test env.done == done
    @test env.rng == rng
    @test env.reward == reward
    @test env.total_reward == total_reward
    @test env.ctrl_num == ctrl_num
    @test env.para_num == para_num
    @test env.f_ini == f_ini
    @test env.f_list == f_list
    @test env.reward_all == reward_all
    @test env.episode == episode
end

# # Test for update! with ControlOpt
# function test_update_ControlOpt()
#     opt = "ControlOpt"
#     alg = "DDPG"
#     obj = "Objective"
#     dynamics = "Dynamics"
#     output = "Output"

#     @test_throws UndefVarError update!(opt, alg, obj, dynamics, output)
# end

# # Test for update! with StateOpt and Lindblad dynamics
# function test_update_StateOpt_Lindblad()
#     opt = "StateOpt"
#     alg = "DDPG"
#     obj = "Objective"
#     dynamics = "Lindblad"
#     output = "Output"

#     @test_throws UndefVarError update!(opt, alg, obj, dynamics, output)
# end

# # Test for update! with StateOpt and Kraus dynamics
# function test_update_StateOpt_Kraus()
#     opt = "StateOpt"
#     alg = "DDPG"
#     obj = "Objective"
#     dynamics = "Kraus"
#     output = "Output"

#     @test_throws UndefVarError update!(opt, alg, obj, dynamics, output)
# end

# Run the tests
function test_all()
    @testset "ControlEnv" begin
        test_ControlEnv()
    end

    @testset "StateEnv" begin
        test_StateEnv()
    end

    # @testset "update! with ControlOpt" begin
    #     test_update_ControlOpt()
    # end

    # @testset "update! with StateOpt and Lindblad dynamics" begin
    #     test_update_StateOpt_Lindblad()
    # end

    # @testset "update! with StateOpt and Kraus dynamics" begin
    #     test_update_StateOpt_Kraus()
    # end
end

# Call the test function
test_all()
