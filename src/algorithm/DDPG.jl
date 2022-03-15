## TODO: Need reconstruct !!!
using Flux: glorot_normal, glorot_uniform
using StableRNGs
using Flux.Losses
using IntervalSets

mutable struct ControlEnv <: AbstractEnv
    obj::Any
    dynamics::Any
    output::Any
    action_space::Space
    state_space::Space
    state::AbstractVector
    dstate::AbstractVector
    done::Bool
    rng::AbstractRNG
    reward::Float64
    total_reward::Float64
    t::Int
    tspan::AbstractVector
    tnum::Int
    ctrl_length::Int
    ctrl_num::Int
    para_num::Int
    f_noctrl::AbstractVector
    f_final::AbstractVector
    ctrl_list::AbstractVector
    ctrl_bound::AbstractVector
    total_reward_all::AbstractVector
    episode::Int
end

RLBase.action_space(env::ControlEnv) = env.action_space
RLBase.state_space(env::ControlEnv) = env.state_space
RLBase.reward(env::ControlEnv) = env.reward
RLBase.is_terminated(env::ControlEnv) = env.done 
RLBase.state(env::ControlEnv) = env.state

#### control optimization ####
function update!(opt::ControlOpt, alg::DDPG, obj, dynamics, output)
    (; max_episode, layer_num, layer_dim, rng) = alg
    #### environment of DDPG ####
    para_num = length(dynamics.data.dH)
    ctrl_num = length(dynamics.data.ctrl)
    tnum = length(dynamics.data.tspan)
    ctrl_length = length(dynamics.data.ctrl[1])
    dim = size(dynamics.data.ρ0)[1]
    state = dynamics.data.ρ0
    dstate = [state |> zero for _ = 1:para_num]
    state = state |> state_flatten
    action_space =
        Space([opt.ctrl_bound[1] .. opt.ctrl_bound[2] for _ = 1:ctrl_num])
    state_space = Space(fill(-1.0e35 .. 1.0e35, length(state)))

    dynamics_copy = set_ctrl(dynamics, [zeros(ctrl_length) for i = 1:ctrl_num])
    f_noctrl = objective(Val{:expm}, obj, dynamics_copy)
    f_ini, f_comp = objective(obj, dynamics)

    ctrl_list = [Vector{Float64}() for _ = 1:ctrl_num]

    set_f!(output, f_ini)
    set_buffer!(output, dynamics.data.ctrl)
    set_io!(output, f_noctrl[end], f_ini)
    show(opt, output, obj)

    total_reward_all = Vector{Float64}()
    episode = 1
    env = ControlEnv(
        obj,
        dynamics,
        output,
        action_space,
        state_space,
        state,
        dstate,
        true,
        rng,
        0.0,
        0.0,
        0,
        dynamics.data.tspan,
        tnum,
        ctrl_length,
        ctrl_num,
        para_num,
        f_noctrl,
        output.f_list,
        ctrl_list,
        opt.ctrl_bound,
        total_reward_all,
        episode,
    )
    reset!(env)

    ns, na = 2 * dim^2, ctrl_num
    init = glorot_uniform(rng)

    create_actor() = Chain(
        Dense(ns, layer_dim, relu; init = init),
        [Dense(layer_dim, layer_dim, relu; init = init) for _ = 1:layer_num]...,
        Dense(layer_dim, na, tanh; init = init),
    )

    create_critic() = Chain(
        Dense(ns + na, layer_dim, relu; init = init),
        [Dense(layer_dim, layer_dim, relu; init = init) for _ = 1:layer_num]...,
        Dense(layer_dim, 1; init = init),
    )

    agent = Agent(
        policy = DDPGPolicy(
            behavior_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            behavior_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            target_actor = NeuralNetworkApproximator(
                model = create_actor(),
                optimizer = ADAM(),
            ),
            target_critic = NeuralNetworkApproximator(
                model = create_critic(),
                optimizer = ADAM(),
            ),
            γ = 0.99f0,
            ρ = 0.995f0,
            na = ctrl_num,
            batch_size = 128,
            start_steps = 100 * ctrl_length,
            start_policy = RandomPolicy(
                Space([-10.0 .. 10.0 for _ = 1:ctrl_num]);
                rng = rng,
            ),
            update_after = 100 * ctrl_length,
            update_freq = 1 * ctrl_length,
            act_limit = opt.ctrl_bound[end],
            act_noise = 0.01,
            rng = rng,
        ),
        trajectory = CircularArraySARTTrajectory(
            capacity = 400 * ctrl_length,
            state = Vector{Float64} => (ns,),
            action = Vector{Float64} => (na,),
        ),
    )
    stop_condition = StopAfterStep(max_episode * ctrl_length, is_show_progress = false)
    hook = TotalRewardPerEpisode(is_display_on_exit = false)
    RLBase.run(agent, env, stop_condition, hook)

    set_io!(output,output.f_list[end])
    if save_type(output) == :no_save
        SaveReward(env.total_reward_all)
    end
end


function Base.rsplit(v, l::Int)
    u = reshape(v, l, length(v) ÷ l)
    [u[:, i] for i in 1:size(u, 2)]
end

state_flatten(s) = vcat((s |> reim .|> vec)...)
rsplit_half(v) = Base.rsplit(v, length(v) ÷ 2)
density_matrix(s) = complex.(rsplit_half(s)...) |> vec2mat

function RLBase.reset!(env::ControlEnv)
    state = env.dynamics.data.ρ0
    env.dstate = [state |> zero for _ = 1:(env.para_num)]
    env.state = state |> state_flatten
    env.t = 0
    env.done = false
    env.reward = 0.0
    env.total_reward = 0.0
    env.ctrl_list = [Vector{Float64}() for _ = 1:env.ctrl_num]
    nothing
end

function (env::ControlEnv)(a)
    bound!(a, env.ctrl_bound)
    _step!(env, a)
end

function _step!(env::ControlEnv, a)
    env.t += 1
    ρₜ, ∂ₓρₜ = env.state |> density_matrix, env.dstate
    ρₜₙ, ∂ₓρₜₙ = propagate(env.dynamics, ρₜ, ∂ₓρₜ,  a, env.t)## TODO:
    env.state = ρₜₙ |> state_flatten
    env.dstate = ∂ₓρₜₙ
    env.done = env.t > env.ctrl_length
    f_out, f_current = objective(env.obj, env.dynamics)
    reward_current = log(f_current / env.f_noctrl[env.t])
    env.reward = reward_current
    env.total_reward += reward_current
    [append!(env.ctrl_list[i], a[i]) for i = 1:length(a)]
    if env.done
        set_f!(env.output, f_out)
        set_buffer!(env.output, env.ctrl_list)
        set_io!(env.output, f_out, env.episode)
        show(env.output, env.obj)

        env.episode += 1
        SaveReward(env.output, env.total_reward)
    end
end

#### state optimization ####


