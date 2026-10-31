function numerical_mzi()
    N = 3
    ψ = sum([
        sin(k * pi / (N + 2)) * kron(basis(N + 1, k), basis(N + 1, N - k + 2)) for
        k in 1:(N + 1)
    ]) |> sparse
    ψ = ψ * sqrt(2 / (2 + N))
    ρ0 = ψ * ψ'
    x = collect(range(-pi, pi; length = 100))
    p = (1.0 / (x[end] - x[1])) .* ones(length(x))
    apt = Adapt_MZI(x, p, ρ0)
    res = [0, 1]
    return mktempdir() do dir
        cd(dir) do
            redirect_stdout(devnull) do
                online(apt; target = "sharpness", output = "phi", res = res)
            end
            QuanEstimationBase.jldopen("adaptive.dat", "r") do file
                (x = copy(file["x"]), y = copy(file["y"]))
            end
        end
    end
end

function numerical_mzi_offline_score(target)
    N = 3
    ψ = sum([
        sin(k * pi / (N + 2)) * kron(basis(N + 1, k), basis(N + 1, N - k + 2)) for
        k in 1:(N + 1)
    ]) |> sparse
    ψ = ψ * sqrt(2 / (2 + N))
    ρ0 = ψ * ψ'
    x = collect(range(-pi, pi; length = 100))
    p = (1.0 / (x[end] - x[1])) .* ones(length(x))
    a = QuanEstimationBase.destroy(N + 1) |> sparse
    device = isdefined(QuanEstimationBase, :_MZIExec) ? QuanEstimationBase._mzi_exec_bake(a) : a
    comb = [[0, 0], [0, 1], [1, 0], [1, 1]]
    delta_phi = [0.1, 0.2]
    score = if target == :sharpness
        QuanEstimationBase.calculate_offline{QuanEstimationBase.sharpness}(
            delta_phi, x, p, ρ0, device, comb, nothing,
        )
    else
        QuanEstimationBase.calculate_offline{QuanEstimationBase.MI}(
            delta_phi, x, p, ρ0, device, comb, nothing,
        )
    end
    return (score = score,)
end

function numerical_control_scheme(; dyn_method = :Expm)
    tspan = collect(range(0.0, 2.0; length = 50))
    ρ0 = complex(0.5 * ones(2, 2))
    H0 = 0.5 * SigmaZ()
    dH = [0.5 * SigmaZ()]
    Hc = [SigmaX(), SigmaY(), SigmaZ()]
    sp = ComplexF64[0 1; 0 0]
    sm = ComplexF64[0 0; 1 0]
    decay = [[sp, 0.0], [sm, 0.1]]
    ctrl = [0.3 .* ones(length(tspan) - 1) for _ in eachindex(Hc)]
    dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl = ctrl, dyn_method = dyn_method)
    return GeneralScheme(; probe = ρ0, param = dynamics)
end

function numerical_evolve()
    ρ, dρ = evolve(numerical_control_scheme())
    return (rho = ρ, drho = dρ)
end

function numerical_evolve_ode()
    ρ, dρ = evolve(numerical_control_scheme(; dyn_method = :Ode))
    return (rho = ρ, drho = dρ)
end

function numerical_pso(; dyn_method = :Expm, p_num = 48, max_episode = [6, 2])
    scheme = numerical_control_scheme(; dyn_method = dyn_method)
    opt = ControlOpt(;
        ctrl = param_data(scheme).ctrl,
        ctrl_bound = [-2.0, 2.0],
        seed = 42,
    )
    alg = PSO(; p_num = p_num, max_episode = max_episode)
    mktempdir() do dir
        cd(dir) do
            redirect_stdout(devnull) do
                optimize!(scheme, opt; algorithm = alg, objective = QFIM_obj(), savefile = false)
            end
        end
    end
    return (F = QFIM(scheme), ctrl = param_data(scheme).ctrl)
end

const NUMERICAL_CASES = Dict{String, Function}(
    "mzi" => numerical_mzi,
    "mzi_offline" => () -> numerical_mzi_offline_score(:sharpness),
    "mzi_offline_mi" => () -> numerical_mzi_offline_score(:MI),
    "evolve" => numerical_evolve,
    "evolve_ode" => numerical_evolve_ode,
    "pso" => () -> numerical_pso(),
    "pso_ode" => () -> numerical_pso(; dyn_method = :Ode, p_num = 12, max_episode = [3, 2]),
)

finite_numerical_result(x::Number) = isfinite(x)
finite_numerical_result(x::AbstractArray) = all(finite_numerical_result, x)
finite_numerical_result(x::NamedTuple) = all(finite_numerical_result, values(x))

numerical_checksum(x::Number) = abs2(x)
numerical_checksum(x::AbstractArray) = sum(numerical_checksum, x; init = 0.0)
numerical_checksum(x::NamedTuple) = sum(numerical_checksum, values(x); init = 0.0)

function numerical_result_payload(result::NamedTuple)
    return Dict{String, Any}(String(key) => value for (key, value) in pairs(result))
end
