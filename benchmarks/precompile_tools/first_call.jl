# Fresh-process first-call benchmark for the public 0.5 API.
# Usage: julia --project=. benchmarks/precompile_tools/first_call.jl qfim

length(ARGS) == 1 || error("usage: first_call.jl <scenario>")
const SCENARIO = ARGS[1]

const load_s = @elapsed using QuanEstimation
using SparseArrays

function lindblad_scheme(; dyn_method = :Expm, measurement = nothing)
    ψ = PlusState()
    ρ = ψ * ψ'
    tspan = collect(range(0.0, 1.0; length = 6))
    dynamics = Lindblad(
        0.7 * σz() / 2,
        [σz() / 2],
        tspan;
        dyn_method = dyn_method,
    )
    return GeneralScheme(; probe = ρ, param = dynamics, measurement = measurement)
end

function bench_qfim()
    return QFIM(lindblad_scheme())
end

function bench_cfim()
    return CFIM(lindblad_scheme(; measurement = SIC(2)))
end

function bench_kraus()
    ψ = PlusState()
    ρ = ψ * ψ'
    η = 0.3
    K = [ComplexF64[1 0; 0 sqrt(1 - η)], ComplexF64[0 sqrt(η); 0 0]]
    dK = [
        [ComplexF64[0 0; 0 -1 / (2 * sqrt(1 - η))]],
        [ComplexF64[0 1 / (2 * sqrt(η)); 0 0]],
    ]
    return QFIM(GeneralScheme(; probe = ρ, param = Kraus(K, dK)))
end

function bench_ode()
    return QFIM(lindblad_scheme(; dyn_method = :Ode))
end

function silent_public(f::F) where {F}
    return mktempdir() do dir
        cd(dir) do
            redirect_stdout(f, devnull)
        end
    end
end

function bench_optimize(alg; dyn_method = :Expm, ctrl = [0.3 .* ones(10)])
    tspan = collect(range(0.0, 1.0; length = 11))
    dynamics = Lindblad(
        0.5 * σz(),
        [0.5 * σz()],
        tspan,
        [σx()],
        [[σz(), 0.5]];
        ctrl = ctrl,
        dyn_method = dyn_method,
    )
    scheme = GeneralScheme(; probe = PlusState(), param = dynamics)
    opt = ControlOpt(; ctrl = param_data(scheme).ctrl, ctrl_bound = [-5.0, 5.0], seed = 42)
    silent_public() do
        optimize!(
            scheme,
            opt;
            algorithm = alg,
            objective = QFIM_obj(),
            savefile = false,
        )
    end
    return QFIM(scheme)
end

function bench_optimize_meas()
    scheme = lindblad_scheme()
    opt = MeasurementOpt(;
        mtype = :LC,
        POVM_basis = SIC(2),
        M_num = 2,
        seed = 42,
    )
    silent_public() do
        optimize!(
            scheme,
            opt;
            algorithm = AD(; max_episode = 2),
            objective = CFIM_obj(),
            savefile = false,
        )
    end
    # optimize! mutates opt.B but leaves scheme's default measurement intact.
    return reduce(vcat, opt.B)
end

function bench_mzi()
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

function bench_nhb()
    ρ = PlusState() * PlusState()'
    G = σz() / 2
    dρ = [-im * (G * ρ - ρ * G)]
    return NHB(ρ, dρ, reshape([1.0], 1, 1))
end

const SCENARIOS = Dict{String, Function}(
    "qfim" => bench_qfim,
    "cfim" => bench_cfim,
    "kraus" => bench_kraus,
    "ode" => bench_ode,
    "grape" => () -> bench_optimize(GRAPE(; max_episode = 2)),
    "autogrape" => () -> bench_optimize(autoGRAPE(; max_episode = 2)),
    "de" => () -> bench_optimize(DE(; max_episode = 2, p_num = 4)),
    "pso" => () -> bench_optimize(PSO(; max_episode = [2, 2], p_num = 4)),
    "ode_opt" => () -> bench_optimize(GRAPE(; max_episode = 2); dyn_method = :Ode),
    "waveform" => () -> bench_optimize(
        GRAPE(; max_episode = 2);
        dyn_method = :Ode,
        ctrl = SineCTRL(; A = 0.3, ω = 2pi, ϕ = 0.0),
    ),
    "meas_opt" => bench_optimize_meas,
    "mzi" => bench_mzi,
    "nhb" => bench_nhb,
)

haskey(SCENARIOS, SCENARIO) || error("unknown scenario: $SCENARIO")
const benchmark = SCENARIOS[SCENARIO]

finite_result(x::Number) = isfinite(x)
finite_result(x::AbstractArray) = all(isfinite, x)
finite_result(x::NamedTuple) = all(finite_result, values(x))
checksum(x::Number) = Float64(x)
# abs2 avoids the constant-L1 checksum of normalized measurement coefficients.
checksum(x::AbstractArray) = Float64(sum(abs2, x))
checksum(x::NamedTuple) = sum(checksum, values(x))
compile_s(stats) = hasproperty(stats, :compile_time) ? stats.compile_time : NaN
recompile_s(stats) = hasproperty(stats, :recompile_time) ? stats.recompile_time : NaN

function print_stats(label, stats)
    println(
        "METRIC scenario=", SCENARIO,
        " julia=", VERSION,
        " sample=", label,
        " load_s=", load_s,
        " time_s=", stats.time,
        " compile_s=", compile_s(stats),
        " recompile_s=", recompile_s(stats),
        " gc_s=", stats.gctime,
        " bytes=", stats.bytes,
        " checksum=", checksum(stats.value),
    )
    return nothing
end

first = @timed benchmark()
finite_result(first.value) || error("non-finite first result for $SCENARIO")
print_stats("first", first)

second = @timed benchmark()
finite_result(second.value) || error("non-finite second result for $SCENARIO")
print_stats("second", second)

warm = [@timed benchmark() for _ in 1:5]
for stats in warm
    finite_result(stats.value) || error("non-finite warm result for $SCENARIO")
end
warm_times = sort([stats.time for stats in warm])
warm_compiles = [compile_s(stats) for stats in warm]
warm_recompiles = [recompile_s(stats) for stats in warm]
warm_bytes = [stats.bytes for stats in warm]
println(
    "METRIC scenario=", SCENARIO,
    " julia=", VERSION,
    " sample=warm5",
    " load_s=", load_s,
    " time_s=", warm_times[3],
    " min_s=", warm_times[1],
    " max_s=", warm_times[end],
    " compile_s=", maximum(warm_compiles),
    " recompile_s=", maximum(warm_recompiles),
    " gc_s=", sum(stats.gctime for stats in warm) / length(warm),
    " bytes=", sum(warm_bytes) / length(warm_bytes),
    " checksum=", checksum(warm[3].value),
)
