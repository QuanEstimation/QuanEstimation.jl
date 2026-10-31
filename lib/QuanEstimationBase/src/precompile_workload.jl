# Canonical first-call workloads for QuanEstimation 0.5.
#
# This is the 0.5 backport of the PrecompileTools governance introduced on
# the 1.0 development line. The cells use the public 0.5 API rather than the
# 1.0 graph API. Each cell returns its principal result so the test suite can
# check that precompilation exercises real, numerically valid paths.
#
# Keep the workload on small, deterministic Float64/ComplexF64 fixtures. New
# cells must include a semantic test and their marginal precompile cost must
# be measured before being added. User-defined functions and uncommon type
# combinations cannot be covered exhaustively by package precompilation.

using PrecompileTools: @compile_workload

# Run legacy 0.5 entry points without leaving their console/file output in the
# package source tree. In 0.5, optimize! and online MZI write results even
# when savefile=false; a temporary working directory contains those artifacts.
function _pcw_silent(f::F) where {F}
    return mktempdir() do dir
        cd(dir) do
            redirect_stdout(f, devnull)
        end
    end
end

function _pcw_lindblad_scheme(; dyn_method = :Expm, measurement = nothing)
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

function _pcw_qfim()
    return QFIM(_pcw_lindblad_scheme())
end

function _pcw_cfim()
    return CFIM(_pcw_lindblad_scheme(; measurement = SIC(2)))
end

function _pcw_kraus()
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

function _pcw_ode()
    return QFIM(_pcw_lindblad_scheme(; dyn_method = :Ode))
end

function _pcw_optimize(
    alg;
    dyn_method = :Expm,
    ctrl = [0.3 .* ones(10)],
)
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
    _pcw_silent() do
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

function _pcw_optimize_meas()
    scheme = _pcw_lindblad_scheme()
    opt = MeasurementOpt(;
        mtype = :LC,
        POVM_basis = SIC(2),
        M_num = 2,
        seed = 42,
    )
    _pcw_silent() do
        optimize!(
            scheme,
            opt;
            algorithm = AD(; max_episode = 2),
            objective = CFIM_obj(),
            savefile = false,
        )
    end
    # The 0.5 optimizer mutates opt.B in place but does not write the optimized
    # POVM back into scheme. Return the optimized coefficients, not the CFIM of
    # the untouched default measurement.
    return reduce(vcat, opt.B)
end

function _pcw_mzi()
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
            # online returns savefile_online's `nothing` in the 0.5 API. Read
            # back the actual estimator/outcome record rather than returning
            # the unchanged input schedule as a vacuous semantic anchor.
            jldopen("adaptive.dat", "r") do file
                (x = copy(file["x"]), y = copy(file["y"]))
            end
        end
    end
end

function _pcw_nhb()
    ρ = PlusState() * PlusState()'
    G = σz() / 2
    dρ = [-im * (G * ρ - ρ * G)]
    return NHB(ρ, dρ, reshape([1.0], 1, 1))
end

@compile_workload begin
    _pcw_qfim()
    _pcw_cfim()
    _pcw_kraus()
    _pcw_ode()
    _pcw_optimize(GRAPE(; max_episode = 2))
    _pcw_optimize(autoGRAPE(; max_episode = 2))
    _pcw_optimize(DE(; max_episode = 2, p_num = 4))
    _pcw_optimize(PSO(; max_episode = [2, 2], p_num = 4))
    _pcw_optimize(GRAPE(; max_episode = 2); dyn_method = :Ode)
    _pcw_optimize(
        GRAPE(; max_episode = 2);
        dyn_method = :Ode,
        ctrl = SineCTRL(; A = 0.3, ω = 2pi, ϕ = 0.0),
    )
    # Do not bake the CFIM measurement AD pullback: its cached reverse path
    # breaks unrelated multiparameter StateOpt/CFIM differentiation.
    _pcw_mzi()
    _pcw_nhb()
end
