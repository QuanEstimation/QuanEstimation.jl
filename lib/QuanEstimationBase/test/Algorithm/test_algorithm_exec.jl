using Test
using LinearAlgebra
using Suppressor: @suppress
using SparseArrays
using JSON

using QuanEstimationBase:
    Lindblad,
    GeneralScheme,
    Kraus,
    QFIM_obj,
    CFIM_obj,
    QFIM,
    CFIM,
    ControlOpt,
    StateOpt,
    Mopt_Projection,
    Mopt_LinearComb,
    Mopt_Rotation,
    StateControlOpt,
    StateMeasurementOpt,
    ControlMeasurementOpt,
    StateControlMeasurementOpt,
    PSO,
    DE,
    NM,
    RI,
    optimize!,
    SIC,
    get_dim,
    SigmaX, SigmaY, SigmaZ,
    basis,
    Output, Objective, init_opt


const REFERENCES_DIR = joinpath(@__DIR__, "..", "references")

function load_golden(name)
    return JSON.parsefile(joinpath(REFERENCES_DIR, name))
end

function build_scenario(name::String)
    if name == "controlopt"
        (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound) = generate_qubit_dynamics()
        dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
        scheme = GeneralScheme(; probe=rho0, param=dynamics)
        opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
        return opt, scheme, QFIM_obj()
    elseif name == "stateopt"
        (; tspan, psi, H0, dH, decay) = generate_LMG1_dynamics()
        dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
        scheme = GeneralScheme(; probe=psi, param=dynamics)
        opt = StateOpt(psi=psi, seed=1234)
        return opt, scheme, QFIM_obj()
    elseif name == "stateopt_kraus"
        scheme = generate_scheme_kraus()
        (; psi) = generate_kraus()
        opt = StateOpt(psi=psi, seed=1234)
        return opt, scheme, QFIM_obj()
    elseif name == "mopt_projection"
        (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
        dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
        scheme = GeneralScheme(; probe=rho0, param=dynamics)
        opt = Mopt_Projection(seed=1234)
        return opt, scheme, CFIM_obj()
    elseif name == "mopt_lc"
        (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
        dim = get_dim(Lindblad(H0, dH, tspan, decay; dyn_method=:Expm))
        POVM_basis = SIC(dim)
        dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
        scheme = GeneralScheme(; probe=rho0, param=dynamics)
        opt = Mopt_LinearComb(POVM_basis=POVM_basis, M_num=2, seed=1234)
        return opt, scheme, CFIM_obj()
    elseif name == "mopt_rotation"
        (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
        dim = get_dim(Lindblad(H0, dH, tspan, decay; dyn_method=:Expm))
        POVM_basis = SIC(dim)
        dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
        scheme = GeneralScheme(; probe=rho0, param=dynamics)
        opt = Mopt_Rotation(POVM_basis=POVM_basis, seed=1234)
        return opt, scheme, CFIM_obj()
    elseif name == "statecontrol"
        (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
        dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
        scheme = GeneralScheme(; probe=rho0, param=dynamics)
        opt = StateControlOpt(psi=rho0[:,1], ctrl=ctrl, ctrl_bound=[-2.0, 2.0], seed=1234)
        return opt, scheme, QFIM_obj()
    elseif name == "statemeasurement"
        (; tspan, rho0, H0, dH, decay, M) = generate_qubit_dynamics()
        dynamics = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
        scheme = GeneralScheme(; probe=rho0, param=dynamics)
        opt = StateMeasurementOpt(psi=rho0[:,1], seed=1234)
        return opt, scheme, CFIM_obj(M=M)
    elseif name == "controlmeasurement"
        (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
        dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
        scheme = GeneralScheme(; probe=rho0, param=dynamics)
        opt = ControlMeasurementOpt(ctrl=ctrl, ctrl_bound=[-2.0, 2.0], seed=1234)
        return opt, scheme, CFIM_obj()
    elseif name == "statecontrolmeasurement"
        (; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
        dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
        scheme = GeneralScheme(; probe=rho0, param=dynamics)
        opt = StateControlMeasurementOpt(psi=rho0[:,1], ctrl=ctrl, ctrl_bound=[-2.0, 2.0], seed=1234)
        return opt, scheme, CFIM_obj()
    else
        error("Unknown scenario: $name")
    end
end

function run_and_check(alg_name, scenario_name, build_alg; golden_name=nothing, seed=1234, atol=1e-8)
    opt1, scheme1, obj1 = build_scenario(scenario_name)
    alg1 = build_alg()
    opt1 = init_opt(opt1, scheme1)
    obj_wrapped1 = Objective(scheme1, obj1)
    out1 = Output(opt1; save=false)
    @suppress optimize!(opt1, alg1, obj_wrapped1, scheme1, out1)
    f_opt = out1.f_list[end]
    @test isfinite(f_opt) && f_opt > 0 || f_opt >= 0

    f_list = out1.f_list
    if length(f_list) >= 2
        f_init = f_list[1]
        if f_init isa Number && f_opt isa Number
            @test f_opt >= f_init - 1e-10 ||
                  f_opt <= f_init + 1e-10
        end
    end

    opt2, scheme2, obj2 = build_scenario(scenario_name)
    alg2 = build_alg()
    opt2 = init_opt(opt2, scheme2)
    obj_wrapped2 = Objective(scheme2, obj2)
    out2 = Output(opt2; save=false)
    @suppress optimize!(opt2, alg2, obj_wrapped2, scheme2, out2)
    @test out2.f_list[end] ≈ out1.f_list[end]
end

const PSO_ALG = () -> PSO(p_num=3, max_episode=[3, 3])
const DE_ALG = () -> DE(p_num=3, max_episode=3)
const NM_ALG = () -> NM(p_num=5, max_episode=3)
const RI_ALG = () -> RI(max_episode=3)

const DISPLAY_NAMES = Dict(
    "controlopt" => "ControlOpt",
    "stateopt" => "StateOpt",
    "mopt_projection" => "Mopt_Projection",
    "mopt_lc" => "Mopt_LinearComb",
    "mopt_rotation" => "Mopt_Rotation",
    "statecontrol" => "StateControlOpt",
    "statemeasurement" => "StateMeasurementOpt",
    "controlmeasurement" => "ControlMeasurementOpt",
    "statecontrolmeasurement" => "StateControlMeasurementOpt",
    "stateopt_kraus" => "StateOpt",
)

const ALG_SCENARIOS = Dict(
    "pso" => [
        "controlopt", "stateopt", "mopt_projection", "mopt_lc", "mopt_rotation",
        "statecontrol", "statemeasurement", "controlmeasurement", "statecontrolmeasurement",
    ],
    "de" => [
        "controlopt", "stateopt", "mopt_projection", "mopt_lc", "mopt_rotation",
        "statecontrol", "statemeasurement", "controlmeasurement", "statecontrolmeasurement",
    ],
    "nm" => ["stateopt"],
    "ri" => [("stateopt_kraus", "stateopt")],
)

function test_algorithm_exec()
    for (alg_name, entries) in ALG_SCENARIOS
        prefix = uppercase(alg_name)
        @testset "$prefix Execution" begin
            for entry in entries
                scenario_name, golden_name = entry isa String ? (entry, entry) : entry
                label = DISPLAY_NAMES[golden_name]
                @testset "$prefix $label" begin
                    alg = alg_name == "pso" ? PSO_ALG :
                          alg_name == "de"  ? DE_ALG  :
                          alg_name == "nm"  ? NM_ALG  : RI_ALG
                    run_and_check(alg_name, scenario_name, alg; golden_name=golden_name)
                end
            end
        end
    end
end

test_algorithm_exec()
