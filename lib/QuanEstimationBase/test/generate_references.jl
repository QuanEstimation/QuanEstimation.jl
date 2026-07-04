# generate_references.jl — Golden reference generation for PSO/DE 18 scenarios
# Usage: julia --project test/generate_references.jl
# Output: test/references/{pso,de}_{scenario}.json

using Random
using LinearAlgebra
using Suppressor: @suppress
using SparseArrays
using JSON

using QuanEstimationBase:
    Lindblad, GeneralScheme, Kraus,
    QFIM_obj, CFIM_obj, QFIM, CFIM,
    ControlOpt, StateOpt, Mopt_Projection, Mopt_LinearComb, Mopt_Rotation,
    StateControlOpt, StateMeasurementOpt, ControlMeasurementOpt, StateControlMeasurementOpt,
    PSO, DE, NM, RI,
    optimize!,
    SIC, get_dim,
    SigmaX, SigmaY, SigmaZ,
    basis, state_data, param_data,
    Output, Objective, init_opt

include("utils.jl")

const REF_DIR = joinpath(@__DIR__, "references")
mkpath(REF_DIR)

function run_and_capture(opt, alg, obj, scheme; seed=1234)
    Random.seed!(seed)
    opt = init_opt(opt, scheme)
    obj_wrapped = Objective(scheme, obj)
    out = Output(opt; save=false)
    @suppress optimize!(opt, alg, obj_wrapped, scheme, out)
    return out
end

function save_golden(name, golden::Dict)
    golden["scenario"] = name
    open(joinpath(REF_DIR, "$name.json"), "w") do io
        JSON.print(io, golden, 4)
    end
end

function save_golden(name, out)
    golden = Dict(
        "f_opt" => out.f_list[end],
        "f_list" => collect(out.f_list),
        "scenario" => name,
    )
    open(joinpath(REF_DIR, "$name.json"), "w") do io
        JSON.print(io, golden, 4)
    end
end

# ============================================================
# Scenarios: each returns (opt, scheme)
# ============================================================

function scenario_qubit_control()
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound) = generate_qubit_dynamics()
    dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dyn)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    return opt, scheme
end

function scenario_lmg1_state()
    (; tspan, psi, H0, dH, decay) = generate_LMG1_dynamics()
    dyn = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
    scheme = GeneralScheme(; probe=psi, param=dyn)
    opt = StateOpt(psi=psi, seed=1234)
    return opt, scheme
end

function scenario_qubit_nocontrol()
    (; tspan, rho0, H0, dH, decay) = generate_qubit_dynamics()
    dyn = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dyn)
    return scheme
end

function scenario_qubit_controlopt_only()
    (; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound) = generate_qubit_dynamics()
    dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
    scheme = GeneralScheme(; probe=rho0, param=dyn)
    opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
    return opt, scheme
end

# ============================================================
# PSO
# ============================================================
println("Generating PSO scenarios...")

# 1: ControlOpt
opt, scheme = scenario_qubit_control()
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), QFIM_obj(), scheme)
save_golden("pso_controlopt", out)
println("  pso_controlopt → f_opt=$(out.f_list[end])")

# 2: StateOpt
opt, scheme = scenario_lmg1_state()
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), QFIM_obj(), scheme)
save_golden("pso_stateopt", out)
println("  pso_stateopt → f_opt=$(out.f_list[end])")

# 3: Mopt_Projection
scheme = scenario_qubit_nocontrol()
opt = Mopt_Projection(seed=1234)
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), CFIM_obj(), scheme)
save_golden("pso_mopt_projection", out)
println("  pso_mopt_projection → f_opt=$(out.f_list[end])")

# 4: Mopt_LinearComb
scheme = scenario_qubit_nocontrol()
dim = get_dim(scheme)
povm = SIC(dim)

# 5: Mopt_Rotation
scheme = scenario_qubit_nocontrol()
dim = get_dim(scheme)
povm = SIC(dim)
opt = Mopt_LinearComb(POVM_basis=povm, M_num=2, seed=1234)
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), CFIM_obj(), scheme)
save_golden("pso_mopt_lc", out)
println("  pso_mopt_lc → f_opt=$(out.f_list[end])")

# 5: Mopt_Rotation
scheme = scenario_qubit_nocontrol()
dim = get_dim(scheme)
povm = SIC(dim)
opt = Mopt_Rotation(POVM_basis=povm, seed=1234)
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), CFIM_obj(), scheme)
save_golden("pso_mopt_rotation", out)
println("  pso_mopt_rotation → f_opt=$(out.f_list[end])")

# 6: StateControlOpt
(; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dyn)
opt = StateControlOpt(ctrl_bound=[-2.0, 2.0], seed=1234)
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), QFIM_obj(), scheme)
save_golden("pso_statecontrol", out)
println("  pso_statecontrol → f_opt=$(out.f_list[end])")

# 7: StateMeasurementOpt
(; tspan, rho0, H0, dH, decay, M) = generate_qubit_dynamics()
dyn = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dyn)
opt = StateMeasurementOpt(seed=1234)
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), CFIM_obj(M=M), scheme)
save_golden("pso_statemeasurement", out)
println("  pso_statemeasurement → f_opt=$(out.f_list[end])")

# 8: ControlMeasurementOpt
(; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dyn)
opt = ControlMeasurementOpt(ctrl_bound=[-2.0, 2.0], seed=1234)
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), CFIM_obj(), scheme)
save_golden("pso_controlmeasurement", out)
println("  pso_controlmeasurement → f_opt=$(out.f_list[end])")

# 9: StateControlMeasurementOpt
(; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dyn)
opt = StateControlMeasurementOpt(ctrl_bound=[-2.0, 2.0], seed=1234)
out = run_and_capture(opt, PSO(p_num=3, max_episode=[5,5]), CFIM_obj(), scheme)
save_golden("pso_statecontrolmeasurement", out)
println("  pso_statecontrolmeasurement → f_opt=$(out.f_list[end])")

# ============================================================
# DE
# ============================================================
println("Generating DE scenarios...")

# 10: ControlOpt
opt, scheme = scenario_qubit_controlopt_only()
out = run_and_capture(opt, DE(p_num=3, max_episode=5), QFIM_obj(), scheme)
save_golden("de_controlopt", out)
println("  de_controlopt → f_opt=$(out.f_list[end])")

# 11: StateOpt
opt, scheme = scenario_lmg1_state()
out = run_and_capture(opt, DE(p_num=3, max_episode=5), QFIM_obj(), scheme)
save_golden("de_stateopt", out)
println("  de_stateopt → f_opt=$(out.f_list[end])")

# 12: Mopt_Projection
scheme = scenario_qubit_nocontrol()
opt = Mopt_Projection(seed=1234)
out = run_and_capture(opt, DE(p_num=3, max_episode=5), CFIM_obj(), scheme)
save_golden("de_mopt_projection", out)
println("  de_mopt_projection → f_opt=$(out.f_list[end])")

# 13: Mopt_LinearComb
scheme = scenario_qubit_nocontrol()
dim = get_dim(scheme)
povm = SIC(dim)
opt = Mopt_LinearComb(POVM_basis=povm, M_num=2, seed=1234)
out = run_and_capture(opt, DE(p_num=3, max_episode=5), CFIM_obj(), scheme)
save_golden("de_mopt_lc", out)
println("  de_mopt_lc → f_opt=$(out.f_list[end])")

# 14: Mopt_Rotation
scheme = scenario_qubit_nocontrol()
dim = get_dim(scheme)
povm = SIC(dim)
opt = Mopt_Rotation(POVM_basis=povm, seed=1234)
out = run_and_capture(opt, DE(p_num=3, max_episode=5), CFIM_obj(), scheme)
save_golden("de_mopt_rotation", out)
println("  de_mopt_rotation → f_opt=$(out.f_list[end])")

# 15: StateControlOpt
(; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dyn)
opt = StateControlOpt(ctrl_bound=[-2.0, 2.0], seed=1234)
out = run_and_capture(opt, DE(p_num=3, max_episode=5), QFIM_obj(), scheme)
save_golden("de_statecontrol", out)
println("  de_statecontrol → f_opt=$(out.f_list[end])")

# 16: StateMeasurementOpt
(; tspan, rho0, H0, dH, decay, M) = generate_qubit_dynamics()
dyn = Lindblad(H0, dH, tspan, decay; dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dyn)
opt = StateMeasurementOpt(seed=1234)
out = run_and_capture(opt, DE(p_num=3, max_episode=5), CFIM_obj(M=M), scheme)
save_golden("de_statemeasurement", out)
println("  de_statemeasurement → f_opt=$(out.f_list[end])")

# 17: ControlMeasurementOpt
(; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dyn)
opt = ControlMeasurementOpt(ctrl_bound=[-2.0, 2.0], seed=1234)
out = run_and_capture(opt, DE(p_num=3, max_episode=5), CFIM_obj(), scheme)
save_golden("de_controlmeasurement", out)
println("  de_controlmeasurement → f_opt=$(out.f_list[end])")

# 18: StateControlMeasurementOpt
(; tspan, rho0, H0, dH, Hc, decay, ctrl) = generate_qubit_dynamics()
dyn = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
scheme = GeneralScheme(; probe=rho0, param=dyn)
opt = StateControlMeasurementOpt(ctrl_bound=[-2.0, 2.0], seed=1234)
out = run_and_capture(opt, DE(p_num=3, max_episode=5), CFIM_obj(), scheme)
save_golden("de_statecontrolmeasurement", out)
println("  de_statecontrolmeasurement → f_opt=$(out.f_list[end])")

# ============================================================
# NM
# ============================================================
println("Generating NM scenario...")

# stateopt
opt, scheme = scenario_lmg1_state()
out = run_and_capture(opt, NM(p_num=5, max_episode=3), QFIM_obj(), scheme)
save_golden("nm_stateopt", out)
println("  nm_stateopt → f_opt=$(out.f_list[end])")

# ============================================================
# RI
# ============================================================
println("Generating RI scenario...")

# stateopt_kraus
scheme = generate_scheme_kraus()
opt = StateOpt(seed=1234)
out = run_and_capture(opt, RI(max_episode=3), QFIM_obj(), scheme)
save_golden("ri_stateopt", out)
println("  ri_stateopt → f_opt=$(out.f_list[end])")

println("\nDone. $(length(readdir(REF_DIR))) golden references saved to $REF_DIR")

# ============================================================
# GRAPE/autoGRAPE Control Optimization Golden References
# CRITICAL: must match test execution order (sequential scheme modification)
# ============================================================
println("Generating GRAPE/control optimization scenarios...")
using QuanEstimationBase: autoGRAPE, GRAPE, PlusState, param_data

# test_copt_qfi: autoGRAPE then GRAPE_Adam on same scheme (QFIM)
function run_grape_qfi(alg, obj, opt, scheme; seed=1234)
    Random.seed!(seed)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
    return Dict{String,Any}("f_opt" => QFIM(scheme)[1])  # QFIM(scheme)[1] for single-para
end

# test_copt_cfi: autoGRAPE then GRAPE_Adam on same scheme (CFIM → QFIM)
function run_grape_cfi(alg, obj, opt, scheme; seed=1234)
    Random.seed!(seed)
    @suppress optimize!(scheme, opt; algorithm=alg, objective=obj, savefile=false)
    return Dict{String,Any}("f_opt" => QFIM(scheme)[1])  # test asserts QFIM(scheme)[1] after CFIM opt
end

(; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound) = generate_qubit_dynamics()
dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
scheme_qfi = GeneralScheme(; probe=PlusState(), param=dynamics)

# Step 1: autoGRAPE + QFI (modifies scheme_qfi)
Random.seed!(1234)
opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
alg = autoGRAPE(Adam=true, max_episode=10, epsilon=0.01, beta1=0.90, beta2=0.99)
golden = run_grape_qfi(alg, QFIM_obj(), opt, scheme_qfi)
save_golden("grape_autogrape_qfi", golden)
println("  grape_autogrape_qfi -> f_opt=$(golden["f_opt"])")

# Step 2: GRAPE_Adam + QFI (runs on scheme_qfi already modified by autoGRAPE)
Random.seed!(1234)
opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
alg = GRAPE(Adam=true, max_episode=3)
golden = run_grape_qfi(alg, QFIM_obj(), opt, scheme_qfi)
save_golden("grape_adam_qfi", golden)
println("  grape_adam_qfi -> f_opt=$(golden["f_opt"])")

# test_copt_cfi: autoGRAPE then GRAPE_Adam on same scheme
# NOTE: autoGRAPE asserts CFIM(scheme)[1]; GRAPE_Adam asserts QFIM(scheme)[1]
(; tspan, rho0, H0, dH, Hc, decay, ctrl, ctrl_bound, M) = generate_qubit_dynamics()
dynamics = Lindblad(H0, dH, tspan, Hc, decay; ctrl=ctrl, dyn_method=:Expm)
scheme_cfi = GeneralScheme(; probe=rho0, param=dynamics, measurement=M)

# Step 1: autoGRAPE + CFI (modifies scheme_cfi)
Random.seed!(1234)
opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
alg = autoGRAPE(Adam=true, max_episode=10, epsilon=0.01, beta1=0.90, beta2=0.99)
@suppress optimize!(scheme_cfi, opt; algorithm=alg, objective=CFIM_obj(M=M), savefile=false)
golden = Dict{String,Any}("f_opt" => CFIM(scheme_cfi)[1])
save_golden("grape_autogrape_cfi", golden)
println("  grape_autogrape_cfi -> f_opt=$(golden["f_opt"])")

# Step 2: GRAPE_Adam + CFI (runs on scheme_cfi already modified)
Random.seed!(1234)
opt = ControlOpt(ctrl=ctrl, ctrl_bound=ctrl_bound, seed=1234)
alg = GRAPE(Adam=true, max_episode=3)
@suppress optimize!(scheme_cfi, opt; algorithm=alg, objective=CFIM_obj(M=M), savefile=false)
golden = Dict{String,Any}("f_opt" => QFIM(scheme_cfi)[1])
save_golden("grape_adam_cfi", golden)
println("  grape_adam_cfi -> f_opt=$(golden["f_opt"])")

println("\nTotal golden references: $(length(readdir(REF_DIR)))")
