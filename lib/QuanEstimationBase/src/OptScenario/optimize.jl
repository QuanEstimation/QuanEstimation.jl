function optimize!(
    scheme,
    opt;
    algorithm = autoGRAPE(),
    objective = QFIM_obj(),
    savefile = false,
)
    show(stdout, "text/plain", scheme)

    opt = init_opt(opt, scheme)
    objective = Objective(scheme, objective)
    output = Output(opt; save = savefile)
    optimize!(opt, algorithm, objective, scheme, output)
    show(objective, output)
end
