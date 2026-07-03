"""
    optimize!(scheme, opt; algorithm=autoGRAPE(), objective=QFIM_obj(), savefile=false)

Top-level optimization entry point. Initializes the optimizer, build the objective, creates an
`Output`, and dispatches to the algorithm-specific `optimize!`.
"""
function optimize!(
    scheme,
    opt;
    algorithm = autoGRAPE(),
    objective = QFIM_obj(),
    savefile = false,
)
    show(stdout,"text/plain", scheme)

    opt = init_opt(opt, scheme)
    objective = Objective(scheme, objective)
    output = Output(opt; save = savefile)
    optimize!(opt, algorithm, objective, scheme, output)
    show(objective, output)
end