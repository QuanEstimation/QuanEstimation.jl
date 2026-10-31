# Re-bake the canonical Base cells on the terminal 0.5 method tables.
#
# NVMagnetometer extends QuanEstimationBase.QFIM/CFIM/HCRB/optimize! and the
# error-control interface. Those added methods can invalidate native code
# inferred while Base was precompiled on its own. The root package is loaded
# after both method sets are visible, so its workload is the full-stack cache.
# Keep the cells themselves in Base: this file intentionally contains no
# duplicate numerical fixtures.

using PrecompileTools: @compile_workload

const QEB = QuanEstimationBase

@compile_workload begin
    QEB._pcw_qfim()
    QEB._pcw_cfim()
    QEB._pcw_kraus()
    QEB._pcw_ode()
    QEB._pcw_optimize(GRAPE(; max_episode = 2))
    QEB._pcw_optimize(autoGRAPE(; max_episode = 2))
    QEB._pcw_optimize(DE(; max_episode = 2, p_num = 4))
    QEB._pcw_optimize(PSO(; max_episode = [2, 2], p_num = 4))
    QEB._pcw_optimize(GRAPE(; max_episode = 2); dyn_method = :Ode)
    QEB._pcw_optimize(
        GRAPE(; max_episode = 2);
        dyn_method = :Ode,
        ctrl = SineCTRL(; A = 0.3, ω = 2pi, ϕ = 0.0),
    )
    QEB._pcw_mzi()
    QEB._pcw_nhb()
end
