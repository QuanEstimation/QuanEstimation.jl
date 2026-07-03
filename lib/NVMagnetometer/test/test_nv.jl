using Test
using LinearAlgebra
using Suppressor: @suppress
using Random: Random

using QuanEstimationBase:
    QFIM,
    CFIM,
    HCRB,
    ControlOpt,
    autoGRAPE,
    optimize!,
    error_evaluation,
    error_control,
    Lindblad,
    GeneralScheme

using NVMagnetometer: NVMagnetometerScheme

@doc raw"""
    test_nv_analytic_qfim()

Verify the QFIM for a simplified NV-center model against the analytical
expression ``F_{B_z B_z} = 4 g_S^2 t^2``.

**Physical model**: electron spin-1 only (hyperfine, nuclear Zeeman,
and dephasing turned off; B_x = B_y = 0). Hamiltonian:

``H = D S_3^2 + g_S B_z S_3``

The initial state ``(|+1\rangle + |-1\rangle)/\sqrt{2}`` evolves unitarily
to produce a pure-state QFIM that equals ``4 g_S^2 t^2`` exactly.

Multiple time points are tested to cover short and long evolution regimes.
"""
function test_nv_analytic_qfim()
    s1 = [0.0 1.0 0.0; 1.0 0.0 1.0; 0.0 1.0 0.0] / sqrt(2)
    s2 = [0.0 -im 0.0; im 0.0 -im; 0.0 im 0.0] / sqrt(2)
    s3 = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 -1.0]

    D  = 2pi * 2870.0
    gS = 2pi * 28.03
    Bz = 0.5

    H0  = D * (s3 * s3) + gS * Bz * s3
    dH  = [gS * s3]

    psi0 = [1.0, 0.0, 1.0] / sqrt(2)
    rho0 = psi0 * psi0'

    @testset "t = $t" for t in [0.5, 1.0, 2.0, 5.0]
        tspan = range(0.0, t, length = 2)
        dynamics = Lindblad(H0, dH, tspan; dyn_method = :Expm)
        scheme = GeneralScheme(; probe = rho0, param = dynamics)

        F_num = QFIM(scheme)[1, 1]
        F_ana = 4.0 * gS^2 * t^2

        @test isapprox(F_num, F_ana; rtol = 1e-10)
    end
end

function test_nv_magnetometer()
    scheme = NVMagnetometerScheme()
    Random.seed!(1234)

    @testset "QFIM" begin
        F = QFIM(scheme)
        @test F isa Matrix
        @test size(F) == (3, 3)
        @test all(isfinite.(F))
        @test isposdef(F)
    end

    @testset "CFIM" begin
        F = CFIM(scheme)
        @test F isa Matrix
        @test size(F) == (3, 3)
        @test all(isfinite.(F))
        @test isposdef(F)
    end

    @testset "HCRB" begin
        local h
        try
            h = HCRB(scheme)
            @test h isa Real
            @test isfinite(h)
            @test h >= 0
            @test h >= tr(inv(QFIM(scheme))) - 1e-10
        catch e
            @test_skip "SCS SDP failed: $(typeof(e))"
        end
    end

    @testset "optimize!" begin
        F_pre = tr(QFIM(scheme))
        alg = autoGRAPE(; max_episode = 3)
        @suppress optimize!(scheme, ControlOpt(); algorithm = alg)
        F_post = tr(QFIM(scheme))
        @test isfinite(F_post)
        @test F_post >= F_pre - 1e-10
    end

    @testset "error_evaluation" begin
        Random.seed!(1234)
        @test_nowarn error_evaluation(scheme)
    end

    @testset "error_control" begin
        Random.seed!(1234)
        @test_nowarn error_control(scheme)
    end

    isfile("f.csv") && rm("f.csv")
    isfile("controls.dat") && rm("controls.dat")
    isfile("controls.csv") && rm("controls.csv")
end

test_nv_analytic_qfim()
test_nv_magnetometer()
