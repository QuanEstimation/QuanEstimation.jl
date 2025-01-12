using Test, QuanEstimationBase

# Test for `init_opt!` function with ControlOpt
function test_init_opt_copt()
    (;tspan, rho0, H0, dH, Hc, ctrl) = generate_qubit_dynamics()
    opt = ControlOpt()
    dynamics = Lindblad(H0, dH, tspan, Hc)
    scheme = GeneralScheme(; probe=rho0, param=dynamics,)
    opt = init_opt(opt, scheme)

    @test opt.ctrl == ctrl
end

function test_init_opt()
    @testset "ControlOpt" begin
        test_init_opt_copt()
    end
end

test_init_opt()