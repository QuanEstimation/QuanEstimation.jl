function test_ControlOpt()
    ctrl = [[1, 2, 3]]
    ctrl_bound = [-1, 1]
    seed = 1234
    opt = ControlOpt(; ctrl = ctrl, ctrl_bound = ctrl_bound, seed = seed)
    @test opt.ctrl == ctrl
    @test opt.ctrl_bound == ctrl_bound
    @test opt.rng == MersenneTwister(seed)
end

function test_StateOpt()
    psi = [0.5, 0.5]
    seed = 1234
    opt = StateOpt(; psi = psi, seed = seed)
    @test opt.psi == psi
    @test opt.rng == MersenneTwister(seed)
end

function test_Mopt_Projection()
    M = QuanEstimationBase.SIC(2)
    seed = 1234
    opt = MeasurementOpt(; mtype = :Projection,M = M, seed = seed)
    @test opt.M == M
    @test opt.rng == MersenneTwister(seed)
end

function test_Mopt_LinearComb()
    POVM_basis = QuanEstimationBase.SIC(2)
    M_num = 2
    seed = 1234
    opt = MeasurementOpt(; mtype = :LC, POVM_basis = POVM_basis, M_num = M_num, seed = seed)
    @test opt.POVM_basis == POVM_basis
    @test opt.M_num == M_num
    @test opt.rng == MersenneTwister(seed)
end

function test_Mopt_Rotation()
    POVM_basis = QuanEstimationBase.SIC(2)
    seed = 1234
    opt = MeasurementOpt(; mtype = :Rotation, POVM_basis = POVM_basis, seed = seed)
    @test opt.POVM_basis == POVM_basis
    @test opt.rng == MersenneTwister(seed)
end

function test_OptScenario()
    @testset "ControlOpt" begin
        test_ControlOpt()
    end

    @testset "StateOpt" begin
        test_StateOpt()
    end

    @testset "Mopt_Projection" begin
        test_Mopt_Projection()
    end

    @testset "Mopt_LinearComb" begin
        test_Mopt_LinearComb()
    end

    @testset "Mopt_Rotation" begin
        test_Mopt_Rotation()
    end

end

test_OptScenario()
