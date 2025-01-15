using Test
using Random
using QuanEstimationBase:
    ControlOpt,
    StateOpt,
    Mopt_Projection,
    Mopt_LinearComb,
    Mopt_Rotation,
    StateControlOpt,
    ControlMeasurementOpt,
    StateMeasurementOpt,
    StateControlMeasurementOpt

# Test for ControlOpt
function test_ControlOpt()
    ctrl = [[1, 2, 3]]
    ctrl_bound = [-1, 1]
    seed = 1234
    opt = ControlOpt(; ctrl = ctrl, ctrl_bound = ctrl_bound, seed = seed)
    @test opt.ctrl == ctrl
    @test opt.ctrl_bound == ctrl_bound
    @test opt.rng == MersenneTwister(seed)
end

# Test for StateOpt
function test_StateOpt()
    psi = [0.5, 0.5]
    seed = 1234
    opt = StateOpt(; psi = psi, seed = seed)
    @test opt.psi == psi
    @test opt.rng == MersenneTwister(seed)
end

# Test for Mopt_Projection
function test_Mopt_Projection()
    M = QuanEstimationBase.SIC(2)
    seed = 1234
    opt = Mopt_Projection(M = M, seed = seed)
    @test opt.M == M
    @test opt.rng == MersenneTwister(seed)
end

# Test for Mopt_LinearComb
function test_Mopt_LinearComb()
    POVM_basis = QuanEstimationBase.SIC(2)
    M_num = 2
    seed = 1234
    opt = Mopt_LinearComb(POVM_basis = POVM_basis, M_num = M_num, seed = seed)
    @test opt.POVM_basis == POVM_basis
    @test opt.M_num == M_num
    @test opt.rng == MersenneTwister(seed)
end

# Test for Mopt_Rotation
function test_Mopt_Rotation()
    POVM_basis = QuanEstimationBase.SIC(2)
    seed = 1234
    opt = Mopt_Rotation(POVM_basis = POVM_basis, seed = seed)
    @test opt.POVM_basis == POVM_basis
    @test opt.rng == MersenneTwister(seed)
end

# # Test for update! function in ControlOpt
# function test_update_ControlOpt()
#     opt = ControlOpt(ctrl=[[1, 2, 3]], ctrl_bound=[-1, 1], seed=1234)
#     alg = PSO(max_episode=10, p_num=5, ini_particle=([opt.ctrl,],), c0=0.1, c1=0.2, c2=0.3)
#     obj = ...
#     dynamics = ...
#     output = ...
#     update!(opt, alg, obj, dynamics, output)
#     # Add assertions here
# end

# # Test for update! function in StateOpt
# function test_update_StateOpt()
#     opt = StateOpt(psi=[0.5, 0.5], seed=1234)
#     alg = PSO(max_episode=10, p_num=5, ini_particle=([opt.psi],), c0=0.1, c1=0.2, c2=0.3)
#     obj = ...
#     dynamics = ...
#     output = ...
#     update!(opt, alg, obj, dynamics, output)
#     # Add assertions here
# end

# # Test for update! function in Mopt_Projection
# function test_update_Mopt_Projection()
#     opt = Mopt_Projection()
#     alg = PSO(max_episode=10, p_num=5, ini_particle=([opt.M],), c0=0.1, c1=0.2, c2=0.3)
#     obj = ...
#     dynamics = ...
#     output = ...
#     update!(opt, alg, obj, dynamics, output)
#     # Add assertions here
# end

# # Test for update! function in Mopt_LinearComb
# function test_update_Mopt_LinearComb()
#     opt = Mopt_LinearComb(POVM_basis=QuanEstimationBase.SIC(2), M_num=2, seed=1234)
#     alg = PSO(max_episode=10, p_num=5, ini_particle=([opt.B],), c0=0.1, c1=0.2, c2=0.3)
#     obj = ...
#     dynamics = ...
#     output = ...
#     update!(opt, alg, obj, dynamics, output)
#     # Add assertions here
# end

# # Test for update! function in Mopt_Rotation
# function test_update_Mopt_Rotation()
#     opt = Mopt_Rotation(POVM_basis=QuanEstimationBase.SIC(2), seed=1234)
#     alg = PSO(max_episode=10, p_num=5, ini_particle=([opt.s],), c0=0.1, c1=0.2, c2=0.3)
#     obj = ...
#     dynamics = ...
#     output = ...
#     update!(opt, alg, obj, dynamics, output)
#     # Add assertions here
# end

# Run the tests
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

    # @testset "update! ControlOpt" begin
    #     test_update_ControlOpt()
    # end

    # @testset "update! StateOpt" begin
    #     test_update_StateOpt()
    # end

    # @testset "update! Mopt_Projection" begin
    #     test_update_Mopt_Projection()
    # end

    # @testset "update! Mopt_LinearComb" begin
    #     test_update_Mopt_LinearComb()
    # end

    # @testset "update! Mopt_Rotation" begin
    #     test_update_Mopt_Rotation()
    # end
end

# Call the test function
test_OptScenario()
