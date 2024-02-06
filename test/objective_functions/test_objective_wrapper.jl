using Test
using QuanEstimation: AbstractDynamics, QFIM_obj, CFIM_obj, HCRB_obj, get_para, get_dim, LDtype, para_type, SIC, Objective

# Test for Objective with QFIM_obj
function test_Objective_QFIM_obj()
    dynamics = mock_dynamics()
    obj = QFIM_obj(W=missing, eps=0.01)
    result = Objective(dynamics, obj)
    # @test result isa QFIM_obj{Float64, Float64}
    @test result.W == I(get_para(dynamics.data)) |> Matrix
    @test result.eps == 0.01
end

# Test for Objective with CFIM_obj
function test_Objective_CFIM_obj()
    dynamics = mock_dynamics()
    obj = CFIM_obj(W=missing, M=missing, eps=0.01)
    result = Objective(dynamics, obj)
    # @test result isa CFIM_obj{Float64}
    @test result.W == I(get_para(dynamics.data)) |> Matrix
    @test result.M == SIC(get_dim(dynamics.data))
    @test result.eps == 0.01
end

# Test for Objective with HCRB_obj
function test_Objective_HCRB_obj()
    dynamics = mock_dynamics()
    obj = HCRB_obj(W=missing, eps=0.01)
    result = Objective(dynamics, obj)
    # @test result isa HCRB_obj{Float64}
    @test result.W == I(get_para(dynamics.data)) |> Matrix
    @test result.eps == 0.01
end


# Mock dynamics for testing

function mock_dynamics()
    # Set up test parameters
    opt = QuanEstimation.ControlOpt()
    tspan = range(0.0, 10.0, length = 100)
    rho0 = 0.5 * ones(2, 2)
    omega = 1.0
    sx = [0.0 1.0; 1.0 0.0im]
    sy = [0.0 -im; im 0.0]
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]
    Hc = [sx, sy, sz]
    decay = [[0.0 1.0; 0.0 0.0im], [0.0 0.0; 1.0 0.0im]]
    
    # Call the function
    return Lindblad(opt, tspan, rho0, H0, dH, Hc, decay)
end


# Run the tests
function test_Objective_Wrapper()
    @testset "Objective with QFIM_obj" begin
        test_Objective_QFIM_obj()
    end

    @testset "Objective with CFIM_obj" begin
        test_Objective_CFIM_obj()
    end

    @testset "Objective with HCRB_obj" begin
        test_Objective_HCRB_obj()
    end
end

# Call the test function
test_Objective_Wrapper()