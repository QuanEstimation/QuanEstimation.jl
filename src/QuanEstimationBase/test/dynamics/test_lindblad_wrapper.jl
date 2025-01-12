using Test

# Test for `init_scheme!` function with ControlOpt
function test_init_scheme_copt()
    (;tspan, rho0, H0, dH, Hc, ctrl) = generate_qubit_dynamics()
    opt = ControlOpt()
    dynamics = Lindblad(H0, dH, tspan, Hc)
    scheme = GeneralScheme(; probe=rho0, param=dynamics,)
    init_scheme!(opt, scheme)

    @test opt.ctrl == ctrl
end

# Test for Lindblad function with StateOpt
function test_Lindblad_StateOpt()
    # Set up test parameters
    opt = QuanEstimationBase.StateOpt()
    tspan = range(0.0, 10.0, length = 100)
    rho0 = 0.5 * ones(2, 2)
    omega = 1.0
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]
    
    # Call the function
    QuanEstimationBase.Lindblad(opt, tspan, H0, dH)
    
    # Add your assertions here to validate the result
    # @test ...
    return true
end

# Test for Lindblad function with MeasurementOpt
function test_Lindblad_MeasurementOpt()
    # Set up test parameters
    
    tspan = range(0.0, 10.0, length = 100)
    rho0 = 0.5 * ones(2, 2)
    omega = 1.0
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]
    dim = size(rho0, 1)
    POVM_basis = QuanEstimationBase.SIC(dim)
    opts = [
        QuanEstimationBase.MeasurementOpt(;
        mtype = :Projection,
        ),
        QuanEstimationBase.MeasurementOpt(;
        mtype = :LC,
        POVM_basis = POVM_basis,
        ),
        QuanEstimationBase.MeasurementOpt(;
        mtype = :Rotation, 
        POVM_basis = POVM_basis,
        )
    ]

    # Call the function for each opt
    for opt in opts
        QuanEstimationBase.Lindblad(opt, tspan, rho0, H0, dH)
        # Add your assertions here to validate the result
        # @test ...
    end

    return true
end

function test_Lindblad_StateControlOpt()
    # Set up test parameters
    opt = QuanEstimationBase.StateControlOpt()
    tspan = range(0.0, 10.0, length = 100)
    H0 = [1.0 0.0im; 0.0 -1.0]
    dH = [0.5 * H0]
    Hc = [H0]
    
    # Call the function
    QuanEstimationBase.Lindblad(opt, tspan, H0, dH, Hc)
    
    # Add your assertions here to validate the result
    # @test ...
    return true
end

# Test for Lindblad function with ControlMeasurementOpt
function test_Lindblad_ControlMeasurementOpt()
    # Set up test parameters
    opt = QuanEstimationBase.ControlMeasurementOpt()
    tspan = range(0.0, 10.0, length = 100)
    ρ₀ = 0.5 * ones(2, 2)
    H0 = [1.0 0.0im; 0.0 -1.0]
    dH = [0.5 * H0]
    Hc = [H0]
    
    # Call the function
    Lindblad(opt, tspan, ρ₀, H0, dH, Hc)
    
    # Add your assertions here to validate the result
    # @test ...
    return true
end

# Test for Lindblad function with StateMeasurementOpt
function test_Lindblad_StateMeasurementOpt()
    # Set up test parameters
    opt = QuanEstimationBase.StateMeasurementOpt()
    tspan = range(0.0, 10.0, length = 100)
    rho0 = 0.5 * ones(2, 2)
    omega = 1.0
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]
    Hc = [sz]
    
    # Call the function
    QuanEstimationBase.Lindblad(opt, tspan, H0, dH)
    
    # Add your assertions here to validate the result
    # @test ...
    return true
end

# Test for Lindblad function with SCMopt
function test_Lindblad_SCMopt()
    # Set up test parameters
    opt = QuanEstimationBase.SCMopt()
    tspan = range(0.0, 10.0, length = 100)
    rho0 = 0.5 * ones(2, 2)
    omega = 1.0
    sz = [1.0 0.0im; 0.0 -1.0]
    H0 = 0.5 * omega * sz
    dH = [0.5 * sz]
    Hc = [sz]
    M = [sz]
    
    # Call the function
    QuanEstimationBase.Lindblad(opt, tspan, H0, dH, Hc)
    
    # Add your assertions here to validate the result
    # @test ...
    return true
end


# Run the tests
function test_Lindblad_wrapper()
    @testset "ControlOpt" begin
        @test test_Lindblad_ControlOpt()
    end
    
    # @testset "StateOpt" begin
    #     @test test_Lindblad_StateOpt()
    # end
    
    # @testset "MeasurementOpt" begin
    #     @test test_Lindblad_MeasurementOpt()
    # end

    # @testset "StateControlOpt" begin
    #     @test test_Lindblad_StateControlOpt()
    # end

    # @testset "ControlMeasurementOpt" begin
    #     @test test_Lindblad_ControlMeasurementOpt()
    # end
    
    # @testset "StateMeasurementOpt" begin
    #     @test test_Lindblad_StateMeasurementOpt()
    # end
    
    # @testset "SCMopt" begin
    #     @test test_Lindblad_SCMopt()
    # end
end



# Call the test function
test_Lindblad_wrapper()