# Test for Kraus function with StateOpt
function test_Kraus_StateOpt()
    opt = StateOpt()
    K = [1 0; 0 1]
    dK = [1 0; 0 1]
    expected_result = Kraus(opt.psi, K, dK)
    result = Kraus(opt, K, dK)
    @test result == expected_result
end

# Test for Kraus function with AbstractMopt
function test_Kraus_AbstractMopt()
    opt = AbstractMopt()
    ρ₀ = [1 0; 0 1]
    K = [1 0; 0 1]
    dK = [1 0; 0 1]
    expected_result = Kraus(ρ₀, K, dK)
    result = Kraus(opt, ρ₀, K, dK)
    @test result == expected_result
end

# Test for Kraus function with CompOpt
function test_Kraus_CompOpt()
    opt = CompOpt()
    K = [1 0; 0 1]
    dK = [1 0; 0 1]
    expected_result = Kraus(opt.psi, K, dK)
    result = Kraus(opt, K, dK)
    @test result == expected_result
end

# Run the tests
function test_KrausWrapper()
    @testset "Kraus_StateOpt" begin
        @test test_Kraus_StateOpt()
    end

    @testset "Kraus_AbstractMopt" begin
        @test test_Kraus_AbstractMopt()
    end

    @testset "Kraus_CompOpt" begin
        @test test_Kraus_CompOpt()
    end
end

# Call the test function
test_KrausWrapper()
