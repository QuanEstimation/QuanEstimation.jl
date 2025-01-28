function test_error_evaluation()
    scheme = generate_qubit_scheme()

    error_evaluation(scheme)
end  # function test_error_evaluation

function test_error_control()
    scheme = generate_qubit_scheme()

    error_control(scheme)
    @test_throws ArgumentError error_control(scheme, objective="HCRB")
end  # function test_error_control

function test_error()
    test_error_evaluation()
    test_error_control()
end  # function test_error

test_error()