function test_resource()
    ρ = 0.5 * ones(2, 2)
    @test SpinSqueezing(ρ) == 1.0
    @test SpinSqueezing(ρ; basis = "Pauli", output = "WBIMH") == 1.0
end  # function test_resource

test_resource()
