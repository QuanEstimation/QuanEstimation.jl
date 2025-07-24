function test_resource()    
    ρ = Matrix(Diagonal([1, 0, 0, 0, 0, 0, 0, 0])) 
    
    # Test both output types
    @test SpinSqueezing(ρ; basis="Pauli", output="KU") ≈ 1.0 atol=1e-6   
    @test SpinSqueezing(ρ; basis="Pauli", output="WBIMH") ≈ 1.0 atol=1e-6

    @test SpinSqueezing(ρ; basis="Dicke",  output="WBIMH") ≈ 1.0 atol=1e-6
    @test SpinSqueezing(ρ; basis="Dicke",  output="KU") ≈ 1.0 atol=1e-6

    @test_throws ErrorException SpinSqueezing(ρ; basis="Pauli", output="invalid")
    @test_throws ErrorException SpinSqueezing(ρ; basis="invalid", output="KU")
end  # function test_resource

test_resource()
