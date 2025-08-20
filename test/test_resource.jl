using Test
using LinearAlgebra

using QuanEstimationBase: 
    SpinSqueezing, 
    TargetTime

function test_resource()    
    ρ = Matrix(Diagonal([1, 0, 0, 0, 0, 0, 0, 0])) 
    
    # Test both output types
    @test SpinSqueezing(ρ; basis="Pauli", output="KU") ≈ 1.0 atol=1e-6   
    @test SpinSqueezing(ρ; basis="Pauli", output="WBIMH") ≈ 1.0 atol=1e-6

    @test SpinSqueezing(ρ; basis="Dicke",  output="WBIMH") ≈ 1.0 atol=1e-6
    @test SpinSqueezing(ρ; basis="Dicke",  output="KU") ≈ 1.0 atol=1e-6

    @test_throws ErrorException SpinSqueezing(ρ; basis="Pauli", output="invalid")
    @test_throws ErrorException SpinSqueezing(ρ; basis="invalid", output="KU")

    ρ1 = [0.5 0.0; 0.0 0.5]
    @test_throws ErrorException SpinSqueezing(ρ1; basis="Pauli", output="KU")
end  # function test_resource

function test_TargetTime()
    # Define test function
    testfunc = (t, omega) -> cos(omega * t)
    
    # Define time span and omega
    tspan = range(0, pi, 10000)
    omega = 1. 
    
    # Call TargetTime with the omega array
    @test TargetTime(0., tspan, testfunc, omega) ≈ pi / 2 atol = 1e-6
    
    # Test print output when target is not found
    @test TargetTime(2., tspan, testfunc, omega) == nothing
end

test_resource()
test_TargetTime()
