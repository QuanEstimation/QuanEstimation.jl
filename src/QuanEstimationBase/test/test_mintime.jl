using Test
include("/home/hmyuuu/.julia/dev/QuanEstimation/src/Common/mintime.jl")

# Test mintime function with binary algorithm
@testset "mintime with binary algorithm" begin
    # Define test inputs
    f = 0.5
    # Call the mintime function
    mintime(:binary, f, system)

    # @test ...

    # Clean up any generated files if necessary
    # rm("mtspan.csv")
    # rm("controls.csv")
end

# Test mintime function with forward algorithm
@testset "mintime with forward algorithm" begin
    # Define test inputs
    f = 0.5

    # Call the mintime function
    mintime(:forward, f, system)

    # @test ...

    # Clean up any generated files if necessary
    # rm("mtspan.csv")
    # rm("controls.csv")
end
