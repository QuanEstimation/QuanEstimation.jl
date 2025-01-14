using QuanEstimationBase:
    suN_generatorU, suN_generatorV, suN_generatorW, suN_generator, basis, SIC
# Test for suN_generatorU function
function test_suN_generatorU()
    n = 3
    k = 2
    expected_result = sparse([1, 3], [3, 1], [1, 1], n, n)
    result = suN_generatorU(n, k)
    @test result == expected_result
end

# Test for suN_generatorV function
function test_suN_generatorV()
    n = 3
    k = 2
    expected_result = sparse([1, 3], [3, 1], [-im, im], n, n)
    result = suN_generatorV(n, k)
    @test result == expected_result
end

# Test for suN_generatorW function
function test_suN_generatorW()
    n = 3
    k = 2
    expected_result = spdiagm(n, n, [1, 1, -2])
    result = suN_generatorW(n, k)
    @test result == expected_result
end

# Test for suN_generator function
function test_suN_generator()
    n = 2
    expected_result = [
        sparse([2, 1], [1, 2], ComplexF64[1.0+0.0im, 1.0+0.0im], 2, 2),
        sparse([2, 1], [1, 2], ComplexF64[0.0+1.0im, 0.0-1.0im], 2, 2),
        sparse([1, 2], [1, 2], ComplexF64[1.0+0.0im, -1.0+0.0im], 2, 2),
    ]
    result = suN_generator(n)
    @test all([r == e for (r, e) in zip(result, expected_result)])
end

# Test for basis function
function test_basis()
    dim = 3
    index = 2
    expected_result = [0.0, 1.0, 0.0]
    result = basis(dim, index)
    @test result == expected_result
end

# Test for SIC function
function test_SIC()
    dim = 2
    expected_result = [
        [
            0.39433756729740616+0.0im 0.14433756729740654+0.1443375672974066im
            0.14433756729740654-0.1443375672974066im 0.10566243270259379+0.0im
        ],
        [
            0.10566243270259379+0.0im 0.14433756729740654-0.1443375672974066im
            0.14433756729740654+0.1443375672974066im 0.39433756729740616+0.0im
        ],
        [
            0.39433756729740616+0.0im -0.14433756729740654-0.1443375672974066im
            -0.14433756729740654+0.1443375672974066im 0.10566243270259379+0.0im
        ],
        [
            0.1056624327025938+0.0im -0.14433756729740652+0.1443375672974066im
            -0.14433756729740652-0.1443375672974066im 0.3943375672974062+0.0im
        ],
    ]
    result = SIC(dim)
    @test result ≈ expected_result
end

# Run the tests
function test_Common()
    @testset "suN_generatorU" begin
        test_suN_generatorU()
    end

    @testset "suN_generatorV" begin
        test_suN_generatorV()
    end

    @testset "suN_generatorW" begin
        test_suN_generatorW()
    end

    @testset "suN_generator" begin
        test_suN_generator()
    end

    @testset "basis" begin
        test_basis()
    end

    @testset "SIC" begin
        test_SIC()
    end
end

# Call the test function
test_Common()
