using QuanEstimationBase: NM

# Test the update! function
@testset "update!" begin
    # Create test data
    alg = NM(; p_num = 10, max_episode = 1000, ar = 1.0, ae = 2.0, ac = 0.5, as0 = 0.5)
end
