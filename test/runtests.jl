using QuanEstimation
using Test
using Trapz
using StableRNGs
using LinearAlgebra
using SparseArrays

@testset verbose = true "Objective Functions" begin
    @testset "Fisher information" begin
        include("objective_functions/test_fisher_information_matrix.jl")
    end
    @testset "Cramer-Rao Bounds" begin
        include("objective_functions/test_cramer_rao_bounds.jl")
    end

    @testset "Analog CramerRao" begin
        include("objective_functions/test_hcrb.jl")
        include("objective_functions/test_nhb.jl")
    end

    @testset "Bayesian Bounds" begin
        include("objective_functions/test_bayesian_cramer_rao.jl")
        include("objective_functions/test_ziv_zakai.jl")
    end
    
    @testset "Objective Wrapper" begin
        include("objective_functions/test_objective_wrapper.jl")
    end
end


@testset verbose = true "Dynamics" begin
    @testset "Lindblad Dynamics" begin
        include("dynamics/test_lindblad.jl")
    end
    @testset "Kraus Dynamics" begin
        include("dynamics/test_kraus.jl")
    end

    @testset "Lindblad Wrapper" begin
        include("dynamics/test_lindblad_wrapper.jl")
    end
    
end


@testset "Algorithm" begin
    @testset "AD" begin
        include("algorithms/test_AD.jl")
    end

    @testset "DE" begin
        include("algorithms/test_DE.jl")
    end

    @testset "NM" begin
        include("algorithms/test_NM.jl")
    end

    @testset "PSO" begin
        include("algorithms/test_PSO.jl")
    end

    @testset "DDPG" begin
        include("algorithms/test_DDPG.jl")
    end
end

@testset "Utils" begin
    # @testset "mintime" begin
    #     include("test_mintime.jl")
    # end
    @testset "SIC" begin
        include("test_sic.jl")
    end
end


@testset "Optimization" begin
    @testset "Optimization Scenario" begin
        include("optimization/test_opt_scenario.jl")
    end
end


# Adaptive Estimation 
@testset "Adaptive" begin
    # @testset "Adaptive Estimation" begin
    #     include("adaptive/test_adaptive_estimation.jl")
    # end

    @testset "Adaptive Estimation MZI" begin
        include("adaptive/test_adaptive_estimation_MZI.jl")
    end
end


@testset "BayesEstimation" begin
    @testset "Bayesian Estimation" begin
        include("bayesian/test_bayesian_estimation.jl")
    end
end