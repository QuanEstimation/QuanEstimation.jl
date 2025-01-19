using QuanEstimationBase
using Test
using Trapz
using StableRNGs
using LinearAlgebra
using SparseArrays

include("utils.jl")

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
end


@testset verbose = true "Dynamics" begin
    @testset "Lindblad Dynamics" begin
        include("dynamics/test_lindblad.jl")
    end
    @testset "Kraus Dynamics" begin
        include("dynamics/test_kraus.jl")
    end

    @testset "Lindblad Wrapper" begin
        include("dynamics/test_init_scheme.jl")
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
end

@testset "Utils" begin
    @testset "SIC" begin
        include("test_sic.jl")
    end
end