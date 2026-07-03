using Test

# Main test suite for QuanEstimation
@testset "QuanEstimation Tests" begin

    # --- Include sub-package tests ---
    include("../lib/QuanEstimationBase/test/runtests.jl")
    include("../lib/NVMagnetometer/test/runtests.jl")

    # --- Root-level integration tests ---

    # Example system integration tests (sysA-sysE)
    include("example/analytic_reference.jl")

    @testset "Example Systems (Integration)" begin
        @testset "Sys-A: Qubit Rotation" begin
            include("example/sysA_qubit_rotation.jl")
        end
        @testset "Sys-B: Spontaneous Emission" begin
            include("example/sysB_spontaneous_emission.jl")
        end
        @testset "Sys-C: Pure Dephasing" begin
            include("example/sysC_pure_dephasing.jl")
        end
        @testset "Sys-D: XX Coupling" begin
            include("example/sysD_xx_coupling.jl")
        end
        @testset "Sys-E: Williamson" begin
            include("example/sysE_williamson.jl")
        end
        @testset "Sys-1: Magnetic Field" begin
            include("example/sys1_magnetic_field.jl")
        end
    end

    # API compatibility tests
    @testset "API Compatibility" begin
        include("api_compat/run_compat.jl")
    end

end
