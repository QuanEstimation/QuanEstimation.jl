# Numerical backport oracles: the optimized kernels must preserve the legacy
# arithmetic, not merely agree to a loose numerical tolerance.

@testset "MZI sparse refill is bitwise identical to a_u" begin
    for F in (Float64, Float32), phi in (-0.2, 0.4), u in (0, 1)
        a = Complex{F}.(QEB.destroy(4)) |> sparse
        ex = QEB._mzi_exec_bake(a)
        Au = QEB._mzi_au_buffer(ex)
        QEB._mzi_fill_a_u!(Au, ex, 0.3, phi, u)
        reference = QEB.a_u(a, 0.3, phi, u)
        @test Au.colptr == reference.colptr
        @test Au.rowval == reference.rowval
        @test all(i -> Au.nzval[i] === reference.nzval[i], eachindex(Au.nzval))
    end
end

@testset "ODE in-place commutator is bitwise identical" begin
    H = ComplexF64[0.2 0.1 + 0.3im; -0.1im 0.7]
    ρ = ComplexF64[0.8 0.2 - 0.1im; 0.1 + 0.2im 0.2]
    left, right = similar(ρ), similar(ρ)
    result = QEB._ode_neg_im_comm!(left, right, H, ρ)
    reference = -im * (H * ρ - ρ * H)
    @test all(i -> result[i] === reference[i], eachindex(result))
end
