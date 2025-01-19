abstract type AbstractOpt end

abstract type AbstractMeasurementType end
abstract type Projection <: AbstractMeasurementType end
abstract type LC <: AbstractMeasurementType end
abstract type Rotation <: AbstractMeasurementType end

abstract type Opt <: AbstractOpt end

mutable struct ControlOpt <: Opt
    ctrl::Union{AbstractVector,Nothing}
    ctrl_bound::AbstractVector
    rng::AbstractRNG
end

"""

	ControlOpt(ctrl=nothing, ctrl_bound=[-Inf, Inf], seed=1234)
	
Control optimization.
- `ctrl`: Guessed control coefficients.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
ControlOpt(; ctrl = nothing, ctrl_bound = [-Inf, Inf], seed = 1234) =
    ControlOpt(ctrl, ctrl_bound, MersenneTwister(seed))

Copt = ControlOpt
ControlOpt(ctrl::Matrix{R}, ctrl_bound::AbstractVector) where {R<:Number} =
    ControlOpt([c[:] for c in eachrow(ctrl)], ctrl_bound)

mutable struct StateOpt <: Opt
    psi::Union{AbstractVector,Nothing}
    rng::AbstractRNG
end

"""

	StateOpt(psi=nothing, seed=1234)
	
State optimization.
- `psi`: Guessed probe state.
- `seed`: Random seed.
"""
StateOpt(; psi = nothing, seed = 1234) = StateOpt(psi, MersenneTwister(seed))

Sopt = StateOpt

abstract type AbstractMopt <: Opt end

mutable struct Mopt_Projection <: AbstractMopt
    M::Union{AbstractVector,Nothing}
    rng::AbstractRNG
end

Mopt_Projection(; M = nothing, seed = 1234) = Mopt_Projection(M, MersenneTwister(seed))

mutable struct Mopt_LinearComb <: AbstractMopt
    B::Union{AbstractVector,Nothing}
    POVM_basis::Union{AbstractVector,Nothing}
    M_num::Int
    rng::AbstractRNG
end

Mopt_LinearComb(; B = nothing, POVM_basis = nothing, M_num = 1, seed = 1234) =
    Mopt_LinearComb(B, POVM_basis, M_num, MersenneTwister(seed))

mutable struct Mopt_Rotation <: AbstractMopt
    s::Union{AbstractVector,Nothing}
    POVM_basis::Union{AbstractVector,Nothing}
    Lambda::Union{AbstractVector,Nothing}
    rng::AbstractRNG
end

Mopt_Rotation(; s = nothing, POVM_basis = nothing, Lambda = nothing, seed = 1234) =
    Mopt_Rotation(s, POVM_basis, Lambda, MersenneTwister(seed))


"""

	MeasurementOpt(mtype=:Projection, kwargs...)
	
Measurement optimization.
- `mtype`: The type of scenarios for the measurement optimization. Options are `:Projection` (default), `:LC` and `:Rotation`.
- `kwargs...`: keywords and the correponding default vaules. `mtype=:Projection`, `mtype=:LC` and `mtype=:Rotation`, the `kwargs...` are `M=nothing`, `B=nothing, POVM_basis=nothing`, and `s=nothing, POVM_basis=nothing`, respectively.
"""
function MeasurementOpt(; mtype = :Projection, kwargs...)
    if mtype == :Projection
        return Mopt_Projection(; kwargs...)
    elseif mtype == :LC
        return Mopt_LinearComb(; kwargs...)
    elseif mtype == :Rotation
        return Mopt_Rotation(; kwargs...)
    end
end

Mopt = MeasurementOpt

abstract type CompOpt <: Opt end

mutable struct StateControlOpt <: CompOpt
    psi::Union{AbstractVector,Nothing}
    ctrl::Union{AbstractVector,Nothing}
    ctrl_bound::AbstractVector
    rng::AbstractRNG
end

StateControlOpt(; psi = nothing, ctrl = nothing, ctrl_bound = [-Inf, Inf], seed = 1234) =
    StateControlOpt(psi, ctrl, ctrl_bound, MersenneTwister(seed))

"""

	SCopt(psi=nothing, ctrl=nothing, ctrl_bound=[-Inf, Inf], seed=1234)
	
State and control optimization.
- `psi`: Guessed probe state.
- `ctrl`: Guessed control coefficients.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
SCopt = StateControlOpt

mutable struct ControlMeasurementOpt <: CompOpt
    ctrl::Union{AbstractVector,Nothing}
    M::Union{AbstractVector,Nothing}
    ctrl_bound::AbstractVector
    rng::AbstractRNG
end

ControlMeasurementOpt(;
    ctrl = nothing,
    M = nothing,
    ctrl_bound = [-Inf, Inf],
    seed = 1234,
) = ControlMeasurementOpt(ctrl, M, ctrl_bound, MersenneTwister(seed))

"""

	CMopt(ctrl=nothing, M=nothing, ctrl_bound=[-Inf, Inf], seed=1234)
	
Control and measurement optimization.
- `ctrl`: Guessed control coefficients.
- `M`: Guessed projective measurement (a set of basis)
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
CMopt = ControlMeasurementOpt

mutable struct StateMeasurementOpt <: CompOpt
    psi::Union{AbstractVector,Nothing}
    M::Union{AbstractVector,Nothing}
    rng::AbstractRNG
end

StateMeasurementOpt(; psi = nothing, M = nothing, seed = 1234) =
    StateMeasurementOpt(psi, M, MersenneTwister(seed))
"""

	SMopt(psi=nothing, M=nothing, seed=1234)
	
State and control optimization.
- `psi`: Guessed probe state.
- `M`: Guessed projective measurement (a set of basis).
- `seed`: Random seed.
"""
SMopt = StateMeasurementOpt

mutable struct StateControlMeasurementOpt <: CompOpt
    psi::Union{AbstractVector,Nothing}
    ctrl::Union{AbstractVector,Nothing}
    M::Union{AbstractVector,Nothing}
    ctrl_bound::AbstractVector
    rng::AbstractRNG
end

StateControlMeasurementOpt(;
    psi = nothing,
    ctrl = nothing,
    M = nothing,
    ctrl_bound = [-Inf, Inf],
    seed = 1234,
) = StateControlMeasurementOpt(psi, ctrl, M, ctrl_bound, MersenneTwister(seed))

"""

	SCMopt(psi=nothing, ctrl=nothing, M=nothing, ctrl_bound=[-Inf, Inf], seed=1234)
	
State, control and measurement optimization.
- `psi`: Guessed probe state.
- `ctrl`: Guessed control coefficients.
- `M`: Guessed projective measurement (a set of basis).
- `ctrl_bound`:  Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
SCMopt = StateControlMeasurementOpt

opt_target(::ControlOpt) = :Copt
opt_target(::StateOpt) = :Sopt
opt_target(::Mopt_Projection) = :Mopt
opt_target(::Mopt_LinearComb) = :Mopt_input
opt_target(::Mopt_Rotation) = :Mopt_input
opt_target(::CompOpt) = :CompOpt
opt_target(::StateControlOpt) = :SCopt
opt_target(::ControlMeasurementOpt) = :CMopt
opt_target(::StateMeasurementOpt) = :SMopt
opt_target(::StateControlMeasurementOpt) = :SCMopt

result(opt::ControlOpt) = [opt.ctrl]
result(opt::StateOpt) = [opt.psi]
result(opt::Mopt_Projection) = [opt.M]
result(opt::Mopt_LinearComb) = [opt.B, opt.POVM_basis, opt.M_num]
result(opt::Mopt_Rotation) = [opt.s]
result(opt::StateControlOpt) = [opt.psi, opt.ctrl]
result(opt::ControlMeasurementOpt) = [opt.ctrl, opt.M]
result(opt::StateMeasurementOpt) = [opt.psi, opt.M]
result(opt::StateControlMeasurementOpt) = [opt.psi, opt.ctrl, opt.M]

#with reward
result(opt, ::Type{Val{:save_reward}}) = [result(opt)..., [0.0]]

const res_file_name = Dict(
    :Copt => ["controls"],
    :Sopt => ["states"],
    :Mopt => ["measurements"],
    :Mopt_input => ["measurements"],
    :SCopt => ["states", "controls"],
    :CMopt => ["controls", "measurements"],
    :SMopt => ["states", "measurements"],
    :SCMopt => ["states", "controls", "measurements"],
)

res_file(opt::AbstractOpt) = res_file_name[opt_target(opt)]

function init_opt(opt::ControlOpt, scheme)
    pdata = param_data(scheme)
    ctrl_num = get_ctrl_num(scheme)
    tnum = length(pdata.tspan)

    if isnothing(opt.ctrl)
        ctrl = [zeros(tnum - 1) for _ = 1:ctrl_num]
        opt.ctrl = ctrl

    else
        ctrl_length = length(opt.ctrl)
        if ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences",
                ),
            )
        elseif ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.",
                ),
            )
        end

        if (length(pdata.tspan) - 1) % length(opt.ctrl[1]) != 0
            ratio_num = ceil((length(pdata.tspan) - 1) / length(opt.ctrl[1]))
            tnum = ratio_num * length(opt.ctrl[1]) |> Int
            pdata.tspan = range(pdata.tspan[1], pdata.tspan[end], length = tnum + 1)
        end
    end
    return opt
end

function init_opt(opt::StateOpt, scheme)
    dim = get_dim(scheme)

    if isnothing(opt.psi)
        r_ini = 2 * rand(opt.rng, dim) - ones(dim)
        r = r_ini ./ norm(r_ini)
        ϕ = 2pi * rand(opt.rng, dim)
        psi = [r * exp(im * ϕ) for (r, ϕ) in zip(r, ϕ)]
        opt.psi = psi
    end

    return opt
end

function init_opt(opt::Mopt_LinearComb, scheme; eps = GLOBAL_EPS)
    (; B, POVM_basis, M_num) = opt
    dim = get_dim(scheme)
    if isnothing(POVM_basis)
        opt.POVM_basis = SIC(dim)
    else
        ## TODO: accuracy ?
        for P in POVM_basis
            if minimum(eigvals(P)) < (-eps)
                throw(ArgumentError("The given POVMs should be semidefinite!"))
            end
        end
        if !(sum(POVM_basis) ≈ I(dim))
            throw(ArgumentError("The sum of the given POVMs should be identity matrix!"))
        end
    end

    if isnothing(B)
        opt.B = [rand(opt.rng, length(opt.POVM_basis)) for _ = 1:M_num]
    end
    return opt
end

function init_opt(opt::Mopt_Projection, scheme)
    dim = get_dim(scheme)
    (; M) = opt
    ## initialize the Mopt target M
    C = [ComplexF64[] for _ = 1:dim]
    if isnothing(M)
        for i = 1:dim
            r_ini = 2 * rand(opt.rng, dim) - ones(dim)
            r = r_ini / norm(r_ini)
            ϕ = 2pi * rand(opt.rng, dim)
            C[i] = [r * exp(im * ϕ) for (r, ϕ) in zip(r, ϕ)]
        end
        opt.M = gramschmidt(C)
    end
    return opt
end

function init_opt(opt::Mopt_Rotation, scheme; eps = GLOBAL_EPS)
    dim = get_dim(scheme)

    (; s, POVM_basis) = opt
    if isnothing(POVM_basis)
        throw(ArgumentError("The initial POVM basis should not be empty!"))
    else
        ## TODO: accuracy ?
        for P in POVM_basis
            if minimum(eigvals(P)) < -eps
                throw(ArgumentError("The given POVMs should be semidefinite!"))
            end
        end
        if !(sum(POVM_basis) ≈ I(dim))
            throw(ArgumentError("The sum of the given POVMs should be identity matrix!"))
        end
    end

    if isnothing(s)
        opt.s = rand(opt.rng, dim^2)
    end
    return opt
end

function init_opt(opt::SCopt, scheme; eps = GLOBAL_EPS)
    dim = get_dim(scheme)
    pdata = param_data(scheme)
    ctrl_num = get_ctrl_num(scheme)
    tnum = length(pdata.tspan)

    if isnothing(opt.psi)
        r_ini = 2 * rand(opt.rng, dim) - ones(dim)
        r = r_ini ./ norm(r_ini)
        ϕ = 2pi * rand(opt.rng, dim)
        psi = [r * exp(im * ϕ) for (r, ϕ) in zip(r, ϕ)]
        opt.psi = psi
    end

    if isnothing(opt.ctrl)
        ctrl = [zeros(tnum - 1) for _ = 1:ctrl_num]
        opt.ctrl = ctrl
    else
        ctrl_length = length(opt.ctrl)
        if ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences",
                ),
            )
        elseif ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.",
                ),
            )
        end

        if (length(pdata.tspan) - 1) % length(opt.ctrl[1]) != 0
            ratio_num = ceil((length(pdata.tspan) - 1) / length(opt.ctrl[1]))
            tnum = ratio_num * length(opt.ctrl[1]) |> Int
            pdata.tspan = range(pdata.tspan[1], pdata.tspan[end], length = tnum + 1)
        end
    end

    return opt
end

function init_opt(opt::CMopt, scheme; eps = GLOBAL_EPS)
    dim = get_dim(scheme)
    pdata = param_data(scheme)
    ctrl_num = get_ctrl_num(scheme)
    tnum = length(pdata.tspan)

    if isnothing(opt.ctrl)
        ctrl = [zeros(tnum - 1) for _ = 1:ctrl_num]
        opt.ctrl = ctrl

    else
        ctrl_length = length(opt.ctrl)
        if ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences",
                ),
            )
        elseif ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.",
                ),
            )
        end

        if (length(pdata.tspan) - 1) % length(opt.ctrl[1]) != 0
            ratio_num = ceil((length(pdata.tspan) - 1) / length(opt.ctrl[1]))
            tnum = ratio_num * length(opt.ctrl[1]) |> Int
            pdata.tspan = range(pdata.tspan[1], pdata.tspan[end], length = tnum + 1)
        end
    end

    if isnothing(opt.M)
        C = [ComplexF64[] for _ = 1:dim]
        for i = 1:dim
            r_ini = 2 * rand(opt.rng, dim) - ones(dim)
            r = r_ini / norm(r_ini)
            ϕ = 2pi * rand(opt.rng, dim)
            C[i] = [r * exp(im * ϕ) for (r, ϕ) in zip(r, ϕ)]
        end
        opt.M = gramschmidt(C)
    end
    return opt
end

function init_opt(opt::SMopt, scheme; eps = GLOBAL_EPS)
    dim = get_dim(scheme)

    if isnothing(opt.psi)
        r_ini = 2 * rand(opt.rng, dim) - ones(dim)
        r = r_ini ./ norm(r_ini)
        ϕ = 2pi * rand(opt.rng, dim)
        psi = [r * exp(im * ϕ) for (r, ϕ) in zip(r, ϕ)]
        opt.psi = psi
    end

    if isnothing(opt.M)
        C = [ComplexF64[] for _ = 1:dim]
        for i = 1:dim
            r_ini = 2 * rand(opt.rng, dim) - ones(dim)
            r = r_ini / norm(r_ini)
            ϕ = 2pi * rand(opt.rng, dim)
            C[i] = [r * exp(im * ϕ) for (r, ϕ) in zip(r, ϕ)]
        end
        opt.M = gramschmidt(C)
    end
    return opt
end

function init_opt(opt::SCMopt, scheme; eps = GLOBAL_EPS)
    dim = get_dim(scheme)
    pdata = param_data(scheme)
    ctrl_num = get_ctrl_num(scheme)
    tnum = length(pdata.tspan)

    if isnothing(opt.psi)
        r_ini = 2 * rand(opt.rng, dim) - ones(dim)
        r = r_ini ./ norm(r_ini)
        ϕ = 2pi * rand(opt.rng, dim)
        psi = [r * exp(im * ϕ) for (r, ϕ) in zip(r, ϕ)]
        opt.psi = psi
    end

    if isnothing(opt.ctrl)
        ctrl = [zeros(tnum - 1) for _ = 1:ctrl_num]
        opt.ctrl = ctrl
    else
        ctrl_length = length(opt.ctrl)
        if ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "There are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences: too many coefficients sequences",
                ),
            )
        elseif ctrl_num < ctrl_length
            throw(
                ArgumentError(
                    "Not enough coefficients sequences: there are $ctrl_num control Hamiltonians but $ctrl_length coefficients sequences. The rest of the control sequences are set to be 0.",
                ),
            )
        end

        if (length(pdata.tspan) - 1) % length(opt.ctrl[1]) != 0
            ratio_num = ceil((length(pdata.tspan) - 1) / length(opt.ctrl[1]))
            tnum = ratio_num * length(opt.ctrl[1]) |> Int
            pdata.tspan = range(pdata.tspan[1], pdata.tspan[end], length = tnum + 1)
        end
    end

    if isnothing(opt.M)
        C = [ComplexF64[] for _ = 1:dim]
        for i = 1:dim
            r_ini = 2 * rand(opt.rng, dim) - ones(dim)
            r = r_ini / norm(r_ini)
            ϕ = 2pi * rand(opt.rng, dim)
            C[i] = [r * exp(im * ϕ) for (r, ϕ) in zip(r, ϕ)]
        end
        opt.M = gramschmidt(C)
    end
    return opt
end

include("optimize.jl")
