"""
    AbstractOpt

Abstract supertype for optimization scenarios.
"""
abstract type AbstractOpt end

"""
    Opt <: AbstractOpt

Abstract supertype for concrete optimization scenarios.
raw"""
abstract type Opt <: AbstractOpt end

@doc raw"""
    ControlOpt <: Opt

Control optimization scenario.

Holds the control variables, bounds ``[c_{\min}, c_{\max}]``, and RNG.

# Fields

- `ctrl::Union{AbstractVector,Nothing}`: Initial control coefficients.
- `ctrl_bound::AbstractVector`: Lower and upper bounds.
- `rng::AbstractRNG`: Random number generator.
"""
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

raw"""

    ControlOpt(ctrl::Matrix, ctrl_bound::AbstractVector)

Construct a `ControlOpt` from a matrix of control coefficients (one row per control Hamiltonian). Each row is flattened into a control sequence vector.
raw"""
ControlOpt(ctrl::Matrix{R}, ctrl_bound::AbstractVector) where {R<:Number} =
    ControlOpt([c[:] for c in eachrow(ctrl)], ctrl_bound)

@doc raw"""
    StateOpt <: Opt

State optimization scenario. Holds an initial guess ``|\psi_0\rangle`` and RNG.

# Fields

- `psi::Union{AbstractVector,Nothing}`: Initial probe state guess.
- `rng::AbstractRNG`: Random number generator.
"""
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

"""
    AbstractMopt <: Opt

Abstract supertype for measurement optimization scenarios.
"""
abstract type AbstractMopt <: Opt end

"""
    Mopt_Projection <: AbstractMopt

Projective measurement optimization scenario. Optimizes a set of orthonormal basis vectors defining a projective measurement.

# Fields

- `M::Union{AbstractVector,Nothing}`: Projective measurement basis (a set of orthonormal vectors).
- `rng::AbstractRNG`: Random number generator.
"""
mutable struct Mopt_Projection <: AbstractMopt
    M::Union{AbstractVector,Nothing}
    rng::AbstractRNG
end

"""
    Mopt_Projection(; M=nothing, seed=1234)

Projective measurement optimization.
- `M`: Guessed projective measurement basis.
- `seed`: Random seed.
"""
Mopt_Projection(; M = nothing, seed = 1234) = Mopt_Projection(M, MersenneTwister(seed))

"""
    Mopt_LinearComb <: AbstractMopt

Linear combination measurement optimization scenario. Expresses each measurement operator as a linear combination of POVM basis elements, optimizing the combination coefficients.

# Fields

- `B::Union{AbstractVector,Nothing}`: Linear combination coefficients.
- `POVM_basis::Union{AbstractVector,Nothing}`: POVM basis elements for the linear combination.
- `M_num::Int`: Number of measurement operators.
- `rng::AbstractRNG`: Random number generator.
"""
mutable struct Mopt_LinearComb <: AbstractMopt
    B::Union{AbstractVector,Nothing}
    POVM_basis::Union{AbstractVector,Nothing}
    M_num::Int
    rng::AbstractRNG
end

"""
    Mopt_LinearComb(; B=nothing, POVM_basis=nothing, M_num=1, seed=1234)

Linear combination measurement optimization.
- `B`: Guessed linear combination coefficients.
- `POVM_basis`: POVM basis elements.
- `M_num`: Number of measurement operators.
- `seed`: Random seed.
"""
Mopt_LinearComb(; B = nothing, POVM_basis = nothing, M_num = 1, seed = 1234) =
    Mopt_LinearComb(B, POVM_basis, M_num, MersenneTwister(seed))

"""
    Mopt_Rotation <: AbstractMopt

Rotation-based measurement optimization scenario. Optimizes rotation parameters applied to a fixed POVM basis to produce the final measurement.

# Fields

- `s::Union{AbstractVector,Nothing}`: Rotation parameters.
- `POVM_basis::Union{AbstractVector,Nothing}`: Fixed POVM basis to be rotated.
- `Lambda::Union{AbstractVector,Nothing}`: Eigenvalues for the rotation generator.
- `rng::AbstractRNG`: Random number generator.
"""
mutable struct Mopt_Rotation <: AbstractMopt
    s::Union{AbstractVector,Nothing}
    POVM_basis::Union{AbstractVector,Nothing}
    Lambda::Union{AbstractVector,Nothing}
    rng::AbstractRNG
end

"""
    Mopt_Rotation(; s=nothing, POVM_basis=nothing, Lambda=nothing, seed=1234)

Rotation-based measurement optimization.
- `s`: Guessed rotation parameters.
- `POVM_basis`: POVM basis to rotate.
- `Lambda`: Eigenvalues for the rotation generator.
- `seed`: Random seed.
"""
Mopt_Rotation(; s = nothing, POVM_basis = nothing, Lambda = nothing, seed = 1234) =
    Mopt_Rotation(s, POVM_basis, Lambda, MersenneTwister(seed))


"""

	MeasurementOpt(mtype=:Projection, kwargs...)
	
Measurement optimization.
- `mtype`: The type of scenarios for the measurement optimization. Options are `:Projection` (default), `:LC` and `:Rotation`.
- `kwargs...`: keywords and the correponding default vaules. `mtype=:Projection`, `mtype=:LC` and `mtype=:Rotation`, 
   the `kwargs...` are `M=nothing`, `B=nothing, POVM_basis=nothing`, and `s=nothing, POVM_basis=nothing`, respectively.
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

"""
    CompOpt <: Opt

Abstract supertype for composite optimization scenarios that combine two or more of state, control, and measurement optimization.
"""
abstract type CompOpt <: Opt end

"""
    StateControlOpt <: CompOpt

Simultaneous state and control optimization scenario.

# Fields

- `psi::Union{AbstractVector,Nothing}`: Initial probe state guess.
- `ctrl::Union{AbstractVector,Nothing}`: Control coefficients.
- `ctrl_bound::AbstractVector`: Lower and upper bounds of the control coefficients.
- `rng::AbstractRNG`: Random number generator.
"""
mutable struct StateControlOpt <: CompOpt
    psi::Union{AbstractVector,Nothing}
    ctrl::Union{AbstractVector,Nothing}
    ctrl_bound::AbstractVector
    rng::AbstractRNG
end

"""
    StateControlOpt(; psi=nothing, ctrl=nothing, ctrl_bound=[-Inf,Inf], seed=1234)

State and control optimization.
- `psi`: Guessed probe state.
- `ctrl`: Guessed control coefficients.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
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

"""
    ControlMeasurementOpt <: CompOpt

Simultaneous control and measurement optimization scenario.

# Fields

- `ctrl::Union{AbstractVector,Nothing}`: Control coefficients.
- `M::Union{AbstractVector,Nothing}`: Projective measurement basis.
- `ctrl_bound::AbstractVector`: Lower and upper bounds of the control coefficients.
- `rng::AbstractRNG`: Random number generator.
"""
mutable struct ControlMeasurementOpt <: CompOpt
    ctrl::Union{AbstractVector,Nothing}
    M::Union{AbstractVector,Nothing}
    ctrl_bound::AbstractVector
    rng::AbstractRNG
end

"""
    ControlMeasurementOpt(; ctrl=nothing, M=nothing, ctrl_bound=[-Inf,Inf], seed=1234)

Control and measurement optimization.
- `ctrl`: Guessed control coefficients.
- `M`: Guessed projective measurement basis.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
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

"""
    StateMeasurementOpt <: CompOpt

Simultaneous state and measurement optimization scenario.

# Fields

- `psi::Union{AbstractVector,Nothing}`: Initial probe state guess.
- `M::Union{AbstractVector,Nothing}`: Projective measurement basis.
- `rng::AbstractRNG`: Random number generator.
"""
mutable struct StateMeasurementOpt <: CompOpt
    psi::Union{AbstractVector,Nothing}
    M::Union{AbstractVector,Nothing}
    rng::AbstractRNG
end

"""
    StateMeasurementOpt(; psi=nothing, M=nothing, seed=1234)

State and measurement optimization.
- `psi`: Guessed probe state.
- `M`: Guessed projective measurement basis.
- `seed`: Random seed.
"""
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

"""
    StateControlMeasurementOpt <: CompOpt

Simultaneous state, control, and measurement optimization scenario.

# Fields

- `psi::Union{AbstractVector,Nothing}`: Initial probe state guess.
- `ctrl::Union{AbstractVector,Nothing}`: Control coefficients.
- `M::Union{AbstractVector,Nothing}`: Projective measurement basis.
- `ctrl_bound::AbstractVector`: Lower and upper bounds of the control coefficients.
- `rng::AbstractRNG`: Random number generator.
"""
mutable struct StateControlMeasurementOpt <: CompOpt
    psi::Union{AbstractVector,Nothing}
    ctrl::Union{AbstractVector,Nothing}
    M::Union{AbstractVector,Nothing}
    ctrl_bound::AbstractVector
    rng::AbstractRNG
end

"""
    StateControlMeasurementOpt(; psi=nothing, ctrl=nothing, M=nothing, ctrl_bound=[-Inf,Inf], seed=1234)

State, control, and measurement optimization.
- `psi`: Guessed probe state.
- `ctrl`: Guessed control coefficients.
- `M`: Guessed projective measurement basis.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
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

"""
    opt_target(opt)

Return the optimization target symbol for `opt`. This symbol identifies the type of optimization scenario and is used to dispatch optimization algorithms and label result files.

Returns `:Copt` (control only), `:Sopt` (state only), `:Mopt` (projective measurement), `:Mopt_input` (linear-combination or rotation measurement), `:SCopt` (state + control), `:CMopt` (control + measurement), `:SMopt` (state + measurement), or `:SCMopt` (state + control + measurement).
"""
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

"""
    result(opt)

Extract the optimized variables from `opt` as a vector. The contents depend on the scenario type:
- `ControlOpt` → `[ctrl]`
- `StateOpt` → `[psi]`
- `Mopt_Projection` → `[M]`
- `Mopt_LinearComb` → `[B, POVM_basis, M_num]`
- `Mopt_Rotation` → `[s]`
- `StateControlOpt` → `[psi, ctrl]`
- `ControlMeasurementOpt` → `[ctrl, M]`
- `StateMeasurementOpt` → `[psi, M]`
- `StateControlMeasurementOpt` → `[psi, ctrl, M]`
"""
result(opt::ControlOpt) = [opt.ctrl]
result(opt::StateOpt) = [opt.psi]
result(opt::Mopt_Projection) = [opt.M]
result(opt::Mopt_LinearComb) = [opt.B, opt.POVM_basis, opt.M_num]
result(opt::Mopt_Rotation) = [opt.s]
result(opt::StateControlOpt) = [opt.psi, opt.ctrl]
result(opt::ControlMeasurementOpt) = [opt.ctrl, opt.M]
result(opt::StateMeasurementOpt) = [opt.psi, opt.M]
result(opt::StateControlMeasurementOpt) = [opt.psi, opt.ctrl, opt.M]

"""
    result(opt, ::Type{Val{:save_reward}})

Same as `result(opt)` but appends a reward entry initialized to `0.0` for tracking optimization reward values.
"""
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

"""
    res_file(opt::AbstractOpt)

Return the list of file name components (e.g., `[\"states\", \"controls\"]`) for saving optimization results, based on the target symbol of `opt`.
"""
res_file(opt::AbstractOpt) = res_file_name[opt_target(opt)]

"""
    init_opt(opt::ControlOpt, scheme)

Initialize the control coefficients. If `opt.ctrl` is `nothing`, a zero-initialized sequence is created for each control Hamiltonian. Otherwise, validates that the number of coefficient sequences matches the number of control Hamiltonians and adjusts the time span if needed.
"""
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
        elseif ctrl_length < ctrl_num
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

"""
    init_opt(opt::StateOpt, scheme)

Initialize the probe state. If `opt.psi` is `nothing`, a random normalized pure state is generated.
"""
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

"""
    init_opt(opt::Mopt_LinearComb, scheme; eps=GLOBAL_EPS)

Initialize the POVM basis and linear combination coefficients. If `POVM_basis` is `nothing`, defaults to a SIC-POVM. If `B` is `nothing`, random coefficients are generated. Validates that the given POVM basis elements are positive semidefinite and sum to the identity.
"""
function init_opt(opt::Mopt_LinearComb, scheme; eps = GLOBAL_EPS)
    (; B, POVM_basis, M_num) = opt
    dim = get_dim(scheme)
    if isnothing(POVM_basis)
        opt.POVM_basis = SIC(dim)
    else
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

"""
    init_opt(opt::Mopt_Projection, scheme)

Initialize the projective measurement basis. If `opt.M` is `nothing`, a random set of orthonormal vectors is generated via Gram-Schmidt orthogonalization.
"""
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

"""
    init_opt(opt::Mopt_Rotation, scheme; eps=GLOBAL_EPS)

Initialize the rotation parameters for measurement optimization. Validates that the provided POVM basis is complete and positive semidefinite. If `s` is `nothing`, random rotation parameters are generated.
"""
function init_opt(opt::Mopt_Rotation, scheme; eps = GLOBAL_EPS)
    dim = get_dim(scheme)

    (; s, POVM_basis) = opt
    if isnothing(POVM_basis)
        throw(ArgumentError("The initial POVM basis should not be empty!"))
    else
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

"""
    init_opt(opt::StateControlOpt, scheme; eps=GLOBAL_EPS)

Initialize both the probe state and control coefficients. If `psi` is `nothing`, a random pure state is generated. If `ctrl` is `nothing`, zero-initialized control sequences are created. Validates control coefficient count and adjusts time span as needed.
"""
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
        elseif ctrl_length < ctrl_num
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

"""
    init_opt(opt::ControlMeasurementOpt, scheme; eps=GLOBAL_EPS)

Initialize both control coefficients and projective measurement basis. If `ctrl` is `nothing`, zero-initialized sequences are created. If `M` is `nothing`, random orthonormal vectors are generated. Validates control coefficient count and adjusts time span as needed.
"""
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
        elseif ctrl_length < ctrl_num
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

"""
    init_opt(opt::StateMeasurementOpt, scheme; eps=GLOBAL_EPS)

Initialize both the probe state and projective measurement basis. If `psi` is `nothing`, a random pure state is generated. If `M` is `nothing`, random orthonormal vectors are generated.
"""
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

"""
    init_opt(opt::StateControlMeasurementOpt, scheme; eps=GLOBAL_EPS)

Initialize the probe state, control coefficients, and projective measurement basis. If `psi` is `nothing`, a random pure state is generated. If `ctrl` is `nothing`, zero-initialized sequences are created. If `M` is `nothing`, random orthonormal vectors are generated. Validates control coefficient count and adjusts time span as needed.
"""
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
        elseif ctrl_length < ctrl_num
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
