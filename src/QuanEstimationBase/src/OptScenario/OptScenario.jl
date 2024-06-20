abstract type AbstractOpt end

abstract type AbstractMeasurementType end
abstract type Projection <: AbstractMeasurementType end
abstract type LC <: AbstractMeasurementType end
abstract type Rotation <: AbstractMeasurementType end

abstract type Opt <: AbstractOpt end

"""
$(TYPEDEF)

### Fields
- `ctrl`: Guessed control coefficients.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `rng`: Random number generator.
"""
mutable struct ControlOpt <: Opt
	ctrl::Union{AbstractVector, Missing}
	ctrl_bound::AbstractVector
	rng::AbstractRNG
end

"""
$TYPEDSIGNATURES
	
Control optimization.
- `ctrl`: Guessed control coefficients.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
ControlOpt(;ctrl=missing, ctrl_bound=[-Inf, Inf], seed=1234) = ControlOpt(ctrl, ctrl_bound, MersenneTwister(seed))

Copt = ControlOpt
ControlOpt(ctrl::Matrix{R}, ctrl_bound::AbstractVector) where R<:Number = ControlOpt([c[:] for c in eachrow(ctrl)], ctrl_bound)

"""
$(TYPEDEF)

### Fields
- `psi`: Guessed probe state.
- `rng`: Random number generator.
"""
mutable struct StateOpt <: Opt
	psi::Union{AbstractVector, Missing}
	rng::AbstractRNG
end

"""
$TYPEDSIGNATURES
	
State optimization.
- `psi`: Guessed probe state.
- `seed`: Random seed.
"""
StateOpt(;psi=missing, seed=1234) = StateOpt(psi, MersenneTwister(seed))

Sopt = StateOpt

abstract type AbstractMopt <: Opt end

"""
$(TYPEDEF)

### Fields
- `M`: A set of orthonormal basis.
- `rng`: Random number generator.
"""
mutable struct Mopt_Projection <: AbstractMopt
	M::Union{AbstractVector, Missing}
	rng::AbstractRNG
end

Mopt_Projection(;M=missing, seed=1234) = Mopt_Projection(M, MersenneTwister(seed))

"""
$(TYPEDEF)

### Fields
- `B`: Guessed combination coefficient.
- 'POVM_basis': A set of POVM.
- 'M_num': Number of operators of the output measurement.
- `rng`: Random number generator.
"""
mutable struct Mopt_LinearComb <: AbstractMopt
	B::Union{AbstractVector, Missing}
	POVM_basis::Union{AbstractVector, Missing}
	M_num::Int
	rng::AbstractRNG
end

Mopt_LinearComb(;B=missing, POVM_basis=missing, M_num=1, seed=1234) = Mopt_LinearComb(B, POVM_basis, M_num, MersenneTwister(seed))

"""
$(TYPEDEF)

### Fields
- `s`: A set of real number between 0 and 2pi.
- 'POVM_basis': A set of POVM.
- 'Lambda': a srt of SU(N) generator.
- `rng`: Random number generator.
"""
mutable struct Mopt_Rotation <: AbstractMopt
	s::Union{AbstractVector, Missing}
	POVM_basis::Union{AbstractVector, Missing}
	Lambda::Union{AbstractVector, Missing}
	rng::AbstractRNG
end

Mopt_Rotation(;s=missing, POVM_basis=missing, Lambda=missing, seed=1234) = Mopt_Rotation(s, POVM_basis, Lambda, MersenneTwister(seed))


"""
$TYPEDSIGNATURES

Measurement optimization.
- `mtype`: The type of scenarios for the measurement optimization. Options are `:Projection` (default), `:LC` and `:Rotation`.
- `kwargs...`: keywords and the correponding default vaules. `mtype=:Projection`, `mtype=:LC` and `mtype=:Rotation`, the `kwargs...` are `M=missing`, `B=missing, POVM_basis=missing`, and `s=missing, POVM_basis=missing`, respectively.
"""
function MeasurementOpt(;mtype=:Projection, kwargs...)
	if mtype==:Projection
		return Mopt_Projection(;kwargs...)
	elseif mtype==:LC
		return Mopt_LinearComb(;kwargs...)
	elseif mtype==:Rotation
		return Mopt_Rotation(;kwargs...)
	end
end

Mopt = MeasurementOpt

abstract type CompOpt <: Opt end

"""
$(TYPEDEF)
	
#### Fields
- `psi`: Guessed probe state.
- `ctrl`: Guessed control coefficients.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `rng`: Random number generator.
"""
mutable struct StateControlOpt <: CompOpt
	psi::Union{AbstractVector, Missing}
	ctrl::Union{AbstractVector, Missing}
	ctrl_bound::AbstractVector
	rng::AbstractRNG
end

"""
$TYPEDSIGNATURES
	
State and control optimization.
- `psi`: Guessed probe state.
- `ctrl`: Guessed control coefficients.
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
StateControlOpt(;psi=missing, ctrl=missing, ctrl_bound=[-Inf, Inf], seed=1234) = StateControlOpt(psi, ctrl, ctrl_bound, MersenneTwister(seed))

SCopt = StateControlOpt

"""
$(TYPEDEF)

### Fields
- `ctrl`: Guessed control coefficients.
- `M`: Guessed projective measurement (a set of basis)
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `rng`: Random number generator.
"""
mutable struct ControlMeasurementOpt <: CompOpt
	ctrl::Union{AbstractVector, Missing}
	M::Union{AbstractVector, Missing}
	ctrl_bound::AbstractVector
	rng::AbstractRNG
end 

"""
$TYPEDSIGNATURES
	
Control and measurement optimization.
- `ctrl`: Guessed control coefficients.
- `M`: Guessed projective measurement (a set of basis)
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
ControlMeasurementOpt(;ctrl=missing, M=missing, ctrl_bound=[-Inf, Inf], seed=1234) = ControlMeasurementOpt(ctrl, M, ctrl_bound, MersenneTwister(seed))

CMopt = ControlMeasurementOpt

"""
$(TYPEDEF)
	
### Fields
- `psi`: Guessed probe state.
- `M`: Guessed projective measurement (a set of basis).
- `rng`: Random number generator.
"""
mutable struct StateMeasurementOpt <: CompOpt 
	psi::Union{AbstractVector, Missing}
	M::Union{AbstractVector, Missing}
	rng::AbstractRNG
end

"""
$TYPEDSIGNATURES
	
State and measurement optimization.
- `psi`: Guessed probe state.
- `M`: Guessed projective measurement (a set of basis).
- `seed`: Random seed.
"""
StateMeasurementOpt(;psi=missing, M=missing, seed=1234) = StateMeasurementOpt(psi, M, MersenneTwister(seed))

SMopt = StateMeasurementOpt

"""
$(TYPEDEF)
	
### Fields
- `psi`: Guessed probe state.
- `ctrl`: Guessed control coefficients.
- `M`: Guessed projective measurement (a set of basis).
- `ctrl_bound`:  Lower and upper bounds of the control coefficients.
- `rng`: Random number generator.
"""
mutable struct StateControlMeasurementOpt <: CompOpt
	psi::Union{AbstractVector, Missing}
	ctrl::Union{AbstractVector, Missing}
	M::Union{AbstractVector, Missing}
	ctrl_bound::AbstractVector
	rng::AbstractRNG
end

"""
$TYPEDSIGNATURES

State, control and measurement optimization.
- `psi`: Guessed probe state.
- `ctrl`: Guessed control coefficients.
- `M`: Guessed projective measurement (a set of basis).
- `ctrl_bound`: Lower and upper bounds of the control coefficients.
- `seed`: Random seed.
"""
StateControlMeasurementOpt(;psi=missing, ctrl=missing, M=missing, ctrl_bound=[-Inf, Inf], seed=1234) = StateControlMeasurementOpt(psi, ctrl, M, ctrl_bound, MersenneTwister(seed))

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
