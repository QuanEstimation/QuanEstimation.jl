"""
    AbstractLDtype

Abstract supertype for logarithmic derivative type tags (`SLD`, `RLD`, `LLD`).
Used as a type parameter in [`QFIM_obj`](@ref) to dispatch to the correct
QFIM computation.
raw"""
abstract type AbstractLDtype end

raw"""
    SLD <: AbstractLDtype

Tag type for the symmetric logarithmic derivative (SLD). The SLD operator
``L_a`` satisfies ``\frac{1}{2}(L_a\rho + \rho L_a) = \partial_a\rho``.
raw"""
abstract type SLD <: AbstractLDtype end

raw"""
    RLD <: AbstractLDtype

Tag type for the right logarithmic derivative (RLD). The RLD operator
``\mathcal{R}_a`` satisfies ``\partial_a\rho = \rho\mathcal{R}_a``.
raw"""
abstract type RLD <: AbstractLDtype end

raw"""
    LLD <: AbstractLDtype

Tag type for the left logarithmic derivative (LLD). The LLD operator
``\mathcal{L}_a`` satisfies ``\partial_a\rho = \mathcal{L}_a\rho``
and ``\mathcal{L}_a = \mathcal{R}_a^\dagger``.
raw"""
abstract type LLD <: AbstractLDtype end

const PARA_TYPE_MAP = Dict{Symbol,Type{<:AbstractParaType}}(
    :single_para => single_para,
    :multi_para  => multi_para,
)
const LD_TYPE_MAP = Dict{Symbol,Type{<:AbstractLDtype}}(
    :SLD => SLD,
    :RLD => RLD,
    :LLD => LLD,
)

@doc raw"""
    QFIM_obj{P,D} <: AbstractObj

Objective function wrapper for the quantum Fisher information (matrix).

Selects ``\mathrm{Tr}(W F^{-1})`` as the optimization objective, where ``F``
is the QFIM (or QFI for single-parameter) and ``W`` is the weight matrix.

# Type Parameters

- `P`: Parameter type tag (`single_para` or `multi_para`).
- `D`: Logarithmic derivative type tag (`SLD`, `RLD`, or `LLD`).

# Fields

- `W::Union{AbstractMatrix,UniformScaling}`: Weight matrix (defaults to ``\mathbb{I}``).
- `eps::Number`: Numerical epsilon threshold.

# See Also

- [`QFIM_obj`](@ref): Keyword constructor.
- [`CFIM_obj`](@ref): Classical Fisher information objective.
- [`HCRB_obj`](@ref): Holevo Cramér-Rao bound objective.
raw"""
struct QFIM_obj{P,D} <: AbstractObj
    W::Union{AbstractMatrix,UniformScaling}
    eps::Number
end

@doc raw"""
    CFIM_obj{P} <: AbstractObj

Objective function wrapper for the classical Fisher information (matrix).

Selects ``\mathrm{Tr}(W I^{-1})`` as the optimization objective, where ``I``
is the CFIM (or CFI for single-parameter).

# Type Parameters

- `P`: Parameter type tag (`single_para` or `multi_para`).

# Fields

- `M::Union{AbstractVecOrMat,Nothing}`: POVM elements (defaults to SIC-POVM if `nothing`).
- `W::Union{AbstractMatrix,UniformScaling}`: Weight matrix (defaults to ``\mathbb{I}``).
- `eps::Number`: Numerical epsilon threshold.

# See Also

- [`CFIM_obj`](@ref): Keyword constructor.
- [`QFIM_obj`](@ref): Quantum Fisher information objective.
raw"""
struct CFIM_obj{P} <: AbstractObj
    M::Union{AbstractVecOrMat,Nothing}
    W::Union{AbstractMatrix,UniformScaling}
    eps::Number
end

@doc raw"""
    HCRB_obj{P} <: AbstractObj

Objective function wrapper for the Holevo Cramér-Rao bound (HCRB).

Selects the HCRB ``\min\mathrm{Tr}(W V)`` (via SDP) as the optimization
objective.

# Type Parameters

- `P`: Parameter type tag (`single_para` or `multi_para`).

# Fields

- `W::Union{AbstractMatrix,UniformScaling}`: Weight matrix (defaults to ``\mathbb{I}``).
- `eps::Number`: Numerical epsilon threshold.

# See Also

- [`HCRB_obj`](@ref): Keyword constructor.
- [`HCRB`](@ref): HCRB computation.
raw"""
struct HCRB_obj{P} <: AbstractObj
    W::Union{AbstractMatrix,UniformScaling}
    eps::Number
end

@doc raw"""

    QFIM_obj(;W=nothing, eps=GLOBAL_EPS, LDtype::Symbol=:SLD)

Choose QFI [``\mathrm{Tr}(WF^{-1})``] as the objective function with ``W`` the weight matrix and ``F`` the QFIM.
- `W`: Weight matrix.
- `eps`: Machine epsilon.
- `LDtype`: Types of QFI (QFIM) can be set as the objective function. Options are `:SLD` (default), `:RLD` and `:LLD`.
raw"""
QFIM_obj(;
    W = nothing,
    eps = GLOBAL_EPS,
    para_type::Symbol = :single_para,
    LDtype::Symbol = :SLD,
) = QFIM_obj{PARA_TYPE_MAP[para_type], LD_TYPE_MAP[LDtype]}(isnothing(W) ? I : W, eps)

@doc raw"""

    CFIM_obj(;M=nothing, W=nothing, eps=GLOBAL_EPS)

Choose CFI [``\mathrm{Tr}(WI^{-1})``] as the objective function with ``W`` the weight matrix and ``I`` the CFIM.
- `M`: A set of positive operator-valued measure (POVM). The default measurement is a set of rank-one symmetric informationally complete POVM (SIC-POVM).
- `W`: Weight matrix.
- `eps`: Machine epsilon.
"""
CFIM_obj(; M = nothing, W = nothing, eps = GLOBAL_EPS, para_type::Symbol = :single_para) =
    CFIM_obj{PARA_TYPE_MAP[para_type]}(M, isnothing(W) ? I : W, eps)

@doc raw"""

    HCRB_obj(;W=nothing, eps=GLOBAL_EPS)

Choose HCRB as the objective function. 
- `W`: Weight matrix.
- `eps`: Machine epsilon.
"""
HCRB_obj(; W = nothing, eps = GLOBAL_EPS, para_type::Symbol = :single_para) =
    HCRB_obj{PARA_TYPE_MAP[para_type]}(isnothing(W) ? I : W, eps)

"""
    QFIM_obj(W, eps, para_type::Symbol, LDtype::Symbol)

Positional-argument constructor for [`QFIM_obj`](@ref).
"""
QFIM_obj(W, eps, para_type::Symbol, LDtype::Symbol) =
    QFIM_obj{PARA_TYPE_MAP[para_type], LD_TYPE_MAP[LDtype]}(W, eps)

"""
    CFIM_obj(M, W, eps, para_type::Symbol)

Positional-argument constructor for [`CFIM_obj`](@ref).
"""
CFIM_obj(M, W, eps, para_type::Symbol) = CFIM_obj{PARA_TYPE_MAP[para_type]}(M, W, eps)

"""
    HCRB_obj(W, eps, para_type::Symbol)

Positional-argument constructor for [`HCRB_obj`](@ref).
"""
HCRB_obj(W, eps, para_type::Symbol) = HCRB_obj{PARA_TYPE_MAP[para_type]}(W, eps)

"""
    QFIM_obj(W, eps, para_type::String, LDtype::String)

String-input convenience constructor for [`QFIM_obj`](@ref).
Converts string arguments to symbols.
"""
QFIM_obj(W::AbstractMatrix, eps::Number, para_type::String, LDtype::String) =
    QFIM_obj(W, eps, Symbol.([para_type, LDtype])...)

"""
    CFIM_obj(M, W, eps, para_type::String)

String-input convenience constructor for [`CFIM_obj`](@ref).
"""
CFIM_obj(M::AbstractVecOrMat, W::AbstractMatrix, eps::Number, para_type::String) =
    CFIM_obj(M, W, eps, Symbol(para_type))

"""
    HCRB_obj(W, eps, para_type::String)

String-input convenience constructor for [`HCRB_obj`](@ref).
"""
HCRB_obj(W::AbstractMatrix, eps::Number, para_type::String) =
    HCRB_obj(W, eps, Symbol(para_type))

"""
    obj_type(obj)

Return the objective type as a symbol (`:QFIM`, `:CFIM`, or `:HCRB`)
for dispatch purposes.
"""
obj_type(::QFIM_obj) = :QFIM
obj_type(::CFIM_obj) = :CFIM
obj_type(::HCRB_obj) = :HCRB

"""
    para_type(obj)

Return the parameter type as a symbol (`:single_para` or `:multi_para`)
for dispatch purposes.
"""
para_type(::QFIM_obj{single_para,D}) where {D} = :single_para
para_type(::QFIM_obj{multi_para,D}) where {D} = :multi_para
para_type(::CFIM_obj{single_para}) = :single_para
para_type(::CFIM_obj{multi_para}) = :multi_para
para_type(::HCRB_obj{single_para}) = :single_para
para_type(::HCRB_obj{multi_para}) = :multi_para

"""
    QFIM_obj(opt::CFIM_obj{P}) where {P}

Convert a [`CFIM_obj`](@ref) to a [`QFIM_obj`](@ref) (SLD-based) while
preserving the weight matrix and epsilon.
"""
QFIM_obj(opt::CFIM_obj{P}) where {P} = QFIM_obj{P,SLD}(opt.W, opt.eps)

const obj_idx = Dict(:QFIM => QFIM_obj, :CFIM => CFIM_obj, :HCRB => HCRB_obj)

"""
    set_M(obj::CFIM_obj{P}, M::AbstractVector) where {P}

Update the POVM in a [`CFIM_obj`](@ref) and return a new instance.

# Arguments

- `obj::CFIM_obj`: Original CFIM objective with old measurement.
- `M::AbstractVector`: New POVM elements.

# Returns

- `CFIM_obj{P}`: New CFIM objective with the updated measurement.
raw"""
function set_M(obj::CFIM_obj{P}, M::AbstractVector) where {P}
    CFIM_obj{P}(M, obj.W, obj.eps)
end

raw"""
    objective(obj::QFIM_obj{single_para,SLD}, scheme)

Evaluate the single-parameter SLD-based QFI for a given scheme.

Returns ``(f, f)`` where ``f = \mathrm{Tr}(\rho L_x^2)`` is the QFI.
raw"""
function objective(obj::QFIM_obj{single_para,SLD}, scheme)
    (; eps) = obj
    ρ, dρ = evolve(scheme)
    f = QFIM_SLD(ρ, dρ[1]; eps = eps)
    return f, f
end

raw"""
    objective(obj::QFIM_obj{multi_para,SLD}, scheme)

Evaluate the multi-parameter SLD-based QFIM objective ``\mathrm{Tr}(W F^{-1})``.
"""
function objective(obj::QFIM_obj{multi_para,SLD}, scheme)
    (; W, eps) = obj
    ρ, dρ = evolve(scheme)
    f = tr(W * pinv(QFIM_SLD(ρ, dρ; eps = eps)))
    return f, 1.0 / f
end

"""
    objective(obj::QFIM_obj{single_para,RLD}, scheme)

Evaluate the single-parameter RLD-based QFI.
raw"""
function objective(obj::QFIM_obj{single_para,RLD}, scheme)
    (; eps) = obj
    ρ, dρ = evolve(scheme)
    f = QFIM_RLD(ρ, dρ[1]; eps = eps)
    return f, f
end

raw"""
    objective(obj::QFIM_obj{multi_para,RLD}, scheme)

Evaluate the multi-parameter RLD-based QFIM objective ``\mathrm{Tr}(W F_{\mathrm{RLD}}^{-1})``.
"""
function objective(obj::QFIM_obj{multi_para,RLD}, scheme)
    (; W, eps) = obj
    ρ, dρ = evolve(scheme)
    f = tr(W * pinv(QFIM_RLD(ρ, dρ; eps = eps)))
    return f, 1.0 / f
end

"""
    objective(obj::QFIM_obj{single_para,LLD}, scheme)

Evaluate the single-parameter LLD-based QFI.
raw"""
function objective(obj::QFIM_obj{single_para,LLD}, scheme)
    (; eps) = obj
    ρ, dρ = evolve(scheme)
    f = QFIM_LLD(ρ, dρ[1]; eps = eps)
    return f, f
end

raw"""
    objective(obj::QFIM_obj{multi_para,LLD}, scheme)

Evaluate the multi-parameter LLD-based QFIM objective ``\mathrm{Tr}(W F_{\mathrm{LLD}}^{-1})``.
"""
function objective(obj::QFIM_obj{multi_para,LLD}, scheme)
    (; W, eps) = obj
    ρ, dρ = evolve(scheme)
    f = tr(W * pinv(QFIM_LLD(ρ, dρ; eps = eps)))
    return f, 1.0 / f
end

"""
    objective(obj::CFIM_obj{single_para}, scheme)

Evaluate the single-parameter classical Fisher information (CFI) from a scheme.
raw"""
function objective(obj::CFIM_obj{single_para}, scheme)
    (; M, eps) = obj
    ρ, dρ = evolve(scheme)
    f = CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

raw"""
    objective(obj::CFIM_obj{multi_para}, scheme)

Evaluate the multi-parameter CFIM objective ``\mathrm{Tr}(W I^{-1})``.
"""
function objective(obj::CFIM_obj{multi_para}, scheme)
    (; M, W, eps) = obj
    ρ, dρ = evolve(scheme)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

"""
    objective(obj::HCRB_obj{multi_para}, scheme)

Evaluate the HCRB objective from a scheme via SDP (see [`Holevo_bound`](@ref)).
"""
function objective(obj::HCRB_obj{multi_para}, scheme)
    (; W, eps) = obj
    ρ, dρ = evolve(scheme)
    f = Holevo_bound_obj(ρ, dρ, W; eps = eps)
    return f, 1.0 / f
end

#### objective function for linear combination in Mopt ####
"""
    objective(opt::Mopt_LinearComb, obj::CFIM_obj{single_para}, scheme)

Evaluate the single-parameter CFI with measurement parameterized as a
linear combination of POVM basis elements.
"""
function objective(opt::Mopt_LinearComb, obj::CFIM_obj{single_para}, scheme)
    (; eps) = obj
    M = [
        sum([opt.B[i][j] * opt.POVM_basis[j] for j in eachindex(opt.POVM_basis)]) for
        i = 1:opt.M_num
    ]
    ρ, dρ = evolve(scheme)
    f = CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

"""
    objective(opt::Mopt_LinearComb, obj::CFIM_obj{multi_para}, scheme)

Evaluate the multi-parameter CFIM objective with measurement parameterized
as a linear combination of POVM basis elements.
raw"""
function objective(opt::Mopt_LinearComb, obj::CFIM_obj{multi_para}, scheme)
    (; W, eps) = obj
    M = [
        sum([opt.B[i][j] * opt.POVM_basis[j] for j in eachindex(opt.POVM_basis)]) for
        i = 1:opt.M_num
    ]
    ρ, dρ = evolve(scheme)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

#### objective function for rotation in Mopt ####
raw"""
    objective(opt::Mopt_Rotation, obj::CFIM_obj{single_para}, scheme)

Evaluate the single-parameter CFI with measurement parameterized by a
unitary rotation ``U M_i U^\dagger`` of the POVM basis.
raw"""
function objective(opt::Mopt_Rotation, obj::CFIM_obj{single_para}, scheme)
    (; eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U * opt.POVM_basis[i] * U' for i in eachindex(opt.POVM_basis)]
    ρ, dρ = evolve(scheme)
    f = CFIM(ρ, dρ[1], M; eps = eps)
    return f, f
end

raw"""
    objective(opt::Mopt_Rotation, obj::CFIM_obj{multi_para}, scheme)

Evaluate the multi-parameter CFIM objective with measurement parameterized
by a unitary rotation ``U M_i U^\dagger`` of the POVM basis.
"""
function objective(opt::Mopt_Rotation, obj::CFIM_obj{multi_para}, scheme)
    (; W, eps) = obj
    U = rotation_matrix(opt.s, opt.Lambda)
    M = [U * opt.POVM_basis[i] * U' for i in eachindex(opt.POVM_basis)]
    ρ, dρ = evolve(scheme)
    f = tr(W * pinv(CFIM(ρ, dρ, M; eps = eps)))
    return f, 1.0 / f
end

include("CramerRao.jl")
include("AnalogCramerRao.jl")
include("AsymptoticBoundWrapper.jl")
