abstract type AbstractDtype end

abstract type SLD <: AbstractDtype end
abstract type RLD <: AbstractDtype end
abstract type LLD <: AbstractDtype end

struct QFIM_Obj{D} <: AbstractObj
    W::AbstractMatrix
    eps::Number
end

struct CFIM_obj <: AbstractObj
    M::AbstractVecOrMat
    W::AbstractMatrix
    eps::Number
end

struct HCRB <: AbstractObj
    W::AbstractMatrix
    eps::Number
end



