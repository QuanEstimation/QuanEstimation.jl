abstract type AbstractDtype end

abstract type SLD <: AbstractDtype end
abstract type RLD <: AbstractDtype end
abstract type LLD <: AbstractDtype end

struct QFIM_Obj{P,D} <: AbstractObj
    W::AbstractMatrix
    eps::Number
end

struct CFIM_obj{P} <: AbstractObj
    M::AbstractVecOrMat
    W::AbstractMatrix
    eps::Number
end

struct HCRB{P} <: AbstractObj
    W::AbstractMatrix
    eps::Number
end

QFIM_Obj(W, eps, syms::Symbol...) = QFIM_Obj{eval.(syms)...}(W, eps)
CFIM_Obj(M, W, eps, syms::Symbol...) = CFIM_Obj{eval.(syms)...}(M, W, eps)
HCRB_Obj(W, eps, syms::Symbol...) = HCRB_Obj{eval.(syms)...}(W, eps)
