abstract type AbstractLDtype end

abstract type SLD <: AbstractLDtype end
abstract type RLD <: AbstractLDtype end
abstract type LLD <: AbstractLDtype end

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


function objective(obj::QFIM{single_para, SLD}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    W[1]*QFIM_SLD(ρ,dρ[1];eps=eps)    
end

function objective(obj::QFIM{multi_para, SLD}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    tr(W*pinv(QFIM_SLD(ρ,dρ;eps=eps)))
end

function objective(obj::QFIM{single_para, RLD}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    W[1]*QFIM_RLD(ρ,dρ[1];eps=eps)    
end

function objective(obj::QFIM{multi_para, RLD}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    tr(W*pinv(QFIM_RLD(ρ,dρ;eps=eps)))  
end

function objective(obj::QFIM{single_para, LLD}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    W[1]*QFIM_LLD(ρ,dρ[1];eps=eps)    
end

function objective(obj::QFIM{multi_para, LLD}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    tr(W*pinv(QFIM_LLD(ρ,dρ;eps=eps)))  
end

function objective(obj::QFIM{single_para, SLD}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    W[1]*QFIM_SLD(ρ,dρ[1];eps=eps)    
end

function objective(obj::QFIM{multi_para, SLD}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    tr(W*pinv(QFIM_SLD(ρ,dρ;eps=eps)))
end

function objective(obj::QFIM{single_para, RLD}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    W[1]*QFIM_RLD(ρ,dρ[1];eps=eps)    
end

function objective(obj::QFIM{multi_para, RLD}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    tr(W*pinv(QFIM_RLD(ρ,dρ;eps=eps)))  
end

function objective(obj::QFIM{single_para, LLD}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    W[1]*QFIM_LLD(ρ,dρ[1];eps=eps)    
end

function objective(obj::QFIM{multi_para, LLD}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    tr(W*pinv(QFIM_LLD(ρ,dρ;eps=eps)))  
end

function objective(obj::CFIM{single_para}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    W[1]*CFIM(ρ,dρ[1];eps=eps)    
end

function objective(obj::CFIM{multi_para}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    tr(W*pinv(CFIM(ρ,dρ;eps=eps)))
end

function objective(obj::CFIM{single_para}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    W[1]*CFIM(ρ,dρ[1];eps=eps)    
end

function objective(obj::CFIM{multi_para}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    tr(W*pinv(CFIM(ρ,dρ;eps=eps)))
end

function objective(obj::HCRB{multi_para}, dynamics::Lindblad)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    Holevo_bound(ρ,dρ,W;eps=eps)
end

function objective(obj::HCRB{multi_para}, dynamics::Kraus)
    (;W,eps) = obj
    ρ, dρ = evolve(dynamics)
    Holevo_bound(ρ,dρ,W;eps=eps)
end
