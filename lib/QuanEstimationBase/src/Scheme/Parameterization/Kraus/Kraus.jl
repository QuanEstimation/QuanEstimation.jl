@doc raw"""
# Kraus operator parameterization

This module provides the Kraus operator representation for quantum dynamics:

```math
\rho = \sum_i K_i \rho_0 K_i^\dagger,\qquad \sum_i K_i^\dagger K_i = \mathbb{I}.
```

The Kraus operators ``K_i`` completely characterize the quantum channel.
"""
abstract type AbstractKraus <: AbstractParameterization end

raw"""
    AbstractKrausData

Abstract supertype for Kraus data containers.
raw"""
abstract type AbstractKrausData end

@doc raw"""
    Kraus{KT,DKT,NK,NP} <: AbstractKraus

Mutable struct representing a full Kraus channel specification.

Combines Kraus operator data with parameter values.

# Type Parameters

- `KT`: Type of the Kraus operators ``K_i``.
- `DKT`: Type of the Kraus operator derivatives ``\partial_a K_i``.
- `NK`: Number of Kraus operators.
- `NP`: Number of parameters.

# Fields

- `data::AbstractKrausData`: The Kraus data container ([`KrausData`](@ref)).
- `params::Union{Nothing,AbstractVector}`: Parameter values.
"""
mutable struct Kraus{KT,DKT,NK,NP} <: AbstractKraus
    data::AbstractKrausData
    params::Union{Nothing,AbstractVector}
end

include("KrausData.jl")
include("KrausParamerization.jl")