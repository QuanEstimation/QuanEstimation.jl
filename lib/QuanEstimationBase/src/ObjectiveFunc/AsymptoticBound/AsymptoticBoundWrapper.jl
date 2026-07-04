"""

    Objective(scheme, obj::QFIM_obj)

Construct a QFIM objective with the correct parameter type tag inferred from the scheme.
"""
function Objective(scheme, obj::QFIM_obj{P,D}) where {P,D}
    (; W, eps) = obj

    p = PARA_TYPE_MAP[para_type(scheme.Parameterization)]

    return QFIM_obj{p,D}(W, eps)
end

"""

    Objective(scheme, obj::CFIM_obj)

Construct a CFIM objective with the correct parameter type tag. Falls back to SIC-POVM if no measurement is provided.
"""
function Objective(scheme, obj::CFIM_obj{P}) where {P}
    (; W, M, eps) = obj

    if isnothing(M)
        M = SIC(get_dim(scheme))
    end

    p = PARA_TYPE_MAP[para_type(scheme.Parameterization)]

    return CFIM_obj{p}(M, W, eps)
end

"""

    Objective(scheme, obj::HCRB_obj)

Construct an HCRB objective with the correct parameter type tag inferred from the scheme.
"""
function Objective(scheme, obj::HCRB_obj{P}) where {P}
    (; W, eps) = obj

    p = PARA_TYPE_MAP[para_type(scheme.Parameterization)]

    return HCRB_obj{p}(W, eps)
end
