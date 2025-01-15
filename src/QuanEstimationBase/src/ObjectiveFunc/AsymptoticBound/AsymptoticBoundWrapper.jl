function Objective(scheme, obj::QFIM_obj{P,D}) where {P,D}
    (; W, eps) = obj

    p = para_type(scheme.Parameterization) |> eval

    return QFIM_obj{p,D}(W, eps)
end

function Objective(scheme, obj::CFIM_obj{P}) where {P}
    (; W, M, eps) = obj

    if isnothing(M)
        M = SIC(get_dim(scheme))
    end

    p = para_type(scheme.Parameterization) |> eval

    return CFIM_obj{p}(M, W, eps)
end

function Objective(scheme, obj::HCRB_obj{P}) where {P}
    (; W, eps) = obj

    p = para_type(scheme.Parameterization) |> eval

    return HCRB_obj{p}(W, eps)
end
