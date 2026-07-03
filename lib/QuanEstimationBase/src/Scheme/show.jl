import AbstractTrees as AT
using Crayons.Box: GREEN_FG, YELLOW_FG, BLUE_FG, DARK_GRAY_FG, MAGENTA_FG
using Base.Docs: modules, meta, aliasof, doc

const IGNORE_FIELD_DOC_PREFIX = "IGNORE"

"""

    AnnotatedStructTree

A tree representation of a Julia struct with field names, types, values, and docstrings, used for pretty-printing scheme structures via `AbstractTrees`.

**Fields:**
- `name`: Field name.
- `type`: Field type (or `nothing`).
- `description`: Docstring for the field.
- `value`: Field value or vector of child `AnnotatedStructTree` nodes.
"""
Base.@kwdef struct AnnotatedStructTree
    name::Symbol = Symbol()
    type::Union{Symbol,Nothing} = nothing
    description::String = ""
    value::Any = AnnotatedStructTree[]
end

"""
    annotated_fields(x)

Return a tuple of field names for the given value (inferred from its type).
"""
annotated_fields(x) = annotated_fields(typeof(x))

"""
    annotated_fields(::Type{T}) where {T}

Return `fieldnames(T)` used during annotation extraction.
"""
annotated_fields(::Type{T}) where {T} = fieldnames(T)

"""
    annotations(x::UnionAll)

Return an empty annotation list for `UnionAll` types (no concrete fields available).
"""
annotations(x::UnionAll) = []

"""
    annotations(x)

Extract annotated field tuples `(name, type, description, value)` from `x` by scanning
loaded module doc metas for field-level docstrings. Fields whose docstring starts with
`IGNORE` are skipped.
"""
function annotations(x)
    binding = aliasof(typeof(x), typeof(typeof(x)))
    res = []
    for f in annotated_fields(x)
        v = getfield(x, f)
        doc = ""
        for mod in Base.loaded_modules_array()
            dict = meta(mod)
            if haskey(dict, binding)
                multidoc = dict[binding]
                if haskey(multidoc.docs, Union{})
                    data_fields = multidoc.docs[Union{}].data
                    if haskey(data_fields, :fields) && haskey(data_fields[:fields], f)
                        doc = data_fields[:fields][f] |> string |> strip
                        break
                    end
                end
            end
        end
        if !startswith(doc, IGNORE_FIELD_DOC_PREFIX)
            push!(res, (name=Symbol(f), type=nameof(typeof(v)), description=doc, value=v))
        end
    end
    return res
end

"""
    Base.convert(::Type{AnnotatedStructTree}, x::AnnotatedStructTree)

Identity conversion for `AnnotatedStructTree` (no-op).
"""
Base.convert(::Type{AnnotatedStructTree}, x::AnnotatedStructTree) = x

"""
    Base.convert(::Type{AnnotatedStructTree}, x; name, type, description)

Recursively convert a value `x` into an `AnnotatedStructTree` by annotating its fields.
Scalar values become leaf nodes; struct values are expanded into children.
"""
function Base.convert(::Type{AnnotatedStructTree}, x; name=Symbol(), type=nameof(typeof(x)), description="")
    ants = annotations(x)
    if isempty(ants)
        AnnotatedStructTree(name=name, type=type, description=description, value=x)
    else
        AnnotatedStructTree(
            name=name,
            type=type,
            description=description,
            value=AnnotatedStructTree[
                convert(AnnotatedStructTree, a.value; name=a.name, type=a.type, description=a.description)
                for a in ants if a.value!=nothing
            ]
        )
    end
end

"""
    AT.children(t::AnnotatedStructTree)

Return the children of an `AnnotatedStructTree` node: its `value` field if it is a vector of child nodes, otherwise an empty list.
"""
AT.children(t::AnnotatedStructTree) = t.value isa Vector{AnnotatedStructTree} ? t.value : []

"""
    AT.printnode(io::IO, t::AnnotatedStructTree)

Pretty-print a single `AnnotatedStructTree` node with colored name, type, value, and description.
"""
function AT.printnode(io::IO, t::AnnotatedStructTree)
    print(io, YELLOW_FG(string(t.name)))

    if !isnothing(t.type)
        print(io, "::", BLUE_FG(string(t.type)))
    end

    if !(t.value isa Vector{AnnotatedStructTree})
        h, w = displaysize(io)
        v = string(t.value)
        if length(v) + length(t.description) + 10 > w
            v = summary(t.value)
        end
        print(io, MAGENTA_FG(" => "), GREEN_FG(v))
    end

    print(io, "\t", DARK_GRAY_FG(t.description))
end

"""
    Base.show(io::IO, ::MIME"text/plain", t::AnnotatedStructTree)

Pretty-print an `AnnotatedStructTree` via `AbstractTrees.print_tree`.
"""
Base.show(io::IO, ::MIME"text/plain", t::AnnotatedStructTree) = AT.print_tree(io, t; maxdepth=get(io, :maxdepth, 10))

"""
    annotated_fields(x::AbstractVecOrMat)

Return an empty tuple — vectors and matrices are treated as leaf values, not expanded.
"""
annotated_fields(x::AbstractVecOrMat) = ()
"""
    annotated_fields(x::Nothing)

Return an empty tuple for `nothing` leaf values.
"""
annotated_fields(x::Nothing) = ()


#####

"""

    Base.show(io::IO, m::MIME"text/plain", s::AbstractScheme)

Pretty-print an `AbstractScheme` as an annotated tree.
"""
Base.show(io::IO, m::MIME"text/plain", s::AbstractScheme) = show(io, m, convert(AnnotatedStructTree, s))