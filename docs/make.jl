using QuanEstimation
using Documenter
using DocumenterMarkdown
import Documenter.Writers.MarkdownWriter: render, renderdoc

DocMeta.setdocmeta!(
    QuanEstimation,
    :DocTestSetup,
    :(using QuanEstimation);
    recursive = true,
)

# function render(io::IO, mime::MIME"text/plain", anchor::Documenter.Anchors.Anchor, page, doc)
#     println(io, "\n", lstrip(Documenter.Anchors.fragment(anchor), '#', " "))
#     if anchor.nth == 1 # add legacy id
#         legacy = lstrip(Documenter.Anchors.fragment(anchor), '#', " ") * "-1"
#         println(io, "\n", legacy)
#     end
#     render(io, mime, anchor.object, page, doc)
# end

function render(io::IO, mime::MIME"text/plain", node::Documenter.Documents.DocsNode, page, doc)
    # Docstring header based on the name of the binding and it's category.
    anchor = "## "
    header = "**`$(node.object.binding)`** &mdash; *$(Documenter.Utilities.doccat(node.object))*."
    println(io, anchor, " ", header, "\n\n")
    # Body. May contain several concatenated docstrings.
    renderdoc(io, mime, node.docstr, page, doc)
end

makedocs(;
    format = Markdown(),
    root    = ".",
    source  = "docs/src",
    build   = "docs/build",
    clean   = true,
    doctest = true,
    modules = [QuanEstimation],
    authors = "Hauiming Yu <Huaimingyuuu@gmail.com> and contributors",
    repo = "https://github.com/QuanEstimation/QuanEstimation.jl/blob/{commit}{path}#{line}",
    sitename = "QuanEstimation.jl",
    # format = Documenter.HTML(;
    #     prettyurls = get(ENV, "CI", "false") == "true",
    #     canonical = "https://HuaimingYuuu.github.io/QuanEstimation.jl",
    #     assets = String[],
    # ),
    # pages = [api.md"],

)

deploydocs(; repo = "github.com/QuanEstimation/QuanEstimation.jl")
