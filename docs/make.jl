using QuanEstimation
using Documenter
using DocumenterMarkdown

DocMeta.setdocmeta!(
    QuanEstimation,
    :DocTestSetup,
    :(using QuanEstimation);
    recursive = true,
)

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
