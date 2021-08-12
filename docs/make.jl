using QuanEstimation
using Documenter

DocMeta.setdocmeta!(QuanEstimation, :DocTestSetup, :(using QuanEstimation); recursive=true)

makedocs(;
    modules=[QuanEstimation],
    authors="Hauiming Yu <Huaimingyuuu@gmail.com> and contributors",
    repo="https://github.com/HuaimingYuuu/QuanEstimation.jl/blob/{commit}{path}#{line}",
    sitename="QuanEstimation.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://HuaimingYuuu.github.io/QuanEstimation.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/HuaimingYuuu/QuanEstimation.jl",
)
