using QuanEstimation
using Documenter
using DocumenterMarkdown
using DocumenterCitations

bib = CitationBibliography("src/refs.bib")

makedocs(
    bib;
    root = ".",
    source = "src",
    build = "build",
    clean = true,
    doctest = true,
    modules = [QuanEstimation],
    authors = "Hauiming Yu <Huaimingyuuu@gmail.com> and contributors",
    repo = "https://github.com/QuanEstimation/QuanEstimation.jl/blob/{commit}{path}#{line}",
    sitename = "QuanEstimation.jl",
    pages = ["API" => ["api/GeneralAPI.md", "api/BaseAPI.md", "api/NVMagnetometerAPI.md"]],
)

#deploydocs(; repo = "github.com/QuanEstimation/QuanEstimation.jl")
