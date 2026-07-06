using QuanEstimation
using Documenter
using DocumenterCitations

format = Documenter.HTML(
    size_threshold = nothing,  # BaseAPI.md generates >200 KiB
)

bib = CitationBibliography("src/refs.bib")

makedocs(
    format = format,
    plugins = [bib],
    root = ".",
    source = "src",
    build = "build",
    clean = true,
    doctest = false,
    warnonly = [:cross_references],
    modules = [QuanEstimation],
    authors = "QuanEstimation Group (Jing Liu et al.)",
    repo = "https://github.com/QuanEstimation/QuanEstimation.jl/blob/{commit}{path}#{line}",
    sitename = "QuanEstimation.jl",
    pages = [
        "Home" => "index.md",
        "Users Guide" => [
            "Parameterization" => "guide/guide_dynamics.md",
            "Metrological Tools" => "guide/guide_bounds.md",
            "Control Optimization" => "guide/guide_Copt.md",
            "State Optimization" => "guide/guide_Sopt.md",
            "Measurement Optimization" => "guide/guide_Mopt.md",
            "Comprehensive Optimization" => "guide/guide_Compopt.md",
            "Adaptive Schemes" => "guide/guide_adaptive.md",
            "Resources" => "guide/guide_resources.md",
            "Output Files" => "guide/output_files.md",
        ],
        "API Reference" => [
            "Base API" => "api/BaseAPI.md",
            "NVMagnetometer API" => "api/NVMagnetometerAPI.md",
        ],
    ],
)

deploydocs(; repo = "github.com/QuanEstimation/QuanEstimation.jl", devbranch = "main")
