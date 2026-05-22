using Pkg

Pkg.activate(@__DIR__)
Pkg.instantiate()

using Documenter
using DualArrays

# Setup for doctests in docstrings
DocMeta.setdocmeta!(DualArrays, :DocTestSetup, :(using DualArrays))

makedocs(;
    format = Documenter.HTML(
        canonical = "https://github.com/dlfivefifty/DualArrays.jl",
    ),
    pages = [
        "Home" => "index.md",
        ],
    sitename = "DualArrays.jl",
)

deploydocs(; repo = "github.com/dlfivefifty/DualArrays.jl")