using VLBILikelihoods
using Documenter

DocMeta.setdocmeta!(VLBILikelihoods, :DocTestSetup, :(using VLBILikelihoods); recursive=true)

makedocs(;
    modules=[VLBILikelihoods],
    authors="Paul Tiede <ptiede91@gmail.com> and contributors",
    repo="https://github.com/ptiede/VLBILikelihoods.jl/blob/{commit}{path}#{line}",
    sitename="VLBILikelihoods.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://ptiede.github.io/VLBILikelihoods.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/ptiede/VLBILikelihoods.jl",
    devbranch="main",
)
