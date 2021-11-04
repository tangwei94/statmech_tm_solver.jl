using statmech_tm_solver
using Documenter

DocMeta.setdocmeta!(statmech_tm_solver, :DocTestSetup, :(using statmech_tm_solver); recursive=true)

makedocs(;
    modules=[statmech_tm_solver],
    authors="Wei Tang <tangwei@smail.nju.edu.cn> and contributors",
    repo="https://github.com/wei.tang.nju@gmail.com/statmech_tm_solver.jl/blob/{commit}{path}#{line}",
    sitename="statmech_tm_solver.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://wei.tang.nju@gmail.com.github.io/statmech_tm_solver.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/wei.tang.nju@gmail.com/statmech_tm_solver.jl",
)
