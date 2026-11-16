# Common setup for the Literate scripts: locate the repository, load the model files
# and provide a small cache for expensive results.
#
# Expensive results are stored as CSV matrices under `docs/src/assets/data` and reused
# unless `ENV["MFE_RECOMPUTE"] == "true"` (or the file is missing). Delete a file, or
# set the variable, to recompute everything from scratch.

using DelimitedFiles, Printf

const ROOT = get(ENV, "MFE_ROOT", normpath(joinpath(@__DIR__, "..")))
const ASSETS = joinpath(ROOT, "docs", "src", "assets")

include(joinpath(ROOT, "src", "mfe.jl"))
include(joinpath(ROOT, "src", "galerkin.jl"))
include(joinpath(ROOT, "src", "tools.jl"))

recompute() = get(ENV, "MFE_RECOMPUTE", "false") == "true"

"""
    cached(f, name)

Return the matrix stored in `assets/data/name.csv`, or compute it with `f()`, store it and
return it.
"""
function cached(f, name)
    path = joinpath(ASSETS, "data", name * ".csv")
    if !recompute() && isfile(path)
        return readdlm(path, ',', Float64)
    end
    A = f()
    mkpath(dirname(path))
    writedlm(path, A, ',')
    return A
end

"Save a figure under `docs/src/assets`; the scripts then show it as a Markdown image."
function savefig(name, fig)
    mkpath(ASSETS)
    save(joinpath(ASSETS, name), fig; px_per_unit = 2)
    return fig
end
