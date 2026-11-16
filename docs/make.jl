# Build the documentation: `julia --project=. -t 4 docs/make.jl` from the repository root.
# On GitHub Actions (`.github/workflows/docs.yml`) the site is also deployed to gh-pages.
#
# Each script in `scripts/` is run by Literate and turned into a page. Expensive results
# are read from the cache in `docs/src/assets/data` (see `src/setup.jl`); set
# `MFE_RECOMPUTE=true` to recompute them while building.

using Documenter, Literate

const ROOT = normpath(joinpath(@__DIR__, ".."))
ENV["MFE_ROOT"] = ROOT
const GENERATED = joinpath(@__DIR__, "src", "generated")

const SCRIPTS = [
    "01_model.jl" => "Model",
    "02_timeseries_lifetimes.jl" => "Time series and lifetimes",
    "03_bifurcations.jl" => "Bifurcations and edge state",
    "04_lagrangian_descriptors.jl" => "Lagrangian descriptors",
]

for (script, _) in SCRIPTS
    Literate.markdown(joinpath(ROOT, "scripts", script), GENERATED;
                      flavor = Literate.DocumenterFlavor(), execute = true, credit = false)
end

makedocs(;
    sitename = "MFE shear flow: bifurcations and Lagrangian descriptors",
    root = @__DIR__,
    repo = Remotes.GitHub("rmsrosa", "MFEShearFlowLagrangianDescriptors"),
    format = Documenter.HTML(; prettyurls = false, size_threshold = 2^21,
        canonical = "https://rmsrosa.github.io/MFEShearFlowLagrangianDescriptors/dev/",
        edit_link = "main", inventory_version = "dev"),
    pages = [
        "Home" => "index.md",
        "Model" => "generated/01_model.md",
        "Validation" => "validation.md",
        "Time series and lifetimes" => "generated/02_timeseries_lifetimes.md",
        "Bifurcations and edge state" => "generated/03_bifurcations.md",
        "Lagrangian descriptors" => "generated/04_lagrangian_descriptors.md",
    ],
)

# Deploys to the gh-pages branch when run on GitHub Actions (does nothing locally).
deploydocs(;
    repo = "github.com/rmsrosa/MFEShearFlowLagrangianDescriptors.git",
    devbranch = "main",
    push_preview = false,
)
