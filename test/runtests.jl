# Run with `julia --project=. test/runtests.jl` from the repository root.
# Set MFE_FAST_TESTS=1 to skip the slower qualitative reproduction of the papers.

using Test, LinearAlgebra, ForwardDiff

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "src", "mfe.jl"))
include(joinpath(ROOT, "src", "galerkin.jl"))

@testset "MFE model" begin
    include("test_model.jl")
end
