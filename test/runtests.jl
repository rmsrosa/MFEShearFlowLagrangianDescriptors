# Run with `julia --project=. test/runtests.jl` from the repository root.
# Set MFE_FAST_TESTS=1 to skip the slower qualitative reproduction of the papers.

using Test, LinearAlgebra, ForwardDiff, OrdinaryDiffEq, Random

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "src", "mfe.jl"))
include(joinpath(ROOT, "src", "galerkin.jl"))
include(joinpath(ROOT, "src", "tools.jl"))

const FAST = get(ENV, "MFE_FAST_TESTS", "0") == "1"

@testset "MFE" begin
    @testset "model" begin
        include("test_model.jl")
    end
    if !FAST
        @testset "dynamics" begin
            include("test_dynamics.jl")
        end
        @testset "bifurcations" begin
            include("test_bifurcation.jl")
        end
    end
end
