# Saddle-node and Hopf points of the equilibria for Lx = 4π, Lz = 2π (SIADS 2005 §4.1),
# computed with BifurcationKit from the same right-hand side.

using Test, BifurcationKit, ForwardDiff, LinearAlgebra

@testset "saddle-node and Hopf (SIADS: Re = 308.16, 310.34)" begin
    p = MFEParams(; Re = 400.0, NBC_DOMAIN...)
    a, ok = find_equilibrium([0.192, 0.048, 0.035, 0.017, 0.036, 0.106, -0.18, 0.063, -0.09], p)
    @test ok
    mfe_bk!(da, a, p) = (mfe!(da, a, p, 0.0); da)
    prob = ODEBifProblem(mfe_bk!, a, p, (@optic _.Re))
    opts = ContinuationPar(p_min = 250.0, p_max = 450.0, ds = -0.5, dsmax = 1.0, dsmin = 1e-6,
        max_steps = 1000, detect_bifurcation = 3, n_inversion = 8, nev = 9,
        newton_options = NewtonPar(tol = 1e-10, max_iterations = 20))
    br = continuation(prob, PALC(tangent = Bordered()), opts; verbosity = 0)
    @test minimum(br.branch.param) ≈ 308.16 atol = 0.01            # turning point
    ih = findfirst(sp -> sp.type == :hopf, br.specialpoint)
    @test ih !== nothing
    @test br.specialpoint[ih].param ≈ 310.34 atol = 0.01
end
