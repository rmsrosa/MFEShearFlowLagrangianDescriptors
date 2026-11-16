# Qualitative reproduction of Moehlis, Faisst & Eckhardt (NJP 2004, SIADS 2005).
# These take a few minutes; skip with MFE_FAST_TESTS=1.

using Test, OrdinaryDiffEq, LinearAlgebra, Random, Statistics

const MFU(Re) = MFEParams(; Re, MFU_DOMAIN...)

@testset "subcritical transition (MFU, Re = 400)" begin
    p = MFU(400.0)
    # small perturbations of the (linearly stable) laminar state decay quickly …
    a_small = laminar_state() .+ 1e-3 .* normalize(ones(9))
    @test lifetime(a_small, p; tmax = 1000.0) < 200
    # … while finite-amplitude ones (NJP §5 initial condition) stay turbulent for long
    @test lifetime(njp_initial_condition(0.2), p; tmax = 2000.0) == 2000.0
end

@testset "stable periodic orbit at Re = 400 (SIADS §4.2, T = 112.51)" begin
    pts, ts, _ = poincare_section(njp_initial_condition(0.2), MFU(400.0); tmax = 8000.0, tskip = 5000.0)
    @test length(pts) > 20
    a1 = [q[1] for q in pts]
    @test maximum(a1) - minimum(a1) < 1e-6          # single point on the section
    @test mean(diff(ts)) ≈ 112.51 rtol = 1e-3
end

@testset "attractor below the torus bifurcation (SIADS Fig. 21)" begin
    # follow the attractor adiabatically downward from the periodic orbit at Re = 400
    _, _, u = poincare_section(njp_initial_condition(0.2), MFU(400.0); tmax = 3000.0)
    for Re in 390.0:-5.0:345.0
        _, _, u = poincare_section(u, MFU(Re); tmax = 2000.0)
    end
    @test !laminar_criterion(u)
    # chaotic at Re = 345: positive largest Lyapunov exponent, no laminarisation
    λ, lam = lyapunov_exponent(u, MFU(345.0); T = 4e4)
    @test !lam
    @test λ > 5e-4
    # on the stable periodic orbit at Re = 400 the largest exponent is zero
    _, _, u400 = poincare_section(njp_initial_condition(0.2), MFU(400.0); tmax = 3000.0)
    λ400, _ = lyapunov_exponent(u400, MFU(400.0); T = 4e4)
    @test abs(λ400) < 2e-4
    # below the boundary crisis (SIADS: Re ≈ 335) the attractor is gone
    @test lifetime(u, MFU(320.0); tmax = 2e4) < 2e4
end

@testset "exponential lifetime distributions (NJP §5, Fig. 7)" begin
    rng = Xoshiro(2004)
    medians = Float64[]
    for Re in (200.0, 250.0)
        p = MFU(Re)
        τ = [lifetime(random_shell_state(rng, 0.3), p; tmax = 1e4) for _ in 1:400]
        push!(medians, median(τ))
        # survival probability P(t) for t past an initial transient
        t = sort(τ[τ .< 1e4])
        P = 1 .- (1:length(t)) ./ length(τ)
        mask = (t .> 300) .& (P .> 0.02)
        X = [ones(count(mask)) t[mask]]
        coef = X \ log.(P[mask])
        resid = log.(P[mask]) - X * coef
        r2 = 1 - sum(abs2, resid) / sum(abs2, log.(P[mask]) .- mean(log.(P[mask])))
        @test coef[2] < 0
        @test r2 > 0.97                             # log P(t) is linear: exponential tail
    end
    @test medians[2] > medians[1]                   # lifetimes grow with Re
end
