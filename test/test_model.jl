# Fast checks of the model itself: basis fields, two independent right-hand sides,
# fixed point, energy conservation, linear stability, allocations.

using Test, LinearAlgebra, Random, ForwardDiff

const DOMAINS = (NBC_DOMAIN, MFU_DOMAIN, (Lx = 3.1, Lz = 5.3))

@testset "basis fields (NJP eqs (7)–(17))" begin
    rng = Xoshiro(42)
    for dom in DOMAINS
        p = MFEParams(; Re = 1.0, dom...)
        (; G) = galerkin_coefficients(p)
        @test G ≈ 2 * p.Lx * p.Lz * I atol = 1e-10            # orthogonal, eq (17)
        for n in 1:9, _ in 1:5
            X = [p.Lx * rand(rng), 2rand(rng) - 1, p.Lz * rand(rng)]
            @test abs(divergence(n, X, p)) < 1e-12              # incompressible
            for yw in (-1.0, 1.0)                               # free slip, eq (4)
                Xw = [X[1], yw, X[3]]
                J = ForwardDiff.jacobian(Y -> basis_field(n, Y, p), Xw)
                @test abs(basis_field(n, Xw, p)[2]) < 1e-12
                @test abs(J[1, 2]) < 1e-12 && abs(J[3, 2]) < 1e-12
            end
        end
    end
end

@testset "explicit equations agree with Galerkin projection" begin
    rng = Xoshiro(1)
    for dom in DOMAINS, Re in (50.0, 400.0)
        p = MFEParams(; Re, dom...)
        g = GalerkinMFE(p)
        (; L) = galerkin_coefficients(p)
        @test norm(L - Diagonal(L)) < 1e-10                    # viscous term is diagonal
        @test -diag(L) ≈ collect(decay_rates(p)) rtol = 1e-10
        d1, d2 = zeros(9), zeros(9)
        for _ in 1:100
            a = randn(rng, 9)
            mfe!(d1, a, p, 0.0)
            galerkin_mfe!(d2, a, g, 0.0)
            @test d1 ≈ d2 atol = 1e-11
        end
    end
end

@testset "laminar state is a fixed point for all Re" begin
    for dom in DOMAINS, Re in (1.0, 10.0, 308.16, 400.0, 1e4)
        @test norm(mfe(laminar_state(), MFEParams(; Re, dom...))) < 1e-14
    end
end

@testset "nonlinear terms conserve energy" begin
    rng = Xoshiro(7)
    for dom in DOMAINS
        p = MFEParams(; Re = 400.0, dom...)
        λ = decay_rates(p)
        for _ in 1:100
            a = randn(rng, 9)
            lin = [-λ[i] * a[i] / p.Re for i in 1:9]
            lin[1] += p.β^2 / p.Re
            nl = mfe(a, p) - lin
            @test abs(dot(a, nl)) < 1e-12 * max(1, norm(a)^3)  # Σᵢ aᵢ Nᵢⱼₖ aⱼ aₖ = 0
        end
    end
end

@testset "laminar state is linearly stable" begin
    for dom in DOMAINS, Re in (1.0, 100.0, 400.0, 1e3, 1e5)
        p = MFEParams(; Re, dom...)
        J = ForwardDiff.jacobian(a -> mfe(a, p), laminar_state())
        @test maximum(real, eigvals(J)) < 0
    end
end

@testset "right-hand side is type stable and allocation-free" begin
    p = MFEParams(Re = 400.0)
    a, da = rand(9), zeros(9)
    @inferred mfe!(da, a, p, 0.0)
    mfe!(da, a, p, 0.0)
    @test (@allocated mfe!(da, a, p, 0.0)) == 0
    # dual-number Re (as used when differentiating w.r.t. the parameter)
    pd = with_Re(p, ForwardDiff.Dual(400.0, 1.0))
    @test mfe(a, pd) isa Vector{<:ForwardDiff.Dual}
end
