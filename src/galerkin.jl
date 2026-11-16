# Independent derivation of the MFE coefficients by Galerkin projection.
#
# The nine basis fields uₙ(x, y, z) are transcribed from NJP eqs (7)–(16). We insert
# u = Σ aₘ uₘ into the forced Navier–Stokes equations (NJP eq (1)) and project onto uₙ
# (NJP eq (20)):
#
#   ⟨uₙ,uₙ⟩ ȧₙ = -Σₘₗ aₘ aₗ ⟨(uₘ·∇)uₗ, uₙ⟩ + (1/Re) Σₘ aₘ ⟨∇²uₘ, uₙ⟩ + ⟨F, uₙ⟩.
#
# Integrals over Ω = [0,Lx]×[-1,1]×[0,Lz] use a uniform grid in x and z (exact for the
# trigonometric polynomials involved) and Gauss–Legendre nodes in y. Derivatives are taken
# with ForwardDiff, so nothing here reuses the hand-transcribed equations in `mfe.jl`.
#
# Requires `using ForwardDiff, LinearAlgebra` and `include("mfe.jl")` beforehand.

"""
    basis_field(n, X, p::MFEParams)

Velocity vector of basis mode `n` at point `X = (x, y, z)` (NJP eqs (7)–(16)).
"""
function basis_field(n::Integer, X, p::MFEParams)
    x, y, z = X[1], X[2], X[3]
    (; α, γ) = p
    T = eltype(X)
    s, c = sin(π * y / 2), cos(π * y / 2)
    c2 = c^2
    if n == 1
        return T[sqrt(2) * s, 0, 0]
    elseif n == 2
        return T[4 / sqrt(3) * c2 * cos(γ * z), 0, 0]
    elseif n == 3
        k = 2 / sqrt(4γ^2 + π^2)
        return T[0, k * 2γ * c * cos(γ * z), k * π * s * sin(γ * z)]
    elseif n == 4
        return T[0, 0, 4 / sqrt(3) * cos(α * x) * c2]
    elseif n == 5
        return T[0, 0, 2 * sin(α * x) * s]
    elseif n == 6
        k = 4sqrt(2) / sqrt(3 * (α^2 + γ^2))
        return T[-k * γ * cos(α * x) * c2 * sin(γ * z), 0, k * α * sin(α * x) * c2 * cos(γ * z)]
    elseif n == 7
        k = 2sqrt(2) / sqrt(α^2 + γ^2)
        return T[k * γ * sin(α * x) * s * sin(γ * z), 0, k * α * cos(α * x) * s * cos(γ * z)]
    elseif n == 8
        N8 = 2sqrt(2) / sqrt((α^2 + γ^2) * (4α^2 + 4γ^2 + π^2))
        return T[N8 * π * α * sin(α * x) * s * sin(γ * z),
                 N8 * 2(α^2 + γ^2) * cos(α * x) * c * sin(γ * z),
                 -N8 * π * γ * cos(α * x) * s * cos(γ * z)]
    elseif n == 9
        return T[sqrt(2) * sin(3π * y / 2), 0, 0]
    end
    throw(ArgumentError("mode index must be in 1:9"))
end

"Body force `F(y)` divided by `1/Re` (NJP eq (5)): `√2 π²/4 · sin(πy/2) ê_x`."
forcing_field(X) = [sqrt(2) * π^2 / 4 * sin(π * X[2] / 2), zero(eltype(X)), zero(eltype(X))]

"Gauss–Legendre nodes and weights on [-1, 1] (Golub–Welsch)."
function gauss_legendre(n::Integer)
    b = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    E = eigen(SymTridiagonal(zeros(n), b))
    return E.values, 2 .* E.vectors[1, :] .^ 2
end

"Quadrature nodes and weights on Ω for the basis fields with parameters `p`."
function quadrature_grid(p::MFEParams; nx = 16, ny = 48, nz = 16)
    xs = (0:nx-1) .* (p.Lx / nx)
    zs = (0:nz-1) .* (p.Lz / nz)
    ys, wy = gauss_legendre(ny)
    pts = [[x, y, z] for x in xs, y in ys, z in zs]
    wts = [p.Lx / nx * w * p.Lz / nz for _ in xs, w in wy, _ in zs]
    return vec(pts), vec(wts)
end

"Divergence of basis mode `n` at `X` (for tests)."
divergence(n, X, p) = tr(ForwardDiff.jacobian(Y -> basis_field(n, Y, p), X))

"Laplacian of basis mode `n` at `X`, componentwise."
function laplacian(n, X, p)
    return [tr(ForwardDiff.hessian(Y -> basis_field(n, Y, p)[c], X)) for c in 1:3]
end

"""
    galerkin_coefficients(p::MFEParams; kwargs...)

Return a named tuple `(; G, L, f, Q)` with

- `G[n, m] = ⟨uₙ, uₘ⟩` (Gram matrix; should be `2 Lx Lz I`, NJP eq (17)),
- `L[n, m] = ⟨∇²uₘ, uₙ⟩ / ⟨uₙ, uₙ⟩` (so the viscous term of `ȧₙ` is `Σₘ L[n,m] aₘ / Re`),
- `f[n] = ⟨F·Re, uₙ⟩ / ⟨uₙ, uₙ⟩` (so the forcing term of `ȧₙ` is `f[n] / Re`),
- `Q[n, m, l] = -⟨(uₘ·∇)uₗ, uₙ⟩ / ⟨uₙ, uₙ⟩` (so the nonlinear term is `Σₘₗ Q[n,m,l] aₘ aₗ`).
"""
function galerkin_coefficients(p::MFEParams; kwargs...)
    pts, wts = quadrature_grid(p; kwargs...)
    npt = length(pts)
    U = [basis_field(n, X, p) for n in 1:9, X in pts]
    J = [ForwardDiff.jacobian(Y -> basis_field(n, Y, p), X) for n in 1:9, X in pts]  # J[i,j] = ∂uᵢ/∂xⱼ
    Δ = [laplacian(n, X, p) for n in 1:9, X in pts]
    F = [forcing_field(X) for X in pts]

    G = [sum(wts[q] * dot(U[n, q], U[m, q]) for q in 1:npt) for n in 1:9, m in 1:9]
    norms = [G[n, n] for n in 1:9]
    L = [sum(wts[q] * dot(Δ[m, q], U[n, q]) for q in 1:npt) / norms[n] for n in 1:9, m in 1:9]
    f = [sum(wts[q] * dot(F[q], U[n, q]) for q in 1:npt) / norms[n] for n in 1:9]
    Q = zeros(9, 9, 9)
    for n in 1:9, m in 1:9, l in 1:9
        Q[n, m, l] = -sum(wts[q] * dot(J[l, q] * U[m, q], U[n, q]) for q in 1:npt) / norms[n]
    end
    return (; G, L, f, Q)
end

"""
    GalerkinMFE(p::MFEParams; tol = 1e-12, kwargs...)

The MFE right-hand side assembled from projected coefficients, stored sparsely.
Use with [`galerkin_mfe!`](@ref). `tol` drops numerically zero couplings.
"""
struct GalerkinMFE{R,T}
    Re::R
    L::Vector{T}                       # diagonal of the viscous operator (checked diagonal)
    f::Vector{T}
    terms::Vector{Tuple{Int,Int,Int,T}} # (n, m, l, c) with c·aₘ·aₗ contributing to ȧₙ, m ≤ l
end

function GalerkinMFE(p::MFEParams; tol = 1e-12, kwargs...)
    (; L, f, Q) = galerkin_coefficients(p; kwargs...)
    terms = Tuple{Int,Int,Int,Float64}[]
    for n in 1:9, m in 1:9, l in m:9
        c = m == l ? Q[n, m, m] : Q[n, m, l] + Q[n, l, m]
        abs(c) > tol && push!(terms, (n, m, l, c))
    end
    fz = [abs(v) > tol ? v : 0.0 for v in f]
    return GalerkinMFE(p.Re, [L[n, n] for n in 1:9], fz, terms)
end

"In-place right-hand side built from the Galerkin coefficients in `g`."
function galerkin_mfe!(da, a, g::GalerkinMFE, t)
    @inbounds for n in 1:9
        da[n] = (g.L[n] * a[n] + g.f[n]) / g.Re
    end
    @inbounds for (n, m, l, c) in g.terms
        da[n] += c * a[m] * a[l]
    end
    return nothing
end
