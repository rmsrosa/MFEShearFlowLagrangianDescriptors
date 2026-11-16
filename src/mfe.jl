# Moehlis–Faisst–Eckhardt (MFE) nine-mode model of sinusoidal shear flow.
#
# Plain include file (not a package module). Provides
#
#   MFEParams, mfe!, mfe, laminar_state, energy, fluctuation_energy,
#   kinetic_energy_rate, MFU_DOMAIN, NBC_DOMAIN
#
# Equations are transcribed from
#   J. Moehlis, H. Faisst, B. Eckhardt, New J. Phys. 6 (2004) 56, eqs (21)–(32),
# which coincide with eqs (3.14)–(3.23) of
#   J. Moehlis, H. Faisst, B. Eckhardt, SIAM J. Appl. Dyn. Syst. 4 (2005) 352.
#
# An independent Galerkin projection of the basis fields lives in `galerkin.jl`.

"""
    MFEParams(; Re, Lx = 4π, Lz = 2π)

Parameters of the MFE model: Reynolds number `Re` and domain lengths `Lx`, `Lz`.
Stores the derived wavenumbers `α = 2π/Lx`, `β = π/2`, `γ = 2π/Lz` and the
combinations `καγ`, `κβγ`, `καβγ` (NJP eqs (30)–(32)).

`Re` has its own type parameter so that it can be a dual number (e.g. when
BifurcationKit differentiates with respect to `Re`) while the geometry stays `Float64`.
"""
struct MFEParams{R<:Real,T<:Real}
    Re::R
    Lx::T
    Lz::T
    α::T
    β::T
    γ::T
    kαγ::T
    kβγ::T
    kαβγ::T
end

function MFEParams(Re::Real, Lx::Real, Lz::Real)
    Lx, Lz = promote(float(Lx), float(Lz))
    α = 2π / Lx
    β = oftype(Lx, π / 2)
    γ = 2π / Lz
    return MFEParams(Re, Lx, Lz, α, β, γ, sqrt(α^2 + γ^2), sqrt(β^2 + γ^2), sqrt(α^2 + β^2 + γ^2))
end
MFEParams(; Re = 400.0, Lx = 4π, Lz = 2π) = MFEParams(Re, Lx, Lz)

"Return a copy of `p` with a new Reynolds number."
with_Re(p::MFEParams, Re) = MFEParams(Re, p.Lx, p.Lz, p.α, p.β, p.γ, p.kαγ, p.kβγ, p.kαβγ)

"Minimal flow unit domain `(Lx, Lz) = (1.75π, 1.2π)` (NJP §4.1, SIADS §4.2)."
const MFU_DOMAIN = (Lx = 1.75π, Lz = 1.2π)
"Nagata–Busse–Clever domain `(Lx, Lz) = (4π, 2π)` (NJP §4.2, SIADS §4.1)."
const NBC_DOMAIN = (Lx = 4π, Lz = 2π)

"Laminar fixed point `a = (1, 0, …, 0)`."
laminar_state(::Type{T} = Float64) where {T} = T[1, 0, 0, 0, 0, 0, 0, 0, 0]

"Total energy `Σ aᵢ²` (in units of the mode normalisation, NJP eq (17))."
energy(a) = sum(abs2, a)

"Fluctuation energy relative to the laminar state, `(1 - a₁)² + Σ_{j≥2} aⱼ²` (SIADS eq (4.1))."
fluctuation_energy(a) = (1 - a[1])^2 + sum(abs2, @view a[2:9])

"""
    mfe!(da, a, p::MFEParams, t)

In-place right-hand side of the nine MFE amplitude equations (NJP eqs (21)–(29)).
Allocation-free and type-stable; usable directly in an `ODEProblem`.
"""
function mfe!(da, a, p::MFEParams, t)
    (; Re, α, β, γ, kαγ, kβγ, kαβγ) = p
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]
    s32 = sqrt(3 / 2)
    s6 = sqrt(6)

    # (21)
    da[1] = β^2 / Re - β^2 / Re * a1 -
            s32 * β * γ / kαβγ * a6 * a8 + s32 * β * γ / kβγ * a2 * a3
    # (22)
    da[2] = -(4β^2 / 3 + γ^2) * a2 / Re +
            5sqrt(2) / (3sqrt(3)) * γ^2 / kαγ * a4 * a6 -
            γ^2 / (s6 * kαγ) * a5 * a7 -
            α * β * γ / (s6 * kαγ * kαβγ) * a5 * a8 -
            s32 * β * γ / kβγ * a1 * a3 -
            s32 * β * γ / kβγ * a3 * a9
    # (23)
    da[3] = -(β^2 + γ^2) / Re * a3 +
            2 / s6 * α * β * γ / (kαγ * kβγ) * (a4 * a7 + a5 * a6) +
            (β^2 * (3α^2 + γ^2) - 3γ^2 * (α^2 + γ^2)) / (s6 * kαγ * kβγ * kαβγ) * a4 * a8
    # (24)
    da[4] = -(3α^2 + 4β^2) / (3Re) * a4 -
            α / s6 * a1 * a5 -
            10 / (3s6) * α^2 / kαγ * a2 * a6 -
            s32 * α * β * γ / (kαγ * kβγ) * a3 * a7 -
            s32 * α^2 * β^2 / (kαγ * kβγ * kαβγ) * a3 * a8 -
            α / s6 * a5 * a9
    # (25)
    da[5] = -(α^2 + β^2) / Re * a5 +
            α / s6 * a1 * a4 +
            α^2 / (s6 * kαγ) * a2 * a7 -
            α * β * γ / (s6 * kαγ * kαβγ) * a2 * a8 +
            α / s6 * a4 * a9 +
            2 / s6 * α * β * γ / (kαγ * kβγ) * a3 * a6
    # (26)
    da[6] = -(3α^2 + 4β^2 + 3γ^2) / (3Re) * a6 +
            α / s6 * a1 * a7 +
            s32 * β * γ / kαβγ * a1 * a8 +
            10 / (3s6) * (α^2 - γ^2) / kαγ * a2 * a4 -
            2sqrt(2 / 3) * α * β * γ / (kαγ * kβγ) * a3 * a5 +
            α / s6 * a7 * a9 +
            s32 * β * γ / kαβγ * a8 * a9
    # (27)
    da[7] = -(α^2 + β^2 + γ^2) / Re * a7 -
            α / s6 * (a1 * a6 + a6 * a9) +
            1 / s6 * (γ^2 - α^2) / kαγ * a2 * a5 +
            1 / s6 * α * β * γ / (kαγ * kβγ) * a3 * a4
    # (28)
    da[8] = -(α^2 + β^2 + γ^2) / Re * a8 +
            2 / s6 * α * β * γ / (kαγ * kαβγ) * a2 * a5 +
            γ^2 * (3α^2 - β^2 + 3γ^2) / (s6 * kαγ * kβγ * kαβγ) * a3 * a4
    # (29)
    da[9] = -9β^2 / Re * a9 +
            s32 * β * γ / kβγ * a2 * a3 -
            s32 * β * γ / kαβγ * a6 * a8
    return nothing
end

"Out-of-place right-hand side (allocates; handy for BifurcationKit and quick checks)."
function mfe(a, p::MFEParams, t = 0)
    da = similar(a, promote_type(eltype(a), typeof(p.Re)))
    mfe!(da, a, p, t)
    return da
end

"Viscous decay rates `λᵢ` such that the linear term of `daᵢ/dt` is `-λᵢ aᵢ / Re`."
function decay_rates(p::MFEParams)
    (; α, β, γ) = p
    return (β^2, 4β^2 / 3 + γ^2, β^2 + γ^2, (3α^2 + 4β^2) / 3, α^2 + β^2,
            (3α^2 + 4β^2 + 3γ^2) / 3, α^2 + β^2 + γ^2, α^2 + β^2 + γ^2, 9β^2)
end
