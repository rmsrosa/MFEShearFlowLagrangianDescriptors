# Self-contained draft of the example page proposed for the LagrangianDescriptors.jl docs.
# Runs without this repository: julia --project=. -t 4 proposals/mfe_example.jl

using OrdinaryDiffEq, CairoMakie, LagrangianDescriptors
using LinearAlgebra: norm

function mfe!(da, a, p, t)
    (; Re, α, β, γ) = p
    kαγ, kβγ, kαβγ = sqrt(α^2 + γ^2), sqrt(β^2 + γ^2), sqrt(α^2 + β^2 + γ^2)
    a1, a2, a3, a4, a5, a6, a7, a8, a9 = a
    s32, s6 = sqrt(3 / 2), sqrt(6)
    da[1] = β^2 / Re * (1 - a1) - s32 * β * γ / kαβγ * a6 * a8 + s32 * β * γ / kβγ * a2 * a3
    da[2] = -(4β^2 / 3 + γ^2) * a2 / Re + 5sqrt(2) / (3sqrt(3)) * γ^2 / kαγ * a4 * a6 -
            γ^2 / (s6 * kαγ) * a5 * a7 - α * β * γ / (s6 * kαγ * kαβγ) * a5 * a8 -
            s32 * β * γ / kβγ * (a1 * a3 + a3 * a9)
    da[3] = -(β^2 + γ^2) / Re * a3 + 2 / s6 * α * β * γ / (kαγ * kβγ) * (a4 * a7 + a5 * a6) +
            (β^2 * (3α^2 + γ^2) - 3γ^2 * (α^2 + γ^2)) / (s6 * kαγ * kβγ * kαβγ) * a4 * a8
    da[4] = -(3α^2 + 4β^2) / (3Re) * a4 - α / s6 * a1 * a5 - 10 / (3s6) * α^2 / kαγ * a2 * a6 -
            s32 * α * β * γ / (kαγ * kβγ) * a3 * a7 - s32 * α^2 * β^2 / (kαγ * kβγ * kαβγ) * a3 * a8 -
            α / s6 * a5 * a9
    da[5] = -(α^2 + β^2) / Re * a5 + α / s6 * a1 * a4 + α^2 / (s6 * kαγ) * a2 * a7 -
            α * β * γ / (s6 * kαγ * kαβγ) * a2 * a8 + α / s6 * a4 * a9 +
            2 / s6 * α * β * γ / (kαγ * kβγ) * a3 * a6
    da[6] = -(3α^2 + 4β^2 + 3γ^2) / (3Re) * a6 + α / s6 * a1 * a7 + s32 * β * γ / kαβγ * a1 * a8 +
            10 / (3s6) * (α^2 - γ^2) / kαγ * a2 * a4 - 2sqrt(2 / 3) * α * β * γ / (kαγ * kβγ) * a3 * a5 +
            α / s6 * a7 * a9 + s32 * β * γ / kαβγ * a8 * a9
    da[7] = -(α^2 + β^2 + γ^2) / Re * a7 - α / s6 * (a1 * a6 + a6 * a9) +
            1 / s6 * (γ^2 - α^2) / kαγ * a2 * a5 + 1 / s6 * α * β * γ / (kαγ * kβγ) * a3 * a4
    da[8] = -(α^2 + β^2 + γ^2) / Re * a8 + 2 / s6 * α * β * γ / (kαγ * kαβγ) * a2 * a5 +
            γ^2 * (3α^2 - β^2 + 3γ^2) / (s6 * kαγ * kβγ * kαβγ) * a3 * a4
    da[9] = -9β^2 / Re * a9 + s32 * β * γ / kβγ * a2 * a3 - s32 * β * γ / kαβγ * a6 * a8
    return nothing
end

p = (Re = 320.0, α = 0.5, β = π / 2, γ = 1.0)          # Lx = 4π, Lz = 2π
edge = [0.956906786337, -0.031470155426, 0.00789835903924, -0.0407712627654, 0.0490210360762,
        -0.088480880902, -0.110164417566, -0.000905441222403, -0.00437376488416]

prob = ODEProblem(mfe!, edge, (0.0, 150.0), p)
M(du, u, p, t) = norm(du)
a4s = range(edge[4] - 0.1, edge[4] + 0.1, length = 101)
a5s = range(edge[5] - 0.1, edge[5] + 0.1, length = 101)
uu0 = [(u = copy(edge); u[4] = x; u[5] = y; u) for y in a5s, x in a4s]
t = @elapsed lagsol = solve(LagrangianDescriptorProblem(prob, M, uu0; direction = :forward), Tsit5();
                            max_trajectory_value = 5.0)
println("solve time: ", round(t, digits = 1), " s")

fig, ax, plt = heatmap(lagsol, :forward; idxs = (4, 5),
    axis = (title = "MFE model, Re = 320: slice through the edge state", xlabel = "a₄", ylabel = "a₅"))
scatter!(ax, [edge[4]], [edge[5]]; color = :cyan, marker = :xcross, markersize = 14)
Colorbar(fig[1, 2], plt; label = "forward arclength, τ = 150")
save(joinpath(@__DIR__, "mfe_example.png"), fig)
