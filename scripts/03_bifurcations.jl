# # Bifurcations, attractors and the edge state
#
# We wrap the same right-hand side `mfe!` in a BifurcationKit.jl problem, continue
# equilibria in ``Re`` (Jacobians by automatic differentiation, BifurcationKit's default),
# detect saddle-node and Hopf points, continue the periodic orbits born at the Hopf points
# by collocation, and compare with Moehlis, Faisst & Eckhardt, *SIAM J. Appl. Dyn. Syst.*
# **4**, 352 (2005) (SIADS below). We then follow the attractors of the minimal flow unit,
# estimate the largest Lyapunov exponent, and compute the edge state by bisection.
#
# The continuations take about two minutes; the attractor scans a few more. Results used
# in the figures are cached under `docs/src/assets/data`.

using LinearAlgebra, ForwardDiff, OrdinaryDiffEq, Random, Statistics, CairoMakie
using BifurcationKit
const BK = BifurcationKit
include(joinpath(get(ENV, "MFE_ROOT", joinpath(@__DIR__, "..")), "src", "setup.jl"))

# ## Wrapping the model
#
# `ODEBifProblem` expects `F!(dx, x, p)`; we pass a thin wrapper around `mfe!`, and the lens
# `@optic _.Re` selects the continuation parameter in `MFEParams`.

mfe_bk!(da, a, p) = (mfe!(da, a, p, 0.0); da)
record_eq(x, p; k...) = (E = fluctuation_energy(x), a1 = x[1])
bifprob(a, p) = ODEBifProblem(mfe_bk!, a, p, (@optic _.Re); record_from_solution = record_eq)

# Nontrivial equilibria are found by Newton's method started from points of a turbulent
# trajectory. For ``L_x = 4\pi``, ``L_z = 2\pi`` at ``Re = 400`` this gives eight of them,
# two groups of four related by the symmetries ``T_{L_x/2}`` and ``T_{L_z/2}`` (SIADS eqs
# (3.24)–(3.25)):

function search_equilibria(p; Em1s = (0.2, 0.5, 1.0), tmax = 3000.0)
    found = Vector{Float64}[]
    for e in Em1s
        sol = solve(ODEProblem(mfe!, njp_initial_condition(e), (0.0, tmax), p), Tsit5(); saveat = 5.0)
        for u in sol.u
            a, ok = find_equilibrium(u, p)
            ok && fluctuation_energy(a) > 1e-6 && all(norm(a - b) > 1e-6 for b in found) && push!(found, a)
        end
    end
    return found
end
p_nbc = MFEParams(; Re = 400.0, NBC_DOMAIN...)
eqs_nbc = search_equilibria(p_nbc)
[(E = round(fluctuation_energy(a), digits = 4),
  unstable = count(>(0), real.(eigvals(ForwardDiff.jacobian(x -> mfe(x, p_nbc), a))))) for a in eqs_nbc]

# ## Continuation of equilibria and periodic orbits
#
# One representative of each symmetry group suffices; the others give identical curves of
# energy against ``Re``. The two groups meet at the saddle-node.

opts_eq(; p_min, p_max, ds) = ContinuationPar(; p_min, p_max, ds, dsmax = 1.0, dsmin = 1e-6,
    max_steps = 2000, detect_bifurcation = 3, n_inversion = 8, nev = 9,
    newton_options = NewtonPar(tol = 1e-10, max_iterations = 20))

@views function record_po(x, p; state, iter, k...)
    xtt = BK.get_periodic_orbit(p.prob, x, BK.getparams(iter, state))
    E = [fluctuation_energy(xtt[:, j]) for j in axes(xtt, 2)]
    return (Emax = maximum(E), Emin = minimum(E), T = getperiod(p.prob, x, p.p))
end

function continue_po(br; p_min, p_max)
    ih = findfirst(sp -> sp.type == :hopf, br.specialpoint)
    opts = ContinuationPar(br.contparams; p_min, p_max, ds = -0.01, dsmax = 2.0, dsmin = 1e-6,
        max_steps = 400, detect_bifurcation = 3, tol_stability = 1e-5,
        newton_options = NewtonPar(tol = 1e-9, max_iterations = 15))
    return continuation(br, ih, opts,
        Collocation(40, 4; meshadapt = true, jacobian = BK.DenseAnalyticalInplace());
        record_from_solution = record_po, linear_algo = BK.COPBLS(), normC = norminf, δp = -0.01)
end

## matrices for the cache: equilibria [Re E a1 n_unstable], periodic orbits [Re Emin Emax T stable],
## special points [Re code], code 1 = fold (bp detected at the turning point), 2 = Hopf,
## 3 = period doubling, 4 = fold of periodic orbits, 5 = other
spcode(t) = t == :bp ? 1 : t == :hopf ? 2 : t == :pd ? 3 : t == :fold ? 4 : 5
eqmatrix(br) = hcat(br.branch.param, br.branch.E, br.branch.a1, br.n_unstable[1:length(br.branch)])
pomatrix(br) = hcat(br.branch.param, br.branch.Emin, br.branch.Emax, br.branch.T, br.stable[1:length(br.branch)])
spmatrix(br) = isempty(br.specialpoint) ? zeros(0, 2) :
    reduce(vcat, [[sp.param spcode(sp.type)] for sp in br.specialpoint if sp.type != :endpoint])

function bifurcation_data(dom, a0, Re0; p_min, p_max, po_min, po_max)
    p = MFEParams(; Re = Re0, dom...)
    br = continuation(bifprob(a0, p), PALC(tangent = Bordered()), opts_eq(; p_min, p_max, ds = -0.5))
    brpo = continue_po(br; p_min = po_min, p_max = po_max)
    return br, brpo
end

function cached_bifurcations(name, dom, Re0, a0f; kw...)
    files = ("eq", "po", "sp_eq", "sp_po") .|> s -> "bif_$(name)_$(s)"
    if !recompute() && all(f -> isfile(joinpath(ASSETS, "data", f * ".csv")), files)
        return map(f -> cached(() -> error(), f), files)
    end
    br, brpo = bifurcation_data(dom, a0f(), Re0; kw...)
    return (cached(() -> eqmatrix(br), files[1]), cached(() -> pomatrix(brpo), files[2]),
            cached(() -> spmatrix(br), files[3]), cached(() -> spmatrix(brpo), files[4]))
end

if recompute()
    ENV["MFE_RECOMPUTE"] = "true"
end
eq_nbc, po_nbc, sp_nbc, sppo_nbc = cached_bifurcations("nbc", NBC_DOMAIN, 400.0,
    () -> eqs_nbc[argmax(fluctuation_energy.(eqs_nbc))];
    p_min = 150.0, p_max = 600.0, po_min = 150.0, po_max = 400.0)
eq_mfu, po_mfu, sp_mfu, sppo_mfu = cached_bifurcations("mfu", MFU_DOMAIN, 1000.0,
    () -> (eqs = search_equilibria(MFEParams(; Re = 1000.0, MFU_DOMAIN...)); eqs[argmax(fluctuation_energy.(eqs))]);
    p_min = 600.0, p_max = 1200.0, po_min = 300.0, po_max = 900.0);

# ## Comparison with SIADS 2005
#
# The fold of the equilibria is reported by BifurcationKit as a branch point (`bp`), since
# it is detected from a real eigenvalue crossing zero; it is the turning point of the
# branch, i.e. the saddle-node. Near the fold the eigenvalue can cross several times
# within one step, so the first detection is taken. The periodic orbits born at the Hopf
# point turn back at a fold (minimum ``Re`` on the branch).

first_of(sp, code) = (i = findfirst(==(code), sp[:, 2]); i === nothing ? NaN : sp[i, 1])
fold_po(po) = (i = argmin(po[:, 1]); (po[i, 1], po[i, 4]))
rows = [
    ("Lx=4π, Lz=2π: saddle-node of equilibria", "Re = 308.16", @sprintf("Re = %.2f", first_of(sp_nbc, 1))),
    ("Lx=4π, Lz=2π: Hopf", "Re = 310.34", @sprintf("Re = %.2f", first_of(sp_nbc, 2))),
    ("Lx=4π, Lz=2π: fold of the Hopf periodic orbits", "Re = 186.99, T = 46.80",
        @sprintf("Re = %.2f, T = %.2f", fold_po(po_nbc)...)),
    ("Lx=1.75π, Lz=1.2π: saddle-node of equilibria", "Re = 794.51", @sprintf("Re = %.2f", first_of(sp_mfu, 1))),
    ("Lx=1.75π, Lz=1.2π: Hopf", "Re = 795.12", @sprintf("Re = %.2f", first_of(sp_mfu, 2))),
    ("Lx=1.75π, Lz=1.2π: fold of the Hopf periodic orbits", "Re = 452.99, T = 51.35",
        @sprintf("Re = %.2f, T = %.2f", fold_po(po_mfu)...)),
]
println(rpad("", 52), rpad("SIADS 2005", 26), "this repository")
for (what, paper, ours) in rows
    println(rpad(what, 52), rpad(paper, 26), ours)
end

# Period-doubling points found on the periodic-orbit branches (SIADS reports a period
# doubling cascade on the branch for ``L_x = 4\pi``, without listing the values):

(nbc = sppo_nbc[sppo_nbc[:, 2] .== 3, 1], mfu = sppo_mfu[sppo_mfu[:, 2] .== 3, 1])

# ## Bifurcation diagrams

function plot_bifurcations!(ax, eq, po, sp)
    for (k, col) in ((0, :black),)
        st = eq[:, 4] .== 0
        lines!(ax, eq[:, 1], eq[:, 2]; color = :gray50, label = "equilibria (unstable)")
    end
    lines!(ax, po[:, 1], po[:, 3]; color = :steelblue, label = "periodic orbits, max/min E")
    lines!(ax, po[:, 1], po[:, 2]; color = :steelblue)
    stable = po[:, 5] .== 1
    scatter!(ax, po[stable, 1], po[stable, 3]; color = :steelblue, markersize = 4, label = "stable periodic orbit")
    for (code, m, c, l) in ((1, :diamond, :red, "saddle-node"), (2, :circle, :orange, "Hopf"))
        i = findfirst(==(code), sp[:, 2])
        i === nothing && continue
        j = argmin(abs.(eq[:, 1] .- sp[i, 1]))
        scatter!(ax, [sp[i, 1]], [eq[j, 2]]; marker = m, color = c, markersize = 14, label = l)
    end
    i = argmin(po[:, 1])
    scatter!(ax, [po[i, 1]], [po[i, 3]]; marker = :diamond, color = :purple, markersize = 14, label = "fold of periodic orbits")
    hlines!(ax, [0.0]; color = :black, linewidth = 2, label = "laminar (stable)")
end
fig = Figure(size = (1100, 480))
ax1 = Axis(fig[1, 1]; xlabel = "Re", ylabel = "fluctuation energy E", title = "Lx = 4π, Lz = 2π")
plot_bifurcations!(ax1, eq_nbc, po_nbc, sp_nbc)
ax2 = Axis(fig[1, 2]; xlabel = "Re", ylabel = "fluctuation energy E", title = "Lx = 1.75π, Lz = 1.2π")
plot_bifurcations!(ax2, eq_mfu, po_mfu, sp_mfu)
Legend(fig[1, 3], ax1; framevisible = false)
savefig("bifurcation_diagram.png", fig); nothing #hide

# ![](../assets/bifurcation_diagram.png)

# All equilibria born at the saddle-node are unstable (the upper branch has 3, the lower
# 4 unstable directions at ``Re = 400`` for ``L_x = 4\pi``), as in SIADS. The energy is the
# fluctuation energy relative to the laminar state (SIADS eq. (4.1)).

# ## Attractors of the minimal flow unit
#
# SIADS §4.2 reports a periodic orbit in the minimal flow unit that is stable between
# torus bifurcations at ``Re = 354.57`` and ``Re = 507.40``, with period ``T = 112.51`` at
# ``Re = 400``; below ``354.57``, a quasi-periodic attractor, then chaos with periodic
# windows, destroyed in a boundary crisis at ``Re \approx 335``. We start on the periodic
# orbit at ``Re = 400`` and follow the attractor adiabatically, recording ``a_1`` on the
# Poincaré section ``a_2 = 0``, ``\dot a_2 < 0`` (as in SIADS Fig. 21).

MFU(Re) = MFEParams(; Re, MFU_DOMAIN...)
pts, ts, _ = poincare_section(njp_initial_condition(0.2), MFU(400.0); tmax = 8000.0, tskip = 5000.0)
(period_at_400 = mean(diff(ts)), siads = 112.51)

function attractor_scan(Res; ttrans = 3000.0, trecord = 3000.0)
    _, _, u = poincare_section(njp_initial_condition(0.2), MFU(400.0); tmax = 5000.0)
    out = zeros(0, 2)
    for Re in Res
        pts, _, u2 = poincare_section(u, MFU(Re); tmax = ttrans + trecord, tskip = ttrans)
        if laminar_criterion(u2) || isempty(pts)
            out = vcat(out, [Re NaN])     # attractor lost
            break
        end
        u = u2
        out = vcat(out, hcat(fill(Re, length(pts)), [q[1] for q in pts]))
    end
    return out
end
scan_down = cached(() -> attractor_scan(range(400, 325, step = -0.25)), "attractor_scan_down")
scan_up = cached(() -> attractor_scan(range(400, 530, step = 0.5)), "attractor_scan_up")

# Largest Lyapunov exponent along the same adiabatic path:

function lyapunov_scan(Res)
    _, _, u = poincare_section(njp_initial_condition(0.2), MFU(400.0); tmax = 5000.0)
    out = zeros(0, 2)
    for Re in Res
        _, _, u2 = poincare_section(u, MFU(Re); tmax = 3000.0)
        laminar_criterion(u2) && break
        u = u2
        λ, lam = lyapunov_exponent(u, MFU(Re); T = 3e4, ttrans = 0.0)
        lam && break
        out = vcat(out, [Re λ])
    end
    return out
end
lyap = cached(() -> lyapunov_scan(range(400, 326, step = -1)), "lyapunov_scan")

fig = Figure(size = (1000, 700))
ax = Axis(fig[1, 1]; ylabel = "a₁ on the section a₂ = 0", title = "Minimal flow unit: attractor followed from Re = 400 (cf. SIADS Fig. 21)")
scatter!(ax, scan_down[:, 1], scan_down[:, 2]; markersize = 1.5, color = :black)
scatter!(ax, scan_up[:, 1], scan_up[:, 2]; markersize = 1.5, color = :black)
vlines!(ax, [354.57, 507.40]; color = :orange, linestyle = :dash, label = "torus bifurcations (SIADS)")
vlines!(ax, [335.0]; color = :red, linestyle = :dot, label = "boundary crisis ≈ 335 (SIADS)")
axislegend(ax; position = :rt)
ax2 = Axis(fig[2, 1]; xlabel = "Re", ylabel = "largest Lyapunov exponent")
scatterlines!(ax2, lyap[:, 1], lyap[:, 2]; markersize = 5)
hlines!(ax2, [0.0]; color = :gray)
linkxaxes!(ax, ax2)
savefig("mfu_attractors.png", fig); nothing #hide

# ![](../assets/mfu_attractors.png)

# Lowest ``Re`` at which the followed attractor survives, and the range of ``Re`` with a
# positive Lyapunov exponent (above an estimated statistical error of ``2\times10^{-4}``):

lost_down = scan_down[findfirst(isnan, scan_down[:, 2]), 1]
(attractor_survives_down_to = minimum(scan_down[.!isnan.(scan_down[:, 2]), 1]),
 lost_at = lost_down,
 chaotic_Re = extrema(lyap[lyap[:, 2] .> 2e-4, 1]))

# The followed attractor goes from the periodic orbit (one point on the section) to a
# quasi-periodic band below ``Re \approx 355``, then to chaos with a positive Lyapunov
# exponent, interrupted by periodic windows (the dip near ``Re = 338`` matches the stable
# periodic orbit of SIADS Fig. 20(c) at ``Re = 338.25``), and is lost a little below
# ``Re = 332``. SIADS places the boundary crisis at ``Re \approx 335``; where we lose the
# attractor depends on the transient time allowed, since long chaotic transients persist
# past a crisis.
#
# Going up, the periodic orbit becomes a quasi-periodic band near ``Re \approx 505``,
# consistent with the torus bifurcation at ``507.40``, and the followed attractor is lost
# at ``Re \approx 518``, beyond which the trajectory decays to the laminar state. The
# papers do not describe the range above ``507.40``, so this upper limit is our own
# observation. Altogether, the minimal flow unit has a non-laminar attractor for roughly
# ``332 \lesssim Re \lesssim 518`` along this path, chaotic only for
# ``Re \lesssim 352``.
#
# These are not the periodic orbits of the bifurcation diagram above: SIADS finds the
# stable orbit on a branch born at a saddle-node at ``Re = 122.37``, not connected to the
# equilibria, which we have not continued here.

# ## The edge state
#
# The boundary of the laminar basin is found by bisection between a perturbation that
# decays and one that becomes turbulent, classifying states by whether the fluctuation
# energy makes a large excursion (``E > 0.1``) before decaying (``E < 10^{-5}``), and then
# tracked in time by repeated bisection (Skufca, Yorke & Eckhardt, *Phys. Rev. Lett.*
# **96**, 174101 (2006)). We start along the direction of the NJP initial condition,
# ``a_2 = a_3 = a_4 = a_5``, added to the laminar state.

function edge_data(Re; dom = NBC_DOMAIN, nsteps = 200, Δt = 5.0)
    p = MFEParams(; Re, dom...)
    lam, d = laminar_state(), njp_initial_condition(1.0)
    states, times = edge_track(lam + 1e-3d, lam + 0.5d, p; nsteps, Δt)
    return hcat(times, reduce(vcat, permutedims.(states)))
end
Re_edge = [305.0, 310.0, 320.0, 350.0, 400.0]
edges = Dict(Re => cached(() -> edge_data(Re), "edge_nbc_Re$(Int(Re))") for Re in Re_edge)

fig = Figure(size = (1000, 420))
ax = Axis(fig[1, 1]; xlabel = "t", ylabel = "fluctuation energy", title = "Edge trajectories, Lx = 4π, Lz = 2π", yscale = log10)
for Re in Re_edge
    X = edges[Re]
    lines!(ax, X[:, 1], [fluctuation_energy(X[i, 2:10]) for i in axes(X, 1)]; label = "Re = $(Int(Re))")
end
axislegend(ax; position = :rb)
ax2 = Axis(fig[1, 2]; xlabel = "a₆", ylabel = "a₇", title = "Edge trajectories projected on (a₆, a₇)")
for Re in Re_edge
    X = edges[Re][50:end, :]
    lines!(ax2, X[:, 7], X[:, 8]; label = "Re = $(Int(Re))")
end
savefig("edge_states.png", fig); nothing #hide

# ![](../assets/edge_states.png)

# Summary of the edge trajectories (after discarding the first 250 time units): mean and
# range of the fluctuation energy and the dominant modes.

[(Re = Re, E_mean = mean(fluctuation_energy(X[i, 2:10]) for i in 50:size(X, 1)),
  E_range = extrema(fluctuation_energy(X[i, 2:10]) for i in 50:size(X, 1)),
  mean_abs_a = round.(vec(mean(abs.(X[50:end, 2:10] .- laminar_state()'); dims = 1)), digits = 3))
 for (Re, X) in sort(collect(edges); by = first)]

# Observations, with ``L_x = 4\pi``, ``L_z = 2\pi``:
#
# - The edge state is **not** one of the equilibria born at the saddle-node: it lies much
#   closer to the laminar state (fluctuation energy of order ``10^{-2}``, against ``0.6``
#   for the equilibria), and Newton's method started on it does not converge to an
#   equilibrium nearby. Its energy decreases with ``Re``.
# - It is dominated by the spanwise flows and normal vortex modes (``a_4`` to ``a_7``), and
#   it oscillates without settling to an equilibrium during the tracking time. Whether it
#   is a periodic orbit, as found by Skufca, Yorke & Eckhardt for ``Re \lesssim 402`` in a
#   *different* nine-variable model (from Schmiegel's thesis, with width ``\pi`` and length
#   ``2\pi``), or a chaotic edge state, we have not settled here.
#
# The final tracked edge states are used as base points of the slices on the Lagrangian
# descriptor page.
