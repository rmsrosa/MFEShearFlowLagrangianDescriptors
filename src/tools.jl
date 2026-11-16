# Analysis helpers shared by scripts and tests: initial conditions, lifetimes,
# Poincaré sections, largest Lyapunov exponent, Newton search for equilibria and
# edge-state bisection.
#
# Requires `using OrdinaryDiffEq, ForwardDiff, LinearAlgebra, Random` and
# `include("mfe.jl")` beforehand.

"""
    njp_initial_condition(Em1)

Initial condition of NJP §5 (Figs 6, 8, 9): `a₂ = a₃ = a₄ = a₅ = √(E−1)/2`, all other
amplitudes zero. Note that the paper sets `a₁ = 0` (not 1), so `Σ aᵢ² = E − 1`.
"""
njp_initial_condition(Em1) = [0.0, fill(sqrt(Em1) / 2, 4)..., 0.0, 0.0, 0.0, 0.0]

"""
    random_shell_state(rng, E)

Uniformly random point on the sphere `Σ aᵢ² = E` in ℝ⁹ (the "energy shell" used for
the lifetime statistics in NJP §5, where `E = 0.3`).
"""
function random_shell_state(rng, E)
    v = randn(rng, 9)
    return sqrt(E) / norm(v) .* v
end

"""
    laminar_criterion(a; threshold = 1e-2)

Whether `a` counts as laminarised: fluctuation energy (SIADS eq (4.1)) below `threshold`.
The papers do not state their criterion; this one is our choice (lifetimes depend on
it only through an additive O(Re) relaxation time).
"""
laminar_criterion(a; threshold = 1e-2) = fluctuation_energy(a) < threshold

"""
    lifetime(a0, p; tmax = 1000.0, threshold = 1e-2, alg = Tsit5(), tol = 1e-8)

Time until the trajectory from `a0` laminarises (see [`laminar_criterion`](@ref)),
capped at `tmax`.
"""
function lifetime(a0, p::MFEParams; tmax = 1000.0, threshold = 1e-2, alg = Tsit5(), tol = 1e-8)
    cb = DiscreteCallback((u, t, i) -> laminar_criterion(u; threshold), terminate!;
                          save_positions = (false, false))
    prob = ODEProblem(mfe!, collect(float.(a0)), (0.0, tmax), p)
    sol = solve(prob, alg; callback = cb, save_everystep = false, save_start = false,
                abstol = tol, reltol = tol)
    return sol.t[end]
end

"""
    poincare_section(a0, p; tmax, tskip = 0, alg = Vern9(), tol = 1e-10)

Intersections with the hyperplane `a₂ = 0` with `ȧ₂ < 0` (the section used in SIADS
§4.2), recorded after `tskip`. Returns `(points, times, final_state)`.
"""
function poincare_section(a0, p::MFEParams; tmax, tskip = 0.0, alg = Vern9(), tol = 1e-10)
    pts, ts = Vector{Float64}[], Float64[]
    affect!(i) = i.t ≥ tskip && (push!(pts, copy(i.u)); push!(ts, i.t))
    cb = ContinuousCallback((u, t, i) -> u[2], nothing, affect!; save_positions = (false, false))
    prob = ODEProblem(mfe!, collect(float.(a0)), (0.0, tmax), p)
    sol = solve(prob, alg; callback = cb, save_everystep = false, abstol = tol, reltol = tol)
    return pts, ts, sol.u[end]
end

# Tangent dynamics: state u = [a; v], u̇ = [f(a); J(a) v], with J v from one dual evaluation.
# The dual buffers live in a callable struct rather than in the ODE parameters: DiffEqBase
# would otherwise promote u0 to dual numbers.
struct TangentRHS{P,D}
    p::P
    ad::Vector{D}
    dd::Vector{D}
end
function TangentRHS(p::MFEParams)
    D = ForwardDiff.Dual{Nothing,Float64,1}
    return TangentRHS(p, zeros(D, 9), zeros(D, 9))
end

function (r::TangentRHS)(du, u, _, t)
    @inbounds for i in 1:9
        r.ad[i] = ForwardDiff.Dual{Nothing}(u[i], u[9+i])
    end
    mfe!(r.dd, r.ad, r.p, t)
    @inbounds for i in 1:9
        du[i] = ForwardDiff.value(r.dd[i])
        du[9+i] = ForwardDiff.partials(r.dd[i], 1)
    end
    return nothing
end

"""
    lyapunov_exponent(a0, p; T = 2e4, ttrans = 1e3, τ = 10.0, rng = Xoshiro(0))

Largest Lyapunov exponent by integrating the tangent linear system and renormalising
every `τ` time units (Benettin et al.). Returns `(λ, laminarised::Bool)`. If the
trajectory laminarises, `λ` is meaningless and `laminarised` is `true`.
"""
function lyapunov_exponent(a0, p::MFEParams; T = 2e4, ttrans = 1e3, τ = 10.0,
                           rng = Xoshiro(0), tol = 1e-9)
    # discard transient
    sol0 = solve(ODEProblem(mfe!, collect(float.(a0)), (0.0, ttrans), p), Vern9();
                 save_everystep = false, abstol = tol, reltol = tol)
    a = sol0.u[end]
    v = normalize(randn(rng, 9))
    u0 = vcat(a, v)
    integ = init(ODEProblem(TangentRHS(p), u0, (0.0, T)), Vern9();
                 save_everystep = false, abstol = tol, reltol = tol)
    logsum = 0.0
    nsteps = floor(Int, T / τ)
    for _ in 1:nsteps
        step!(integ, τ, true)
        u = integ.u
        laminar_criterion(view(u, 1:9)) && return (NaN, true)
        nv = norm(view(u, 10:18))
        logsum += log(nv)
        u[10:18] ./= nv
        OrdinaryDiffEq.SciMLBase.u_modified!(integ, true)
    end
    return (logsum / (nsteps * τ), false)
end

"""
    find_equilibrium(a0, p; maxiter = 100, tol = 1e-12)

Damped Newton iteration for `f(a) = 0` with a ForwardDiff Jacobian. Returns
`(a, converged::Bool)`.
"""
function find_equilibrium(a0, p::MFEParams; maxiter = 100, tol = 1e-12)
    a = collect(float.(a0))
    for _ in 1:maxiter
        F = mfe(a, p)
        norm(F) < tol && return (a, true)
        J = ForwardDiff.jacobian(x -> mfe(x, p), a)
        δ = try
            J \ F
        catch
            return (a, false)
        end
        s = 1.0
        while s > 1e-4 && norm(mfe(a - s * δ, p)) > (1 - s / 4) * norm(F)
            s /= 2
        end
        a -= s * δ
    end
    return (a, norm(mfe(a, p)) < 1e-9)
end

"""
    edge_bisection(a_lam, a_turb, p; tmax = 2000.0, tol = 1e-8, kwargs...)

Bisect along the segment between a state `a_lam` that laminarises and a state `a_turb`
that does not (within `tmax`). Returns the pair `(lo, hi)` of states, `|hi - lo| < tol`,
straddling the edge of chaos (the laminar basin boundary).
"""
function edge_bisection(a_lam, a_turb, p::MFEParams; tmax = 2000.0, tol = 1e-8, kwargs...)
    decays(a) = lifetime(a, p; tmax, kwargs...) < tmax
    lo, hi = collect(float.(a_lam)), collect(float.(a_turb))
    @assert decays(lo) && !decays(hi) "endpoints must lie on opposite sides of the edge"
    while norm(hi - lo) > tol
        mid = (lo + hi) / 2
        decays(mid) ? (lo = mid) : (hi = mid)
    end
    return lo, hi
end

"""
    edge_track(a_lam, a_turb, p; nsteps = 50, Δt = 50.0, tmax = 2000.0, tol = 1e-8)

Edge tracking (Skufca, Yorke & Eckhardt 2006): bisect, integrate the bracketing pair
for `Δt`, re-bisect, repeat. Returns the visited edge states and their times.
"""
function edge_track(a_lam, a_turb, p::MFEParams; nsteps = 50, Δt = 50.0, tmax = 2000.0,
                    tol = 1e-8, alg = Vern9())
    lo, hi = edge_bisection(a_lam, a_turb, p; tmax, tol)
    states, times = Vector{Float64}[], Float64[]
    t = 0.0
    for _ in 1:nsteps
        sol = solve(ODEProblem(mfe!, hi, (0.0, Δt), p), alg; save_everystep = false,
                    abstol = 1e-11, reltol = 1e-11)
        sol2 = solve(ODEProblem(mfe!, lo, (0.0, Δt), p), alg; save_everystep = false,
                     abstol = 1e-11, reltol = 1e-11)
        t += Δt
        lo2, hi2 = sol2.u[end], sol.u[end]
        # re-bracket: if both drifted to the same side, fall back to the previous pair
        lo, hi = try
            edge_bisection(lo2, hi2, p; tmax, tol)
        catch
            break
        end
        push!(states, (lo + hi) / 2)
        push!(times, t)
    end
    return states, times
end
