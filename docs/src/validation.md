# Validation

The tests live in `test/` and run with `julia --project=. test/runtests.jl` (about two
minutes; `MFE_FAST_TESTS=1` skips the slower dynamical checks, about one minute).

## The model (`test/test_model.jl`)

| check | how |
|:----- |:--- |
| basis fields are orthogonal with ``\langle \mathbf u_n,\mathbf u_m\rangle = 2L_xL_z\delta_{nm}`` (NJP eq. 17) | quadrature, three domains |
| basis fields are divergence-free and satisfy the free-slip conditions (NJP eq. 4) | ForwardDiff at random points |
| explicit equations (NJP eqs 21–32) = Galerkin projection of the basis fields | 100 random states, three domains, two ``Re``, to ``10^{-11}`` |
| viscous operator is diagonal with the decay rates of the paper | from the projection |
| laminar state ``a = (1,0,\dots,0)`` is a fixed point | ``Re`` from 1 to ``10^4`` |
| nonlinear terms conserve energy, ``\sum_i a_i N_{ijk}a_ja_k = 0`` | random states |
| laminar state is linearly stable | Jacobian eigenvalues, ``Re`` up to ``10^5`` |
| `mfe!` is type stable and allocation-free, and accepts a dual-number ``Re`` | `@inferred`, `@allocated` |

## Dynamics (`test/test_dynamics.jl`)

| check | paper | result |
|:----- |:----- |:------ |
| subcritical transition, minimal flow unit, ``Re = 400``: small perturbations decay, finite ones stay turbulent | NJP §3–4 | yes |
| stable periodic orbit at ``Re = 400``, minimal flow unit | SIADS §4.2, ``T = 112.51`` | ``T = 112.50`` (within ``10^{-3}``) |
| chaotic attractor at ``Re = 345`` (positive largest Lyapunov exponent), gone at ``Re = 320`` | SIADS §4.2: chaos below ``354.57``, crisis at ``\approx 335`` | ``\lambda \approx 1.5\times10^{-3}``; decays at 320 |
| zero largest Lyapunov exponent on the periodic orbit at ``Re = 400`` | — | ``\lvert\lambda\rvert < 2\times10^{-4}`` |
| exponential lifetime distributions, median lifetime increasing with ``Re`` | NJP §5, Fig. 7 | ``\log P(t)`` linear (``R^2 > 0.97``) at ``Re = 200, 250`` |
| edge tracking stays on the edge, at energies between laminar and turbulent | Skufca et al. 2006 (method) | 1000 time units at ``Re = 400`` |

## Bifurcations (`test/test_bifurcation.jl`)

| check | SIADS 2005 | BifurcationKit here |
|:----- |:---------- |:------------------- |
| ``L_x = 4\pi, L_z = 2\pi``: saddle-node of equilibria | 308.16 | 308.16 |
| ``L_x = 4\pi, L_z = 2\pi``: Hopf | 310.34 | 310.34 |

The full comparison, including the periodic-orbit folds and the minimal flow unit, is on
the [bifurcation page](generated/03_bifurcations.md).

## Claims we could not confirm or that differ from a source

- The range "chaotic attractor for ``335 \lesssim Re \lesssim 515``" does not appear in the
  papers. SIADS reports, for the minimal flow unit, a stable periodic orbit between torus
  bifurcations at ``Re = 354.57`` and ``507.40``, quasi-periodic and chaotic attractors below
  354.57, and a boundary crisis at ``Re \approx 335``. Following the attractor, we find a
  non-laminar attractor for roughly ``332 \lesssim Re \lesssim 518``, chaotic only below
  ``Re \approx 352``.
- The papers do not state their criterion for laminarisation in the lifetime
  computations; we use fluctuation energy below ``10^{-2}``.
- NJP §5 sets ``a_1 = 0`` in the initial conditions of the lifetime maps (not ``a_1 = 1``),
  so ``\sum_i a_i^2 = E - 1``; the nonzero lifetimes at ``E - 1 \to 0`` in NJP Figs 6 and 9
  are consistent with this reading, and we follow it.
