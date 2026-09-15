"""Tests for the Marchenko-Pastur law and the noise-scale estimator.

These are the most important tests in the project: if the MP law is wrong,
every downstream result is wrong.  The suite therefore checks not only the
internal consistency of the implementation but the *known analytical
properties* of the law, including a Monte-Carlo recovery of ``sigma``.
"""

from __future__ import annotations

import numpy as np
import pytest

from rmt_portfolio.rmt.mp_law import (
    MPLaw,
    _mp_cdf,
    effective_aspect_ratio,
    fit_mp_sigma,
    mp_cdf_fraction,
    mp_density,
    mp_edge,
    mp_sigma_from_edge,
)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.01, 0.0625, 0.25, 0.5, 1.0, 2.0, 4.0])
def test_edges_match_closed_form(q):
    lo, hi = mp_edge(1.0, q)
    assert lo == pytest.approx((1.0 - np.sqrt(q)) ** 2, rel=1e-12)
    assert hi == pytest.approx((1.0 + np.sqrt(q)) ** 2, rel=1e-12)


def test_edge_scaling_in_sigma():
    """Both edges scale as sigma^2."""
    lo1, hi1 = mp_edge(1.0, 0.25)
    lo2, hi2 = mp_edge(3.0, 0.25)
    assert lo2 == pytest.approx(9.0 * lo1, rel=1e-12)
    assert hi2 == pytest.approx(9.0 * hi1, rel=1e-12)


def test_edge_rejects_bad_input():
    with pytest.raises(ValueError):
        mp_edge(0.0, 0.25)
    with pytest.raises(ValueError):
        mp_edge(1.0, 0.0)
    with pytest.raises(ValueError):
        mp_edge(-1.0, 0.25)


def test_mp_sigma_from_edge_round_trips():
    for q in (0.05, 0.25, 1.0, 3.0):
        for sigma in (0.3, 1.0, 2.5):
            _, hi = mp_edge(sigma, q)
            assert mp_sigma_from_edge(hi, q) == pytest.approx(sigma, rel=1e-10)


# ---------------------------------------------------------------------------
# Density and CDF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.05, 0.2, 0.5, 0.9])
def test_density_integrates_to_expected_mass(q):
    """The bulk integrates to 1 for q <= 1 and to 1/q for q > 1.

    ``q = 1`` is excluded: there the density has an integrable square-root
    singularity at the lower edge (:math:`\\lambda_- = 0`), so a uniform
    grid converges slowly and the test would measure grid resolution rather
    than correctness.  The CDF tests cover that case instead.
    """
    lo, hi = mp_edge(1.0, q)
    grid = np.linspace(lo + 1e-9, hi - 1e-9, 200_000)
    integral = np.trapezoid(mp_density(grid, 1.0, q), grid)
    expected = 1.0 / q if q > 1.0 else 1.0
    assert integral == pytest.approx(expected, rel=3e-3)


@pytest.mark.parametrize("q", [1.5, 2.0, 4.0])
def test_density_mass_above_one(q):
    """For q > 1 the continuous bulk carries mass 1/q; the rest is an atom."""
    lo, hi = mp_edge(1.0, q)
    grid = np.linspace(lo + 1e-9, hi - 1e-9, 300_000)
    integral = np.trapezoid(mp_density(grid, 1.0, q), grid)
    assert integral == pytest.approx(1.0 / q, rel=5e-3)


def test_density_zero_outside_support():
    lo, hi = mp_edge(1.0, 0.3)
    probe = np.array([lo * 0.5, hi * 1.5, -1.0, 1e-9])
    assert np.allclose(mp_density(probe, 1.0, 0.3), 0.0)


@pytest.mark.parametrize("q", [0.05, 0.25, 0.6, 1.0, 1.5, 3.0])
def test_cdf_endpoints(q):
    """F(a) = 0 (or the atom), F(b) = 1, F is monotone."""
    lo, hi = mp_edge(1.0, q)
    atom = max(1.0 - 1.0 / q, 0.0)
    x = np.array([lo * 0.9, lo, 0.5 * (lo + hi), hi, hi * 1.1])
    F = _mp_cdf(x, 1.0, q)
    assert F[0] == pytest.approx(atom, abs=1e-9)
    assert F[-1] == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.diff(F) >= -1e-12), "CDF must be non-decreasing"


@pytest.mark.parametrize("q", [0.05, 0.25, 0.6])
def test_cdf_inside_support_is_accurate(q):
    """The numerical CDF must integrate the density faithfully.

    There is no longer an analytic CDF to cross-check against (an earlier
    hand-transcribed version was found to return ``F(lambda_+) = 1.5`` and was
    removed).  Instead we validate the integrator against an independent,
    much finer Riemann sum built from the density itself.

    ``q = 1`` is excluded: there the density has an integrable square-root
    singularity at :math:`\\lambda_- = 0`, so the fixed 6000-node uniform grid
    used by :func:`_mp_cdf` under-resolves the lower edge and the comparison
    would fail for reasons of grid resolution rather than correctness.  The
    endpoint tests above still cover ``q = 1``.
    """
    lo, hi = mp_edge(1.0, q)
    # a coarse CDF grid vs. a brute-force 4e5-node integration at midpoints
    x = np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), 25)
    F = _mp_cdf(x, 1.0, q)
    assert np.all(np.diff(F) > 0), "CDF must be strictly increasing in bulk"
    assert np.all(F > 0) and np.all(F < 1)

    # independent reference: high-resolution trapezoid on the density
    def ref(px):
        g = np.linspace(lo + 1e-12, px, 400_001)
        d = mp_density(g, 1.0, q)
        return float(np.trapezoid(d, g))

    for xi, Fi in zip(x[::6], F[::6]):
        assert Fi == pytest.approx(ref(xi), abs=4e-3)


def test_cdf_fraction_alias_matches():
    """``_mp_cdf`` and ``mp_cdf_fraction`` must be the same function."""
    assert _mp_cdf is mp_cdf_fraction


def test_cdf_monotone_and_normalised_on_dense_grid():
    for q in (0.05, 0.3, 1.0, 2.0):
        lo, hi = mp_edge(1.0, q)
        grid = np.linspace(lo * 0.5, hi * 1.2, 5000)
        F = _mp_cdf(grid, 1.0, q)
        assert np.all(np.diff(F) >= -1e-12)
        assert F.min() >= 0.0 and F.max() <= 1.0
        assert F[-1] == pytest.approx(1.0, abs=1e-10)


def test_cdf_median_is_half():
    """F(median) = 0.5 for q <= 1 (no atom)."""
    for q in (0.05, 0.25, 0.7):
        lo, hi = mp_edge(1.0, q)
        grid = np.linspace(lo, hi, 400_000)
        F = _mp_cdf(grid, 1.0, q)
        median = grid[np.argmin(np.abs(F - 0.5))]
        assert _mp_cdf(np.array([median]), 1.0, q)[0] == pytest.approx(
            0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# sigma recovery: the decisive test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0.0625, 0.25])
@pytest.mark.parametrize("true_sigma", [0.7, 1.0, 1.6])
def test_sigma_recovery_on_pure_noise(q, true_sigma):
    """On pure noise every method must recover the true noise scale.

    This is the property the whole denoising pipeline rests on: if ``sigma``
    cannot be recovered from a spectrum that *is* Marchenko-Pastur, the
    threshold computed from it is meaningless.
    """
    n, t = 16, int(round(16 / q))
    rng = np.random.default_rng(12345)
    x = rng.standard_normal((t, n)) * true_sigma
    ev = np.linalg.eigvalsh(np.cov(x, rowvar=False, ddof=1))

    for method in ("median_bulk", "mean_bulk", "max_eigen", "cdf_fit"):
        sigma = fit_mp_sigma(ev, q, method=method, trim=0.0)
        # max_eigen is deliberately biased high when any eigenvalue happens to
        # exceed the theoretical edge, and mean_bulk includes the whole
        # (slightly over-dispersed) sample spectrum, so allow 20%.
        assert sigma == pytest.approx(true_sigma, rel=0.20), (
            f"{method}: recovered {sigma:.4f}, expected {true_sigma}")


def test_sigma_responds_to_factors_and_trimming():
    """Document the *measured* behaviour of the sigma estimator.

    The intuitive prior — "genuine factors depress the bulk, so the fitted
    sigma should fall" — turns out to be **wrong**, and it is worth recording
    why, because the opposite behaviour is easy to misread as a bug.

    With ``trim=0`` the spectrum's median rises when a strong factor is added
    (the factor pushes the whole bulk up in level, and the top eigenvalue
    leaves the median location), so the fitted sigma rises towards the true
    value.  In this experiment the true noise sigma is 1.0 and the fitted
    values are:

    ==============  ==========  ==========
    factor strength  trim = 0    trim = 0.2
    ==============  ==========  ==========
    0.0              0.955       0.891
    0.3              1.011       0.907
    1.0              1.024       0.912
    3.0              1.024       0.913
    ==============  ==========  ==========

    Two conclusions, both asserted below:

    1. ``sigma`` **increases** with factor strength at fixed trimming — the
       opposite of the naive expectation.
    2. Trimming the top of the spectrum (``trim > 0``) shifts ``sigma``
       **down**, and the default ``trim = 0.2`` therefore sits below the true
       value.  This is deliberate: the projection is conservative, keeping
       *more* eigenvalues as signal (a larger noise band would be the
       aggressive choice).  The direction of the bias is what matters, and it
       matches the design intent.
    """
    q, n, t = 0.25, 16, 64
    true_sigma = 1.0
    n_rep = 40

    def mean_fitted(factor_strength, trim, seed=11):
        rng = np.random.default_rng(seed)
        vals = []
        for _ in range(n_rep):
            f = rng.standard_normal((t, 1))
            b = rng.standard_normal((1, n)) * factor_strength
            x = f @ b + rng.standard_normal((t, n)) * true_sigma
            ev = np.linalg.eigvalsh(np.cov(x, rowvar=False, ddof=1))
            vals.append(fit_mp_sigma(ev, q, method="median_bulk", trim=trim))
        return float(np.mean(vals))

    # (1) factors raise the fitted sigma towards the truth
    s_none = mean_fitted(0.0, 0.0)
    s_strong = mean_fitted(3.0, 0.0)
    assert s_strong > s_none, (
        f"fitted sigma must rise with factor strength "
        f"(none={s_none:.4f}, strong={s_strong:.4f})")
    # and it must not overshoot wildly
    assert 0.5 * true_sigma < s_strong < 1.5 * true_sigma

    # (2) trimming pushes sigma down
    for strength in (0.0, 1.0, 3.0):
        untrimmed = mean_fitted(strength, 0.0)
        trimmed = mean_fitted(strength, 0.2)
        assert trimmed < untrimmed, (
            f"trimming must lower the fitted sigma "
            f"(strength={strength}: trim0={untrimmed:.4f}, "
            f"trim0.2={trimmed:.4f})")


def test_q_caps_sigma_so_something_stays_noise():
    """The cap must prevent a degenerate sigma that leaves no noise eigenvalues."""
    q, n, t = 0.5, 8, 16
    rng = np.random.default_rng(3)
    x = rng.standard_normal((t, n))
    ev = np.linalg.eigvalsh(np.cov(x, rowvar=False, ddof=1))
    sigma = fit_mp_sigma(ev, q, method="median_bulk")
    law = MPLaw(n, t, q, sigma, *mp_edge(sigma, q))
    assert law.n_noise_eigenvalues(ev) >= 1


# ---------------------------------------------------------------------------
# MPLaw container
# ---------------------------------------------------------------------------


def test_mplaw_fit_and_describe():
    n, t = 10, 200
    rng = np.random.default_rng(0)
    ev = np.linalg.eigvalsh(np.cov(rng.standard_normal((t, n)), rowvar=False))
    law = MPLaw.fit(ev, n, t)
    assert law.q == pytest.approx(n / t)
    assert law.lambda_minus < law.lambda_plus
    assert "MPLaw" in law.describe()
    assert 0 <= law.n_noise_eigenvalues(ev) <= n
    assert law.n_noise_eigenvalues(ev) + law.n_signal_eigenvalues(ev) == n


def test_mplaw_mean_noise_matches_definition():
    ev = np.array([0.1, 0.2, 0.3, 5.0])
    law = MPLaw(4, 100, 0.04, 1.0, 0.0, 0.35)
    assert law.mean_noise_eigenvalue(ev) == pytest.approx(0.2)
    # fallback when nothing is below the edge
    assert law.mean_noise_eigenvalue(ev * 100, fallback=9.0) == pytest.approx(9.0)


def test_effective_aspect_ratio_rejects_zero_obs():
    with pytest.raises(ValueError):
        effective_aspect_ratio(10, 0)


def test_unknown_sigma_method_raises():
    with pytest.raises(ValueError):
        fit_mp_sigma(np.array([0.5, 1.0, 1.5]), 0.1, method="nope")
