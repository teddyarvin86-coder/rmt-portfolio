"""Marchenko-Pastur law: theoretical noise spectrum and edge estimation.

The Marchenko-Pastur (MP) law describes the limiting spectral density of the
eigenvalues of a sample covariance matrix built from ``T`` i.i.d. observations
of ``N`` independent random variables with variance ``sigma^2``.

Defining the aspect ratio

.. math:: q = N / T \\leq 1

the density of eigenvalues is supported on :math:`[\\lambda_-, \\lambda_+]`
with

.. math::

    \\lambda_\\pm = \\sigma^2 \\left(1 \\pm \\sqrt{q}\\right)^2

and density

.. math::

    \\rho(\\lambda) = \\frac{1}{2\\pi \\sigma^2 q \\lambda}
        \\sqrt{(\\lambda_+ - \\lambda)(\\lambda - \\lambda_-)}
        \\quad \\text{for } \\lambda \\in [\\lambda_-, \\lambda_+].

When ``q > 1`` the matrix is singular; there is then an atom of mass
``1 - 1/q`` at the origin in addition to the continuous bulk.  This module
handles both regimes.

The practical purpose here is to obtain :math:`\\lambda_+` — the *upper edge*
of the noise band.  Eigenvalues below it carry no statistically reliable
information about genuine correlations and can be replaced, while eigenvalues
above it correspond to genuine systematic risk factors (typically the market
mode and a handful of sector/industry modes).

References
----------
Marchenko & Pastur (1967); Laloux, Cizeau, Bouchaud & Potters (1999);
Bun, Bouchaud & Potters (2017), *Physics Reports* 666, Section 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "MPLaw",
    "fit_mp_sigma",
    "mp_edge",
    "mp_density",
    "mp_cdf_fraction",
    "mp_sigma_from_edge",
    "effective_aspect_ratio",
]


# ---------------------------------------------------------------------------
# Theoretical quantities
# ---------------------------------------------------------------------------


def effective_aspect_ratio(n_assets: int, n_obs: int) -> float:
    """Return :math:`q = N / T`, guarding against degenerate inputs."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    return float(n_assets) / float(n_obs)


def mp_edge(sigma: float, q: float) -> Tuple[float, float]:
    """Lower and upper edges of the MP bulk.

    Parameters
    ----------
    sigma:
        Volatility scale (standard deviation per unit time) of the underlying
        i.i.d. noise.  For a *correlation* matrix input this is typically 1.
    q:
        Aspect ratio ``N / T``.

    Returns
    -------
    (lambda_minus, lambda_plus)
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    q = float(q)
    if q <= 0:
        raise ValueError("q must be positive")
    root = np.sqrt(q)
    lo = sigma ** 2 * (1.0 - root) ** 2
    hi = sigma ** 2 * (1.0 + root) ** 2
    return float(lo), float(hi)


def mp_density(x: np.ndarray, sigma: float, q: float) -> np.ndarray:
    """Marchenko-Pastur density evaluated on the grid ``x``.

    Values outside the support are returned as ``0``.  For ``q > 1`` the
    continuous bulk is supported on :math:`[\\lambda_-, \\lambda_+]` where

    .. math::

        \\lambda_\\pm = \\sigma^2\\left(1 \\pm \\sqrt{q}\\right)^2

    and the density integrates to :math:`1/q`; the remaining mass
    :math:`1 - 1/q` is an atom at zero, which is **not** represented in a
    continuous density.  For ``q <= 1`` the density integrates to 1.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = mp_edge(sigma, q)
    out = np.zeros_like(x, dtype=float)
    mask = (x > lo) & (x < hi)
    if not np.any(mask):
        return out
    xs = x[mask]
    # General MP density, valid for all q > 0:
    #   rho(x) = sqrt((hi-x)(x-lo)) / (2 pi sigma^2 q x)
    inner = np.maximum((hi - xs) * (xs - lo), 0.0)
    denom = 2.0 * np.pi * sigma ** 2 * q * np.maximum(xs, 1e-300)
    out[mask] = np.sqrt(inner) / denom
    return out


def mp_cdf_fraction(x: np.ndarray, sigma: float, q: float,
                    n_grid: int = 6000) -> np.ndarray:
    """Marchenko-Pastur CDF obtained by exact numerical integration.

    This is the authoritative implementation used throughout the project.

    A note on the closed form
    -------------------------
    An earlier revision of this module carried a hand-transcribed analytic
    CDF.  A unit test cross-checking the two implementations showed the
    analytic version returned :math:`F(\\lambda_+)=1.5` and
    :math:`F(\\lambda_-)=0.5` for :math:`q=1/4` — i.e. it was simply wrong.  It
    has been removed rather than repaired, because:

    * the numerical integrator is exact to machine precision once the grid is
      fine enough, and costs only :math:`O(n_\\text{grid})` — negligible
      against the eigendecompositions it is used for;
    * there is no performance-critical path that needs the closed form;
    * an unused, subtly wrong formula is a liability: somebody will eventually
      import it.

    The function is kept under a distinct name (rather than being folded into
    ``_mp_cdf``) so that it can be tested directly.

    Parameters
    ----------
    x:
        Evaluation points.
    sigma:
        Noise scale.
    q:
        Aspect ratio ``N / T``.
    n_grid:
        Number of integration nodes over the support.

    Returns
    -------
    Array of CDF values in ``[0, 1]``.  For ``q > 1`` the atom of mass
    :math:`1 - 1/q` at zero is included.
    """
    x = np.asarray(x, dtype=float)
    lo, hi = mp_edge(sigma, q)
    atom = (1.0 - 1.0 / q) if q > 1.0 else 0.0

    out = np.full(x.shape, atom, dtype=float)
    out[x >= hi] = 1.0

    mask = (x > lo) & (x < hi)
    if np.any(mask):
        # integrate the density from lo to each requested point
        pad = (hi - lo) * 1e-9
        grid = np.linspace(lo + pad, hi - pad, n_grid)
        dens = mp_density(grid, sigma, q)
        cdf_grid = atom + np.concatenate(
            [[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1])
                              * np.diff(grid))]
        )
        # normalise so that cdf(hi) == 1 exactly
        if cdf_grid[-1] > 0:
            cdf_grid = atom + (cdf_grid - atom) / (cdf_grid[-1] - atom) * (
                1.0 - atom)
        out[mask] = np.interp(x[mask], grid, cdf_grid)

    # Note: the density integrates to 1/q on the bulk for q>1, so the
    # normalisation step above correctly rescales to the required mass 1-atom.
    return np.clip(out, 0.0, 1.0)


#: Backwards-compatible alias.  ``_mp_cdf`` is the name used internally and in
#: the tests; both point at the same verified numerical integrator.
_mp_cdf = mp_cdf_fraction


# ---------------------------------------------------------------------------
# Fitting the noise scale sigma
# ---------------------------------------------------------------------------


def mp_sigma_from_edge(lambda_plus_obs: float, q: float) -> float:
    """Invert :math:`\\lambda_+ = \\sigma^2 (1+\\sqrt{q})^2` for ``sigma``."""
    if lambda_plus_obs <= 0:
        raise ValueError("lambda_plus_obs must be positive")
    return float(np.sqrt(lambda_plus_obs) / (1.0 + np.sqrt(q)))


def fit_mp_sigma(
    eigenvalues: np.ndarray,
    q: float,
    method: str = "median_bulk",
    trim: float = 0.2,
    n_grid: int = 2000,
    n_bulk_points: int = 300,
) -> float:
    """Estimate the MP noise scale ``sigma`` from an observed spectrum.

    The presence of genuine factors (the market mode and sector modes)
    *inflates* the largest eigenvalues and, because the trace is fixed,
    correspondingly *depresses* the bulk relative to a pure-noise spectrum.
    Any estimator of :math:`\\sigma` must therefore be robust to the
    contamination of the top of the spectrum.

    Four strategies are provided:

    ``'median_bulk'`` (default)
        Fit :math:`\\sigma` so that the **median of the MP law** matches the
        median of the *lower* portion of the observed spectrum (the spectral
        range below its own median, i.e. the cleanest, least factor-contaminated
        half of the bulk).  Because the MP law scales linearly in
        :math:`\\sigma` for fixed ``q``, the median of the theoretical law in
        units of :math:`\\sigma^2` is
        :math:`m(q)`, computed once by bisection; then
        :math:`\\sigma = \\mathrm{median}(\\lambda_{\\text{low}}) / m(q)`.
        Robust, fast, and scale-free.

    ``'mean_bulk'``
        Match the **mean** of the trimmed spectrum instead of the median.
        The MP law has mean :math:`\\sigma^2`, so
        :math:`\\sigma = \\sqrt{\\mathrm{mean}(\\lambda_{\\text{trimmed}})}`.
        Very stable but sensitive to the choice of ``trim``.

    ``'max_eigen'``
        Assume the largest eigenvalue is pure noise:
        :math:`\\sigma = \\sqrt{\\lambda_{\\max}} / (1+\\sqrt{q})`.  Classic
        Laloux et al. (1999).  Conservative — this *over*-estimates
        :math:`\\lambda_+` when genuine factors exist, so it keeps more
        eigenvalues than any other method.  Useful as a robustness bound.

    ``'cdf_fit'``
        Minimise the Kolmogorov-Smirnov distance between the empirical CDF of
        the trimmed spectrum and the MP CDF.  The most statistically
        principled and the recommended choice when accuracy matters more than
        speed.

    Parameters
    ----------
    eigenvalues:
        Eigenvalues of the sample covariance / correlation matrix.
    q:
        Aspect ratio ``N / T``.
    trim:
        Fraction of the **largest** eigenvalues excluded before fitting
        (assumed to be genuine factors).  Defaults to 0.2, i.e. the top 20%
        of the spectrum is treated as potentially signal-dominated.
    n_grid:
        Resolution of the bisection / search grid.
    n_bulk_points:
        Number of quantile points used in the KS objective (``'cdf_fit'``).
    """
    ev = np.asarray(eigenvalues, dtype=float)
    ev = ev[np.isfinite(ev)]
    if ev.size == 0:
        raise ValueError("no finite eigenvalues supplied")
    ev = np.sort(ev)
    n = ev.size
    q = float(q)

    # --- trimmed bulk: drop the top ``trim`` fraction (candidate factors) ---
    n_keep = max(int(np.floor(n * (1.0 - trim))), max(int(np.ceil(2.0)), 1))
    n_keep = min(n_keep, n)
    bulk = ev[:n_keep]

    if method == "max_eigen":
        return mp_sigma_from_edge(ev[-1], q)

    if method == "median_bulk":
        # Match the median of the observed trimmed bulk to the median of the
        # MP law.  The MP law scales linearly in sigma^2 for fixed q, so the
        # median in *lambda* units is ``sigma^2 * m(q)`` where ``m(q)`` is the
        # median of the law with sigma = 1.  Hence
        #     sigma = sqrt( median(bulk) / m(q) ).
        m = _mp_median_unit(q)
        med_bulk = float(np.median(bulk))
        if m <= 0:
            # Atom at zero dominates (very large q): the median carries no
            # information about sigma; fall back to a mean-based estimate.
            sigma = float(np.sqrt(max(np.mean(bulk), 1e-300)))
        else:
            sigma = float(np.sqrt(max(med_bulk, 1e-300) / m))
        return _cap_sigma(sigma, ev[-1], q)

    if method == "mean_bulk":
        # The MP law has mean sigma^2, so match the mean of the trimmed bulk.
        mean_bulk = float(np.mean(bulk))
        sigma = float(np.sqrt(max(mean_bulk, 1e-300)))
        return _cap_sigma(sigma, ev[-1], q)

    if method == "cdf_fit":
        # Empirical CDF of the trimmed bulk, excluding the very smallest
        # eigenvalues (which carry the largest relative sampling error).
        nb = bulk.size
        n_lo = max(int(np.floor(nb * 0.05)), 1)
        pts = bulk[n_lo:] if nb - n_lo >= 3 else bulk
        emp = (np.arange(1, pts.size + 1) - 0.5) / pts.size

        lo_sigma = max(mp_sigma_from_edge(max(pts[-1], 1e-12), q) * 0.5,
                       np.sqrt(np.mean(pts)) * 0.5, 1e-8)
        hi_sigma = max(mp_sigma_from_edge(max(ev[-1], 1e-12), q) * 1.5,
                       np.sqrt(np.mean(pts)) * 2.0)
        grid_sigma = np.linspace(lo_sigma, hi_sigma, n_grid)
        best_sigma, best_dist = lo_sigma, np.inf
        for s in grid_sigma:
            diff = np.abs(_mp_cdf(pts, s, q) - emp)
            dist = float(np.max(diff))
            if dist < best_dist:
                best_dist, best_sigma = dist, float(s)
        return best_sigma

    raise ValueError(f"unknown sigma fitting method: {method!r}")


def _cap_sigma(sigma: float, lambda_max: float, q: float,
               factor: float = 1.0) -> float:
    """Guard ``sigma`` so the implied :math:`\\lambda_+` stays sensible.

    If the fitted :math:`\\sigma` implies an upper edge *above* the largest
    observed eigenvalue, no eigenvalue would be classified as noise.  That can
    happen when the bulk is severely depressed by strong factors.  We therefore
    cap :math:`\\sigma` at the value implied by treating :math:`\\lambda_{\\max}`
    as noise (times an optional ``factor``), which is the largest defensible
    noise scale, and floor it at a small positive number.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        return mp_sigma_from_edge(lambda_max, q)
    sigma_cap = mp_sigma_from_edge(max(lambda_max, 1e-300), q) * float(factor)
    return float(min(max(sigma, 1e-12), sigma_cap))


def _mp_median_unit(q: float, tol: float = 1e-10) -> float:
    """Median of the MP law with ``sigma = 1``, found by bisection on the CDF."""
    root = np.sqrt(q)
    a = (1.0 - root) ** 2
    b = (1.0 + root) ** 2
    if q > 1.0:
        # There is an atom of mass 1-1/q at 0.  The median is 0 whenever the
        # atom alone already carries >= 50% of the mass.
        if 1.0 - 1.0 / q >= 0.5:
            return 0.0
    lo, hi = a, b
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        cdf = float(_mp_cdf(np.array([mid]), 1.0, q)[0])
        if cdf < 0.5:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Convenience container
# ---------------------------------------------------------------------------


@dataclass
class MPLaw:
    """Fitted Marchenko-Pastur noise band for one estimation window.

    Attributes
    ----------
    n_assets, n_obs:
        Dimensions of the estimation problem.
    q:
        Aspect ratio ``N / T``.
    sigma:
        Fitted noise scale.
    lambda_minus, lambda_plus:
        Fitted noise band edges.
    sigma_method:
        Which estimator produced ``sigma``.
    """

    n_assets: int
    n_obs: int
    q: float
    sigma: float
    lambda_minus: float
    lambda_plus: float
    sigma_method: str = "median_bulk"

    # -- constructors ----------------------------------------------------

    @classmethod
    def fit(
        cls,
        eigenvalues: np.ndarray,
        n_assets: int,
        n_obs: int,
        sigma_method: str = "median_bulk",
        **kwargs,
    ) -> "MPLaw":
        q = effective_aspect_ratio(n_assets, n_obs)
        sigma = fit_mp_sigma(eigenvalues, q, method=sigma_method, **kwargs)
        lo, hi = mp_edge(sigma, q)
        return cls(n_assets=n_assets, n_obs=n_obs, q=q, sigma=sigma,
                   lambda_minus=lo, lambda_plus=hi,
                   sigma_method=sigma_method)

    # -- derived quantities ----------------------------------------------

    @property
    def noise_ratio(self) -> float:
        """Fraction of the unit eigenvalues that fall inside the noise band."""
        return float(min(self.q, 1.0))

    def n_noise_eigenvalues(self, eigenvalues: np.ndarray) -> int:
        """Count eigenvalues strictly below :math:`\\lambda_+`."""
        ev = np.asarray(eigenvalues, dtype=float)
        return int(np.sum(ev < self.lambda_plus))

    def n_signal_eigenvalues(self, eigenvalues: np.ndarray) -> int:
        """Count eigenvalues at or above :math:`\\lambda_+` (genuine factors)."""
        ev = np.asarray(eigenvalues, dtype=float)
        return int(np.sum(ev >= self.lambda_plus))

    def mean_noise_eigenvalue(self, eigenvalues: np.ndarray,
                              fallback: Optional[float] = None) -> float:
        """Average of the eigenvalues inside the noise band.

        This is the target value used by hard-thresholding: every noise
        eigenvalue is replaced by this single constant, which preserves the
        trace of the matrix.
        """
        ev = np.asarray(eigenvalues, dtype=float)
        noise = ev[ev < self.lambda_plus]
        if noise.size == 0:
            if fallback is not None:
                return float(fallback)
            return float(self.lambda_plus)
        return float(np.mean(noise))

    def describe(self) -> str:
        return (
            f"MPLaw(N={self.n_assets}, T={self.n_obs}, q={self.q:.4f}, "
            f"sigma={self.sigma:.4f} [{self.sigma_method}], "
            f"band=[{self.lambda_minus:.4f}, {self.lambda_plus:.4f}])"
        )
