"""Portfolio weight allocators.

Four allocators are implemented, each taking a **covariance matrix** (and
optionally a correlation matrix) and returning long-only, fully-invested
weights:

==========================  ==================================================
``minimum_variance``        :math:`\\min_w w^T\\Sigma w` s.t.
                            :math:`\\sum w = 1`, :math:`w \\ge 0`.
``risk_parity``             Equal Risk Contribution (ERC): every asset
                            contributes the same share of total portfolio risk.
                            Solved by cyclical coordinate descent on the
                            log-barrier formulation of Spinu (2013), which is
                            convex and needs no general-purpose solver.
``hrp``                     Hierarchical Risk Parity (Lopez de Prado 2016):
                            quasi-diagonalise, recursively bisect, allocate by
                            inverse cluster variance.
``max_diversification``     Maximise the diversification ratio
                            :math:`(\\sum_i w_i \\sigma_i)/\\sqrt{w^T \\Sigma w}`
                            (Choueifaty & Coignard 2008).
``equal_weight``            :math:`1/N` — the hardest benchmark to beat.
``inverse_variance``        :math:`w_i \\propto 1/\\sigma_i^2` — the naive
                            "risk parity" that ignores correlations.
==========================  ==================================================

Design notes
------------
* **Minimum variance** is solved with ``cvxpy`` using the ``CLARABEL``
  interior-point solver, which handles the PSD quadratic objective and the
  simplex constraint robustly.  A closed-form inverse-variance fallback is
  used if the solver fails (it should not for a PSD input).
* **Risk parity** uses the coordinate-descent algorithm of Griveau-Billion,
  Richard & Roncalli (2013) on the dual problem
  :math:`\\min_y \\tfrac12 y^T \\Sigma y - \\sum_i \\log y_i`, whose solution
  maps to the ERC weights by normalisation.  This is both faster and more
  accurate than a generic nonlinear solver, and its convergence is
  monotone.
* **HRP** allocates down the dendrogram with the inverse-variance rule at
  each split, using the *cluster* variance (not the asset variance) so that
  intra-cluster correlations are respected.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from rmt_portfolio.portfolio.clustering import (
    ClusterResult,
    correlation_distance,
    hierarchical_cluster,
)

__all__ = [
    "PortfolioResult",
    "minimum_variance",
    "risk_parity",
    "hrp",
    "max_diversification",
    "equal_weight",
    "inverse_variance",
    "ALLOCATORS",
    "allocate",
    "risk_contributions",
    "effective_number_of_bets",
    "diversification_ratio",
    "portfolio_volatility",
]


# ---------------------------------------------------------------------------
# Result container and diagnostics
# ---------------------------------------------------------------------------


@dataclass
class PortfolioResult:
    """Weights plus the diagnostics needed for reporting.

    Attributes
    ----------
    weights:
        Long-only weights summing to one.
    method:
        Allocator name.
    volatility:
        Ex-ante annualised volatility under the input covariance, *per unit of
        the covariance's time scale* (callers annualise by supplying an
        annualised covariance or by scaling afterwards).
    risk_contributions:
        Per-asset fractional risk contribution, summing to one.
    enb:
        Effective Number of Bets, :math:`1/\\sum_i \\mathrm{RC}_i^2`.
    diversification_ratio:
        :math:`(\\sum_i w_i \\sigma_i)/\\sigma_p \\ge 1`.
    n_clusters:
        Number of clusters (HRP only).
    info:
        Free-form extra diagnostics.
    """

    weights: np.ndarray
    method: str
    volatility: float = np.nan
    risk_contributions: Optional[np.ndarray] = None
    enb: float = np.nan
    diversification_ratio: float = np.nan
    n_clusters: Optional[int] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).ravel()
        # strip numerical dust and renormalise
        w = np.where(self.weights < 1e-12, 0.0, self.weights)
        s = w.sum()
        self.weights = w / s if s > 0 else np.full_like(w, 1.0 / w.size)

    @property
    def n_assets(self) -> int:
        return self.weights.size

    @property
    def herfindahl(self) -> float:
        """Herfindahl concentration index, :math:`\\sum_i w_i^2`."""
        return float(np.sum(self.weights ** 2))

    def top_holdings(self, k: int = 5) -> List[Tuple[int, float]]:
        idx = np.argsort(self.weights)[::-1][:k]
        return [(int(i), float(self.weights[i])) for i in idx]

    def describe(self) -> str:
        return (
            f"PortfolioResult(method={self.method!r}, N={self.n_assets}, "
            f"vol={self.volatility:.6f}, ENB={self.enb:.2f}, "
            f"DR={self.diversification_ratio:.3f})"
        )


# ---------------------------------------------------------------------------
# Risk diagnostics
# ---------------------------------------------------------------------------


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    w = np.asarray(weights, float).ravel()
    return float(np.sqrt(max(w @ np.asarray(cov, float) @ w, 0.0)))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fractional risk contributions ``RC_i = w_i (Σw)_i / (wᵀΣw)``.

    These are *fractions*: they sum to one.  Under an ERC portfolio they all
    equal ``1/N``.
    """
    w = np.asarray(weights, float).ravel()
    c = np.asarray(cov, float)
    var = float(w @ c @ w)
    if var <= 0:
        return np.full(w.size, 1.0 / w.size)
    return w * (c @ w) / var


def effective_number_of_bets(weights: np.ndarray, cov: np.ndarray) -> float:
    """Effective Number of Bets: :math:`1/\\sum_i \\mathrm{RC}_i^2`."""
    rc = risk_contributions(weights, cov)
    s = float(np.sum(rc ** 2))
    return 1.0 / s if s > 0 else float(weights.size)


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    """Choueifaty-Coignard diversification ratio.

    :math:`\\mathrm{DR} = \\frac{\\sum_i w_i \\sigma_i}{\\sqrt{w^T\\Sigma w}}`,
    with :math:`\\sigma_i = \\sqrt{\\Sigma_{ii}}`.  ``DR >= 1`` always, with
    equality only for a single-asset portfolio.
    """
    w = np.asarray(weights, float).ravel()
    c = np.asarray(cov, float)
    sd = np.sqrt(np.maximum(np.diag(c), 0.0))
    num = float(np.sum(w * sd))
    den = portfolio_volatility(w, c)
    if den <= 0:
        return 1.0
    return num / den


def _attach_diagnostics(res: PortfolioResult, cov: np.ndarray) -> PortfolioResult:
    """Fill volatility, risk contributions, ENB and DR in place."""
    cov = np.asarray(cov, float)
    res.volatility = portfolio_volatility(res.weights, cov)
    res.risk_contributions = risk_contributions(res.weights, cov)
    res.enb = effective_number_of_bets(res.weights, cov)
    res.diversification_ratio = diversification_ratio(res.weights, cov)
    return res


# ---------------------------------------------------------------------------
# Baseline allocators
# ---------------------------------------------------------------------------


def equal_weight(n_assets: int, cov: Optional[np.ndarray] = None,
                 **_: Any) -> PortfolioResult:
    """The ``1/N`` portfolio — DeMiguel, Garlappi & Uppal (2009)."""
    w = np.full(int(n_assets), 1.0 / int(n_assets))
    res = PortfolioResult(weights=w, method="equal_weight")
    return _attach_diagnostics(res, cov) if cov is not None else res


def inverse_variance(cov: np.ndarray, **_: Any) -> PortfolioResult:
    """Inverse-variance weights, :math:`w_i \\propto 1/\\sigma_i^2`.

    Optimal when correlations are zero (or equal and constant, provided the
    covariance is used rather than the correlation matrix).
    """
    c = np.asarray(cov, float)
    iv = 1.0 / np.maximum(np.diag(c), 1e-300)
    w = iv / iv.sum()
    res = PortfolioResult(weights=w, method="inverse_variance")
    return _attach_diagnostics(res, c)


# ---------------------------------------------------------------------------
# Minimum variance
# ---------------------------------------------------------------------------


def minimum_variance(
    cov: np.ndarray,
    long_only: bool = True,
    max_weight: Optional[float] = None,
    solver: Optional[str] = None,
    **_: Any,
) -> PortfolioResult:
    """Global minimum-variance portfolio.

    Solves

    .. math::

        \\min_{w} \\; w^T \\Sigma w
        \\quad \\text{s.t.} \\quad \\mathbf{1}^T w = 1, \\; w \\ge 0
        \\;(\\text{optional } w_i \\le w_{\\max}).

    Uses ``cvxpy`` when available (robust to near-singular ``Σ``), and falls
    back to the analytic solution for the unconstrained long-short case.

    Parameters
    ----------
    long_only:
        Enforce :math:`w \\ge 0`.  ``False`` allows shorting (weights may be
        negative; the budget constraint still holds).
    max_weight:
        Optional per-asset cap, useful for diversification-constrained MVO.
    """
    c = np.asarray(cov, float)
    n = c.shape[0]
    c = 0.5 * (c + c.T)
    # CVXPY requires PSD; symmetrise and add a whisper of ridge for solvers.
    c_psd = c + 1e-10 * np.eye(n)

    try:
        import cvxpy as cp

        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(c_psd)))
        constraints = [cp.sum(w) == 1]
        if long_only:
            constraints.append(w >= 0)
        if max_weight is not None:
            constraints.append(w <= float(max_weight))
        problem = cp.Problem(objective, constraints)
        solved = False
        for slv in ([solver] if solver else ["CLARABEL", "OSQP", "SCS"]):
            try:
                problem.solve(solver=slv, verbose=False)
                if w.value is not None and np.all(np.isfinite(w.value)):
                    solved = True
                    break
            except Exception:      # pragma: no cover - solver availability
                continue
        if solved:
            weights = np.maximum(np.asarray(w.value, float).ravel(), 0.0) \
                if long_only else np.asarray(w.value, float).ravel()
            res = PortfolioResult(weights=weights, method="minimum_variance",
                                  info={"solver": slv,
                                        "objective": float(problem.value)})
            return _attach_diagnostics(res, c)
    except ImportError:            # pragma: no cover
        pass

    # ---- fallback: analytic / projected solution ------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            inv = np.linalg.solve(c_psd, np.ones(n))
            w = inv / inv.sum()
        except np.linalg.LinAlgError:      # pragma: no cover
            w = np.full(n, 1.0 / n)
    if long_only:
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.full(n, 1.0 / n)
        else:
            w = w / w.sum()
    res = PortfolioResult(weights=w, method="minimum_variance",
                          info={"solver": "analytic_fallback"})
    return _attach_diagnostics(res, c)


# ---------------------------------------------------------------------------
# Risk parity (equal risk contribution)
# ---------------------------------------------------------------------------


def risk_parity(
    cov: np.ndarray,
    budget: Optional[np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
    **_: Any,
) -> PortfolioResult:
    """Equal Risk Contribution portfolio via cyclical coordinate descent.

    The ERC portfolio satisfies :math:`w_i (\\Sigma w)_i = w_j (\\Sigma w)_j`
    for all ``i, j``.  Following Griveau-Billion, Richard & Roncalli (2013) we
    solve the *dual*

    .. math::

        \\min_{y > 0} \\; \\tfrac{1}{2} y^T \\Sigma y
            - \\sum_i b_i \\log y_i

    whose stationary point, after normalisation, gives the risk-budget
    weights.  Coordinate descent converges monotonically; each coordinate
    update solves a scalar quadratic in closed form:

    .. math::

        y_i \\leftarrow \\frac{-c_i + \\sqrt{c_i^2 + 4 \\sigma_{ii} b_i}}
                               {2 \\sigma_{ii}},
        \\qquad c_i = \\sum_{j \\ne i} \\sigma_{ij} y_j

    Parameters
    ----------
    budget:
        Target risk budget :math:`b_i > 0`, normalised internally.  Defaults
        to equal budgets :math:`1/N` (classical ERC).  Supplying unequal
        budgets gives a *risk-budgeting* portfolio.
    """
    c = np.asarray(cov, float)
    n = c.shape[0]
    c = 0.5 * (c + c.T)
    diag = np.diag(c).copy()

    # nudge any non-positive diagonal (can happen after aggressive denoising)
    if np.any(diag <= 0):
        diag = np.maximum(diag, 1e-12)
        c = c - np.diag(np.diag(c)) + np.diag(diag)

    if budget is None:
        b = np.full(n, 1.0 / n)
    else:
        b = np.asarray(budget, float).ravel()
        if b.size != n or np.any(b <= 0):
            raise ValueError("budget must be positive with one entry per asset")
        b = b / b.sum()

    # initialise at inverse-volatility (a good feasible starting point)
    y = 1.0 / np.maximum(np.sqrt(diag), 1e-12)
    y = y / y.sum()

    for _ in range(max_iter):
        y_old = y.copy()
        for i in range(n):
            # c_i = sum_{j != i} sigma_ij y_j
            c_i = float(np.dot(c[i], y) - c[i, i] * y[i])
            s_ii = float(c[i, i])
            if s_ii <= 0:
                s_ii = 1e-12
            disc = c_i ** 2 + 4.0 * s_ii * b[i]
            y[i] = (-c_i + np.sqrt(max(disc, 0.0))) / (2.0 * s_ii)
        if np.max(np.abs(y - y_old)) < tol * max(np.max(np.abs(y_old)), 1e-30):
            break

    w = y / y.sum()
    res = PortfolioResult(weights=w, method="risk_parity",
                          info={"n_iter": _,
                                "budget": b.tolist(),
                                "max_rc_deviation":
                                    float(np.max(np.abs(
                                        risk_contributions(w, c) - b)))})
    return _attach_diagnostics(res, c)


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------


def _cluster_variance(cov: np.ndarray, idx: Sequence[int],
                      weights: np.ndarray) -> float:
    """Variance of an inverse-variance-weighted sub-portfolio."""
    sub_cov = cov[np.ix_(idx, idx)]
    return float(weights @ sub_cov @ weights)


def _inverse_variance_weights(cov: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """Inverse-variance weights within a cluster.

    :math:`\\alpha_i = \\frac{1/\\tilde\\sigma_{ii}}{\\sum_j 1/\\tilde\\sigma_{jj}}`
    where :math:`\\tilde\\sigma_{ii}` is the *cluster-conditional* variance
    obtained from the covariance matrix restricted to the cluster.
    """
    sub = cov[np.ix_(idx, idx)]
    iv = 1.0 / np.maximum(np.diag(sub), 1e-300)
    return iv / iv.sum()


def _recursive_bisection(
    cov: np.ndarray,
    order: np.ndarray,
    cluster_result: ClusterResult,
    depth: int = 0,
) -> np.ndarray:
    """Recursively bisect the quasi-diagonalised order and allocate risk."""
    n = len(order)
    if n == 1:
        return np.array([1.0])

    # split the ordered list down the middle (the standard HRP bisection)
    mid = n // 2
    left_idx = order[:mid]
    right_idx = order[mid:]

    w_left = _inverse_variance_weights(cov, left_idx)
    w_right = _inverse_variance_weights(cov, right_idx)

    var_left = _cluster_variance(cov, left_idx, w_left)
    var_right = _cluster_variance(cov, right_idx, w_right)

    # inverse-variance allocation across the two clusters
    denom = var_left + var_right
    if denom <= 0:
        alpha = 0.5
    else:
        alpha = 1.0 - var_left / denom

    alloc = np.concatenate([
        alpha * _recursive_bisection(cov, left_idx, cluster_result, depth + 1),
        (1.0 - alpha) * _recursive_bisection(cov, right_idx, cluster_result,
                                             depth + 1),
    ])
    return alloc


def hrp(
    cov: np.ndarray,
    corr: Optional[np.ndarray] = None,
    n_clusters: Optional[int] = None,
    linkage_method: str = "single",
    **_: Any,
) -> PortfolioResult:
    """Hierarchical Risk Parity (Lopez de Prado 2016).

    Three steps:

    1. **Tree clustering** — agglomerative clustering on the correlation
       distance :math:`\\sqrt{(1-\\rho)/2}`.
    2. **Quasi-diagonalisation** — reorder assets so that similar assets are
       adjacent.
    3. **Recursive bisection** — split the ordered list in half, and split
       the portfolio risk between the two halves in inverse proportion to
       their (inverse-variance-weighted) cluster variances.  Recurse until
       each asset has a weight.

    The method never inverts ``Σ``, so it is immune to the ill-conditioning
    that plagues minimum-variance optimisation on denoised matrices.

    Parameters
    ----------
    corr:
        Correlation matrix used for *clustering only*.  Defaults to the
        correlation implied by ``cov``.  Passing a denoised correlation while
        passing the sample covariance here lets you study the two ingredients
        separately.
    """
    c = np.asarray(cov, float)
    n = c.shape[0]
    c = 0.5 * (c + c.T)

    if corr is None:
        sd = np.sqrt(np.maximum(np.diag(c), 1e-300))
        corr = c / np.outer(sd, sd)
        np.fill_diagonal(corr, 1.0)
    corr = np.asarray(corr, float)

    cl = hierarchical_cluster(corr, method=linkage_method,
                              n_clusters=n_clusters)
    alloc = _recursive_bisection(c, cl.order, cl)
    weights = np.empty(n)
    weights[cl.order] = alloc

    res = PortfolioResult(
        weights=weights, method="hrp", n_clusters=cl.n_clusters,
        info={"linkage": linkage_method,
              "cluster_order": cl.order.tolist(),
              "cluster_labels": cl.labels.tolist(),
              "cluster_sizes": sorted(
                  [len(v) for v in cl.cluster_members().values()],
                  reverse=True)},
    )
    return _attach_diagnostics(res, c)


# ---------------------------------------------------------------------------
# Maximum diversification
# ---------------------------------------------------------------------------


def max_diversification(
    cov: np.ndarray,
    long_only: bool = True,
    max_weight: Optional[float] = None,
    **_: Any,
) -> PortfolioResult:
    """Maximum diversification portfolio (Choueifaty & Coignard 2008).

    Maximises the diversification ratio

    .. math::

        \\max_w \\; \\frac{\\sum_i w_i \\sigma_i}{\\sqrt{w^T \\Sigma w}}
        \\quad \\text{s.t.} \\quad \\mathbf{1}^T w = 1, \\; w \\ge 0.

    The objective is scale-invariant, so the budget constraint is a pure
    normalisation.  We solve it with SLSQP on the unconstrained-direction
    reformulation, which is smooth and reliable at this dimension.
    """
    c = np.asarray(cov, float)
    n = c.shape[0]
    c = 0.5 * (c + c.T) + 1e-12 * np.eye(n)
    sd = np.sqrt(np.maximum(np.diag(c), 1e-300))

    def neg_dr(w: np.ndarray) -> float:
        num = float(np.dot(w, sd))
        var = float(w @ c @ w)
        if var <= 0 or num <= 0:
            return 1e6
        return -num / np.sqrt(var)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, float(max_weight) if max_weight else 1.0)
              for _ in range(n)] if long_only else [(None, None)] * n

    w0 = np.ones(n) / n
    best_w, best_f = None, np.inf
    for start in (w0, 1.0 / np.maximum(sd, 1e-12) / np.sum(1.0 / np.maximum(sd, 1e-12))):
        try:
            r = minimize(neg_dr, start, method="SLSQP", constraints=cons,
                         bounds=bounds,
                         options={"maxiter": 500, "ftol": 1e-12})
            if r.success and r.fun < best_f:
                best_f, best_w = float(r.fun), np.asarray(r.x, float).ravel()
        except Exception:          # pragma: no cover
            continue

    if best_w is None:             # pragma: no cover - fallback
        best_w = 1.0 / np.maximum(sd, 1e-12)
        best_w = best_w / best_w.sum()
        if long_only:
            best_w = np.maximum(best_w, 0.0)
            best_w = best_w / best_w.sum()

    if long_only:
        best_w = np.maximum(best_w, 0.0)
    res = PortfolioResult(weights=best_w, method="max_diversification",
                          info={"neg_dr": best_f})
    return _attach_diagnostics(res, c)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

#: Registry of allocators.  Every entry is callable as
#: ``fn(cov=..., corr=..., **kwargs) -> PortfolioResult``.
ALLOCATORS: Dict[str, Callable[..., PortfolioResult]] = {
    "equal_weight": lambda cov=None, corr=None, n_assets=None, **kw:
        equal_weight(n_assets if n_assets is not None else cov.shape[0],
                     cov=cov),
    "inverse_variance": lambda cov=None, corr=None, **kw:
        inverse_variance(cov),
    "minimum_variance": lambda cov=None, corr=None, **kw:
        minimum_variance(cov, **kw),
    "risk_parity": lambda cov=None, corr=None, **kw:
        risk_parity(cov, **kw),
    "hrp": lambda cov=None, corr=None, **kw:
        hrp(cov, corr=corr, **kw),
    "max_diversification": lambda cov=None, corr=None, **kw:
        max_diversification(cov, **kw),
}

#: Human-readable labels for tables and the paper.
ALLOCATOR_LABELS: Dict[str, str] = {
    "equal_weight": "Equal weight (1/N)",
    "inverse_variance": "Inverse variance",
    "minimum_variance": "Minimum variance",
    "risk_parity": "Risk parity (ERC)",
    "hrp": "Hierarchical Risk Parity",
    "max_diversification": "Maximum diversification",
}


def allocate(
    cov: np.ndarray,
    method: str = "hrp",
    corr: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> PortfolioResult:
    """Allocate weights by name.

    Parameters
    ----------
    cov:
        Covariance matrix used for the risk model.
    method:
        Key of :data:`ALLOCATORS`.
    corr:
        Optional separate correlation matrix (used by HRP for clustering).
    """
    if method not in ALLOCATORS:
        raise ValueError(
            f"unknown allocator {method!r}; available: {sorted(ALLOCATORS)}")
    return ALLOCATORS[method](cov=np.asarray(cov, float), corr=corr, **kwargs)
