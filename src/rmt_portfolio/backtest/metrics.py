"""Performance and risk metrics for return series.

All metrics are computed from a **daily** return series (or any periodic
series) and annualised with an explicit ``periods_per_year`` factor, so the
same code works for daily, weekly and monthly rebalancing without silently
changing the meaning of "annualised".

Conventions
-----------
* **Annualised return** is the geometric (compound) rate:
  :math:`(1 + R_{\\text{cum}})^{p/T} - 1`, not the arithmetic mean times ``p``.
  Geometric annualisation is the correct comparison against a buy-and-hold
  benchmark and is what practitioners quote.
* **Annualised volatility** is :math:`\\sigma_{\\text{period}} \\sqrt{p}`.
* **Sharpe ratio** uses the excess return over a supplied risk-free rate,
  annualised consistently with the return.
* **Sortino ratio** replaces the denominator with the *downside deviation*,
  computed against the target (default 0) rather than the mean.
* **Maximum drawdown** is the largest peak-to-trough decline of the
  compounded wealth path, reported as a positive number.
* **Calmar ratio** is annualised return divided by maximum drawdown.
* **Turnover** is one-way: :math:`\\tfrac{1}{2}\\sum_i |w_i^{\\text{new}} -
  w_i^{\\text{old,drifted}}|`, so a full liquidation-and-repurchase counts as
  1, matching the transaction-cost convention used in the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "PerformanceMetrics",
    "annualised_return",
    "annualised_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_series",
    "calmar_ratio",
    "value_at_risk",
    "conditional_value_at_risk",
    "skewness",
    "kurtosis",
    "ulcer_index",
    "turnover",
    "compute_metrics",
    "METRIC_LABELS",
]


# ---------------------------------------------------------------------------
# Return / risk primitives
# ---------------------------------------------------------------------------


def annualised_return(returns: np.ndarray, periods_per_year: float = 252.0,
                      compounding: bool = True) -> float:
    """Annualised return of a periodic return series.

    With ``compounding=True`` (default) this is the geometric rate; with
    ``compounding=False`` it is the arithmetic mean times ``periods_per_year``.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    n = r.size
    if compounding:
        total = float(np.prod(1.0 + r))
        if total <= 0:
            # portfolio wiped out (or worse); report the arithmetic rate
            return float(np.mean(r) * periods_per_year)
        return float(total ** (periods_per_year / n) - 1.0)
    return float(np.mean(r) * periods_per_year)


def annualised_volatility(returns: np.ndarray,
                          periods_per_year: float = 252.0,
                          ddof: int = 1) -> float:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    return float(np.std(r, ddof=ddof) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0,
                 periods_per_year: float = 252.0) -> float:
    """Annualised Sharpe ratio.

    ``risk_free`` is an **annual** rate and is de-annualised internally, so
    ``sharpe_ratio(r, risk_free=0.02)`` does the right thing on daily data.
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    rf_period = (1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf_period
    sd = np.std(excess, ddof=1)
    if sd <= 0:
        return float("nan")
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def sortino_ratio(returns: np.ndarray, target: float = 0.0,
                  periods_per_year: float = 252.0) -> float:
    """Annualised Sortino ratio using downside deviation below ``target``."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    downside = np.minimum(r - target, 0.0)
    dd = np.sqrt(np.mean(downside ** 2))
    if dd <= 0:
        return float("nan")
    return float((np.mean(r) - target) / dd * np.sqrt(periods_per_year))


def drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Drawdown path (values <= 0) of the compounded wealth index.

    The wealth index starts at 1.0 *before* the first return, and that initial
    level participates in the running peak.  Omitting it — e.g. by taking
    ``cumprod(1 + r)`` and running the maximum over only those points — drops
    the first period's drawdown entirely whenever the series opens with a
    loss, which understates the reported maximum drawdown.  The implementation
    below therefore prepends the starting wealth of 1.0 and returns a series
    aligned with ``returns`` (length ``len(returns)``, first entry <= 0).
    """
    r = np.asarray(returns, dtype=float).ravel()
    if r.size == 0:
        return np.empty(0, dtype=float)
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + r)))
    peak = np.maximum.accumulate(wealth)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, wealth / peak - 1.0, 0.0)
    # Drop the artificial starting point so the output aligns with ``returns``.
    return dd[1:]


def max_drawdown(returns: np.ndarray) -> float:
    """Maximum drawdown as a **positive** number (e.g. ``0.25`` for -25%)."""
    dd = drawdown_series(returns)
    if dd.size == 0:
        return float("nan")
    return float(-np.min(dd))


def calmar_ratio(returns: np.ndarray, periods_per_year: float = 252.0) -> float:
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd <= 0:
        return float("nan")
    return float(annualised_return(returns, periods_per_year) / mdd)


def value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical VaR at level ``alpha`` (a positive loss number)."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(-np.quantile(r, alpha))


def conditional_value_at_risk(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Expected shortfall below the ``alpha`` quantile (positive loss)."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    if tail.size == 0:
        return float(-q)
    return float(-np.mean(tail))


def skewness(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("nan")
    sd = np.std(r, ddof=0)
    if sd <= 0:
        return float("nan")
    return float(np.mean(((r - np.mean(r)) / sd) ** 3))


def kurtosis(returns: np.ndarray) -> float:
    """Excess kurtosis (0 for a normal distribution)."""
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size < 4:
        return float("nan")
    sd = np.std(r, ddof=0)
    if sd <= 0:
        return float("nan")
    return float(np.mean(((r - np.mean(r)) / sd) ** 4) - 3.0)


def ulcer_index(returns: np.ndarray) -> float:
    """Ulcer index: RMS of the drawdown path — penalises depth *and* duration."""
    dd = drawdown_series(returns)
    if dd.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(dd ** 2)))


def turnover(
    weights_new: np.ndarray,
    weights_old_drifted: np.ndarray,
) -> float:
    """One-way portfolio turnover.

    :math:`\\tfrac{1}{2} \\sum_i |w_i^{\\text{new}} - w_i^{\\text{old,drifted}}|`

    ``weights_old_drifted`` must be the target weights of the previous period
    carried forward through the realised returns of that period, *not* the raw
    previous targets — otherwise the measured turnover omits the passive drift
    that a real manager would have to trade against.
    """
    wn = np.asarray(weights_new, float).ravel()
    wo = np.asarray(weights_old_drifted, float).ravel()
    if wn.size != wo.size:
        raise ValueError("weight vectors must have the same length")
    return float(0.5 * np.sum(np.abs(wn - wo)))


# ---------------------------------------------------------------------------
# Aggregate container
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Full performance record for one strategy over one backtest.

    Metrics are stored as attributes; :meth:`to_dict` produces a flat,
    table-ready mapping.
    """

    name: str
    # --- return & risk ---
    ann_return: float = np.nan
    ann_volatility: float = np.nan
    sharpe: float = np.nan
    sortino: float = np.nan
    max_drawdown: float = np.nan
    calmar: float = np.nan
    # --- tail / shape ---
    var_95: float = np.nan
    cvar_95: float = np.nan
    skew: float = np.nan
    excess_kurtosis: float = np.nan
    ulcer: float = np.nan
    # --- trading ---
    avg_turnover: float = np.nan
    total_turnover: float = np.nan
    n_rebalances: int = 0
    # --- relative to a benchmark ---
    ann_return_net: float = np.nan
    sharpe_net: float = np.nan
    # --- misc ---
    n_periods: int = 0
    cumulative_return: float = np.nan
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "name": self.name,
            "ann_return": self.ann_return,
            "ann_volatility": self.ann_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "var_95": self.var_95,
            "cvar_95": self.cvar_95,
            "skew": self.skew,
            "excess_kurtosis": self.excess_kurtosis,
            "ulcer": self.ulcer,
            "avg_turnover": self.avg_turnover,
            "total_turnover": self.total_turnover,
            "n_rebalances": self.n_rebalances,
            "ann_return_net": self.ann_return_net,
            "sharpe_net": self.sharpe_net,
            "n_periods": self.n_periods,
            "cumulative_return": self.cumulative_return,
        }
        out.update(self.extra)
        return out

    @property
    def sharpe_loss_from_costs(self) -> float:
        if not (np.isfinite(self.sharpe) and np.isfinite(self.sharpe_net)):
            return float("nan")
        return float(self.sharpe - self.sharpe_net)

    def describe(self) -> str:
        return (
            f"{self.name}: ret={self.ann_return:.2%} vol={self.ann_volatility:.2%} "
            f"Sharpe={self.sharpe:.3f} (net {self.sharpe_net:.3f}) "
            f"MDD={self.max_drawdown:.2%} Calmar={self.calmar:.3f} "
            f"turnover={self.avg_turnover:.2%}"
        )


#: Display labels for metric keys.
METRIC_LABELS: Dict[str, str] = {
    "ann_return": "Ann. return",
    "ann_volatility": "Ann. volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max drawdown",
    "calmar": "Calmar",
    "var_95": "VaR 95%",
    "cvar_95": "CVaR 95%",
    "skew": "Skewness",
    "excess_kurtosis": "Excess kurtosis",
    "ulcer": "Ulcer index",
    "avg_turnover": "Avg. turnover",
    "total_turnover": "Total turnover",
    "ann_return_net": "Ann. return (net)",
    "sharpe_net": "Sharpe (net)",
}


def compute_metrics(
    returns: np.ndarray,
    name: str = "strategy",
    risk_free: float = 0.0,
    periods_per_year: float = 252.0,
    turnovers: Optional[np.ndarray] = None,
    returns_net: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> PerformanceMetrics:
    """Compute the full metric battery for a return series.

    Parameters
    ----------
    returns:
        Gross periodic returns (before transaction costs).
    risk_free:
        Annual risk-free rate.
    periods_per_year:
        252 for daily, 52 for weekly, 12 for monthly.
    turnovers:
        Optional per-period one-way turnover series.  When supplied, the
        average and total are recorded and used to derive the *net* metrics if
        ``returns_net`` is not given.
    returns_net:
        Optional net-of-cost return series (produced by the backtester).
    """
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]

    m = PerformanceMetrics(name=name)
    m.n_periods = int(r.size)
    if r.size == 0:
        return m

    m.ann_return = annualised_return(r, periods_per_year)
    m.ann_volatility = annualised_volatility(r, periods_per_year)
    m.sharpe = sharpe_ratio(r, risk_free, periods_per_year)
    m.sortino = sortino_ratio(r, 0.0, periods_per_year)
    m.max_drawdown = max_drawdown(r)
    m.calmar = calmar_ratio(r, periods_per_year)
    m.var_95 = value_at_risk(r, 0.05)
    m.cvar_95 = conditional_value_at_risk(r, 0.05)
    m.skew = skewness(r)
    m.excess_kurtosis = kurtosis(r)
    m.ulcer = ulcer_index(r)
    m.cumulative_return = float(np.prod(1.0 + r) - 1.0)

    if turnovers is not None:
        tv = np.asarray(turnovers, dtype=float).ravel()
        tv = tv[np.isfinite(tv)]
        if tv.size:
            m.avg_turnover = float(np.mean(tv))
            m.total_turnover = float(np.sum(tv))
            m.n_rebalances = int(tv.size)

    if returns_net is not None:
        rn = np.asarray(returns_net, dtype=float).ravel()
        rn = rn[np.isfinite(rn)]
        if rn.size:
            m.ann_return_net = annualised_return(rn, periods_per_year)
            m.sharpe_net = sharpe_ratio(rn, risk_free, periods_per_year)
    elif np.isfinite(m.sharpe):
        m.ann_return_net = m.ann_return
        m.sharpe_net = m.sharpe

    if extra:
        m.extra.update(extra)
    return m
