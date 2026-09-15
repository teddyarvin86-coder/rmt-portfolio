"""Rolling out-of-sample backtest engine.

Design
------
The engine is deliberately *walk-forward* and free of look-ahead:

::

    |<---- estimation window T ---->|<-- holding period h -->|
                                    ^                        ^
                                 rebalance              rebalance

At each rebalance date ``t``:

1. Take the trailing ``T`` observations ``r[t-T : t]``.
2. Estimate and (optionally) denoise the covariance matrix using **only**
   those observations — this is the crux of the RMT study, since ``q = N/T``
   is fixed by the same window.
3. Allocate weights with the chosen allocator.
4. Hold for ``h`` periods (monthly = 21 trading days, weekly = 5), accruing
   the realised returns **of the next period(s)** — never of the window.
5. Record the drifted weights, compute turnover against the previous target,
   and charge transaction costs.

Transaction costs
-----------------
Two models are supported:

* ``'proportional'`` — cost ``= c * turnover * NAV`` with ``c`` in basis
  points.  ``turnover`` is one-way, so a ``c = 10`` bp assumption charges
  10 bp per unit of one-way turnover.
* ``'linear_plus_impact'`` — adds a square-root market-impact term
  ``k * sqrt(turnover)``, which penalises large abrupt reallocations more than
  the proportional model.  This is closer to how real execution behaves and
  makes the *stability* benefit of RMT denoising show up more sharply.

The engine also supports an optional **no-trade band**: if the L1 distance
between the new target and the drifted current holdings is below a threshold
``band``, the portfolio is left untouched and turnover is recorded as zero.
This is the standard practical method for suppressing turnover.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rmt_portfolio.backtest.metrics import PerformanceMetrics, compute_metrics, turnover
from rmt_portfolio.portfolio.optimizers import ALLOCATORS, PortfolioResult, allocate
from rmt_portfolio.rmt.denoise import estimate_covariance

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "rebalance_dates",
    "TRADING_DAYS_PER_YEAR",
]

TRADING_DAYS_PER_YEAR = 252

#: Rebalance frequency -> number of trading days in one holding period.
FREQUENCY_DAYS: Dict[str, int] = {
    "daily": 1,
    "weekly": 5,
    "biweekly": 10,
    "monthly": 21,
    "quarterly": 63,
    "semiannual": 126,
    "annual": 252,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """Everything that parameterises one backtest run.

    Attributes
    ----------
    window:
        Estimation window length ``T`` in periods (trading days).
    frequency:
        Rebalance frequency key of :data:`FREQUENCY_DAYS`, or an explicit
        integer number of days.
    cost_bps:
        One-way proportional transaction cost in basis points.
    cost_model:
        ``'proportional'`` or ``'linear_plus_impact'``.
    impact_coef:
        Coefficient ``k`` of the square-root impact term (only used by the
        ``'linear_plus_impact'`` model), in basis points.
    no_trade_band:
        Suppress a rebalance when the L1 weight change is below this value.
    risk_free:
        Annual risk-free rate used by the Sharpe/Sortino metrics.
    long_only:
        Passed to allocators that support it.
    max_weight:
        Optional per-asset cap.
    periods_per_year:
        Annualisation factor; defaults to 252.
    cov_kwargs:
        Extra keyword arguments forwarded to the covariance estimator
        (e.g. ``sigma_method``, ``lambda_plus``, ``beta``, ``n_factors``).
    alloc_kwargs:
        Extra keyword arguments forwarded to the allocator
        (e.g. ``linkage_method``, ``n_clusters``, ``budget``).
    """

    window: int = 252
    frequency: str | int = "monthly"
    cost_bps: float = 10.0
    cost_model: str = "proportional"
    impact_coef: float = 5.0
    no_trade_band: float = 0.0
    risk_free: float = 0.0
    long_only: bool = True
    max_weight: Optional[float] = None
    periods_per_year: float = TRADING_DAYS_PER_YEAR
    cov_kwargs: Dict[str, Any] = field(default_factory=dict)
    alloc_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.window < 20:
            raise ValueError("window must be at least 20 observations")
        if self.cost_model not in ("proportional", "linear_plus_impact"):
            raise ValueError(
                f"unknown cost_model {self.cost_model!r}")

    @property
    def holding_days(self) -> int:
        if isinstance(self.frequency, int):
            return int(self.frequency)
        key = str(self.frequency).lower()
        if key not in FREQUENCY_DAYS:
            raise ValueError(
                f"unknown frequency {self.frequency!r}; "
                f"available: {sorted(FREQUENCY_DAYS)}")
        return FREQUENCY_DAYS[key]

    @property
    def rebalances_per_year(self) -> float:
        return self.periods_per_year / max(self.holding_days, 1)

    def cost_of_turnover(self, turnover_value: float) -> float:
        """Cost (as a fraction of NAV) of a given one-way turnover."""
        proportional = (self.cost_bps / 1e4) * turnover_value
        if self.cost_model == "linear_plus_impact":
            return proportional + (self.impact_coef / 1e4) * np.sqrt(
                max(turnover_value, 0.0))
        return proportional


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Full record of one walk-forward run."""

    name: str
    config: BacktestConfig
    dates: pd.DatetimeIndex                       # holding-period return dates
    gross_returns: np.ndarray
    net_returns: np.ndarray
    turnover_series: np.ndarray
    cost_series: np.ndarray
    weights_history: pd.DataFrame                 # index = rebalance date
    metrics: PerformanceMetrics
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    extra: Dict[str, Any] = field(default_factory=dict)

    # -- convenient derived series ---------------------------------------

    @property
    def gross_equity(self) -> pd.Series:
        return pd.Series(np.cumprod(1.0 + self.gross_returns),
                         index=self.dates, name=f"{self.name}_gross")

    @property
    def net_equity(self) -> pd.Series:
        return pd.Series(np.cumprod(1.0 + self.net_returns),
                         index=self.dates, name=f"{self.name}_net")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            "gross_return": self.gross_returns,
            "net_return": self.net_returns,
            "turnover": self.turnover_series,
            "cost": self.cost_series,
        }, index=self.dates)

    def describe(self) -> str:
        return self.metrics.describe()


# ---------------------------------------------------------------------------
# Rebalance schedule
# ---------------------------------------------------------------------------


def rebalance_dates(n_obs: int, window: int, holding_days: int,
                    start_offset: int = 0) -> np.ndarray:
    """Indices ``t`` at which a portfolio is formed.

    Returns the positions ``t`` such that the estimation window is
    ``[t - window, t)`` and the holding period is ``[t, t + holding_days)``.
    The last returned index always satisfies ``t + holding_days <= n_obs``.
    """
    first = window + int(start_offset)
    if first >= n_obs:
        return np.array([], dtype=int)
    last = n_obs - holding_days
    if last < first:
        return np.array([], dtype=int)
    return np.arange(first, last + 1, holding_days, dtype=int)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def run_backtest(
    returns: pd.DataFrame,
    cov_method: str = "sample",
    allocator: str = "minimum_variance",
    config: Optional[BacktestConfig] = None,
    name: Optional[str] = None,
    corr_method: Optional[str] = None,
    verbose: bool = False,
) -> BacktestResult:
    """Run a single walk-forward backtest.

    Parameters
    ----------
    returns:
        ``T_dates x N_assets`` DataFrame of **simple** periodic returns, with a
        DatetimeIndex.  Must contain no NaN (align and ffill upstream).
    cov_method:
        Covariance estimator key (see
        :data:`rmt_portfolio.rmt.denoise.ESTIMATORS`).
    allocator:
        Allocator key (see :data:`rmt_portfolio.portfolio.optimizers.ALLOCATORS`).
    config:
        :class:`BacktestConfig`; defaults are used if omitted.
    name:
        Label for the run; defaults to ``"{cov_method}+{allocator}"``.
    corr_method:
        Optional *separate* estimator used only to build the correlation
        matrix handed to clustering-based allocators (HRP).  When ``None`` the
        correlation implied by ``cov_method``'s covariance is used.  This lets
        you isolate "denoise the risk model" from "denoise the clustering
        input".
    verbose:
        Print progress every rebalance.
    """
    cfg = config or BacktestConfig()
    label = name or f"{cov_method}+{allocator}"
    R = np.asarray(returns.values, dtype=float)
    n_obs, n_assets = R.shape
    dates = returns.index

    if not np.all(np.isfinite(R)):
        raise ValueError("returns contains non-finite values; clean it first")
    if cfg.window >= n_obs:
        raise ValueError(
            f"window ({cfg.window}) must be smaller than the sample "
            f"({n_obs})")

    holding = cfg.holding_days
    reb_idx = rebalance_dates(n_obs, cfg.window, holding)
    if reb_idx.size == 0:
        raise ValueError("no rebalance dates: sample too short for the window")

    n_reb = reb_idx.size
    target_weights: List[np.ndarray] = []
    reb_dates: List[pd.Timestamp] = []
    diag_rows: List[Dict[str, Any]] = []

    for k, t in enumerate(reb_idx):
        win = R[t - cfg.window:t]                      # estimation window only

        # --- 1. covariance estimation / denoising ------------------------
        cov_kwargs = dict(cfg.cov_kwargs)
        cov = estimate_covariance(win, method=cov_method, n_obs=cfg.window,
                                  **cov_kwargs)
        if isinstance(cov, tuple):
            cov = cov[0]

        # --- 2. correlation for clustering ------------------------------
        if corr_method is not None and corr_method != cov_method:
            corr_cov = estimate_covariance(win, method=corr_method,
                                           n_obs=cfg.window, **cov_kwargs)
            if isinstance(corr_cov, tuple):
                corr_cov = corr_cov[0]
        else:
            corr_cov = cov
        # NOTE: HRP re-derives the correlation internally from this matrix
        # because the allocator's contract is (cov, corr) with identical
        # scaling; passing the covariance keeps the two consistent.

        # --- 3. allocation ----------------------------------------------
        alloc_kwargs = dict(cfg.alloc_kwargs)
        if allocator == "minimum_variance" and cfg.max_weight is not None:
            alloc_kwargs.setdefault("max_weight", cfg.max_weight)
        if allocator in ("minimum_variance", "max_diversification"):
            alloc_kwargs.setdefault("long_only", cfg.long_only)

        res: PortfolioResult = allocate(
            cov, method=allocator,
            corr=corr_cov if allocator == "hrp" else None,
            **alloc_kwargs,
        )
        w = res.weights

        # --- 4. no-trade band -------------------------------------------
        if cfg.no_trade_band > 0 and target_weights:
            prev = target_weights[-1]
            l1 = float(np.sum(np.abs(w - prev)))
            if 0.5 * l1 < cfg.no_trade_band:
                w = prev.copy()
                diag_rows.append({
                    "date": dates[t], "vol": res.volatility, "enb": res.enb,
                    "dr": res.diversification_ratio, "no_trade": 1,
                })
                target_weights.append(w)
                reb_dates.append(dates[t])
                if verbose:
                    print(f"[{k+1}/{n_reb}] {dates[t].date()} no-trade")
                continue

        target_weights.append(w)
        reb_dates.append(dates[t])
        diag_rows.append({
            "date": dates[t],
            "vol": res.volatility,
            "enb": res.enb,
            "dr": res.diversification_ratio,
            "weight_min": float(w.min()),
            "weight_max": float(w.max()),
            "weight_hhi": float(np.sum(w ** 2)),
            "n_nonzero": int(np.sum(w > 1e-8)),
            "no_trade": 0,
        })
        if verbose and (k % 12 == 0 or k == n_reb - 1):
            print(f"[{k+1}/{n_reb}] {dates[t].date()} vol={res.volatility:.4f} "
                  f"ENB={res.enb:.1f}")

    # ---------------- realise returns ---------------------------------
    gross = np.zeros(n_reb, dtype=float)
    net = np.zeros(n_reb, dtype=float)
    tvs = np.zeros(n_reb, dtype=float)
    costs = np.zeros(n_reb, dtype=float)
    hold_dates: List[pd.Timestamp] = []

    prev_w: Optional[np.ndarray] = None

    for k, t in enumerate(reb_idx):
        w_target = target_weights[k]

        # --- turnover: compare the new target against where the previous
        #     target would have drifted over its own holding period --------
        if prev_w is None:
            # first rebalance: buy the whole portfolio from cash
            traded = turnover(w_target, np.zeros(n_assets))
        else:
            drifted_prev = _drift(prev_w, R[t - holding:t])
            traded = turnover(w_target, drifted_prev)
            # a suppressed (no-trade-band) rebalance leaves w_target == prev_w
            # but the *dragged* weights still differ; the band means we do not
            # trade, so turnover is exactly zero.
            if cfg.no_trade_band > 0 and np.array_equal(w_target, prev_w):
                traded = 0.0

        tvs[k] = traded
        cost = cfg.cost_of_turnover(traded)
        costs[k] = cost

        # --- realised return of the holding period ----------------------
        window_ret = R[t:t + holding]
        daily = window_ret @ w_target
        period_ret = float(np.prod(1.0 + daily) - 1.0)

        gross[k] = period_ret
        net[k] = period_ret - cost
        hold_dates.append(dates[t + holding - 1])
        prev_w = w_target

    # ---------------- diagnostics -------------------------------------
    wdf = pd.DataFrame(
        np.vstack(target_weights),
        index=pd.DatetimeIndex(reb_dates),
        columns=list(returns.columns),
    )
    diag = pd.DataFrame(diag_rows)
    if not diag.empty:
        diag = diag.set_index("date")

    # annualise metrics using the holding period as the period
    p_per_year = cfg.periods_per_year / holding
    metrics = compute_metrics(
        gross, name=label, risk_free=cfg.risk_free,
        periods_per_year=p_per_year, turnovers=tvs, returns_net=net,
        extra={
            "cov_method": cov_method,
            "allocator": allocator,
            "window": cfg.window,
            "holding_days": holding,
            "rebalances_per_year": p_per_year,
            "n_rebalances": n_reb,
            "cost_bps": cfg.cost_bps,
            "no_trade_band": cfg.no_trade_band,
            "avg_vol": float(np.nanmean(diag["vol"])) if not diag.empty else np.nan,
            "avg_enb": float(np.nanmean(diag["enb"])) if not diag.empty else np.nan,
            "avg_dr": float(np.nanmean(diag["dr"])) if not diag.empty else np.nan,
            "avg_weight_hhi": float(np.nanmean(diag["weight_hhi"]))
                               if "weight_hhi" in diag else np.nan,
            "total_cost": float(np.sum(costs)),
        },
    )

    return BacktestResult(
        name=label,
        config=cfg,
        dates=pd.DatetimeIndex(hold_dates),
        gross_returns=gross,
        net_returns=net,
        turnover_series=tvs,
        cost_series=costs,
        weights_history=wdf,
        metrics=metrics,
        diagnostics=diag,
        extra={"corr_method": corr_method},
    )


def _drift(weights: np.ndarray, period_returns: np.ndarray) -> np.ndarray:
    """Grow weights through realised returns, then renormalise.

    ``period_returns`` is ``h x N``.  The weights are compounded day by day
    (matching a constant-mix implementation within the holding period) and
    normalised so they sum to one.
    """
    w = np.asarray(weights, float).copy()
    for r in np.atleast_2d(period_returns):
        w = w * (1.0 + r)
        s = w.sum()
        if s > 0:
            w = w / s
    return w


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------


def run_matrix(
    returns: pd.DataFrame,
    cov_methods: Sequence[str],
    allocators: Sequence[str],
    config: Optional[BacktestConfig] = None,
    extra_alloc_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, BacktestResult]:
    """Run the full ``cov_method x allocator`` grid and label each run.

    Returns a dict keyed by ``"{cov_method}+{allocator}"``.  Failures are
    reported and skipped rather than aborting the sweep, which matters when a
    solver occasionally fails on a degenerate window.

    ``extra_alloc_kwargs`` is merged into the allocator keyword arguments for
    every run; it is used by the robustness sweeps to vary a constraint (for
    example the per-asset cap) without rebuilding the configuration object.
    """
    out: Dict[str, BacktestResult] = {}
    total = len(cov_methods) * len(allocators)
    i = 0
    for cm in cov_methods:
        for al in allocators:
            i += 1
            key = f"{cm}+{al}"
            if verbose:
                print(f"  [{i}/{total}] {key}")
            try:
                cfg = config or BacktestConfig()
                if extra_alloc_kwargs:
                    cfg = _with_extra_alloc_kwargs(cfg, extra_alloc_kwargs)
                out[key] = run_backtest(returns, cov_method=cm,
                                        allocator=al, config=cfg,
                                        name=key)
            except Exception as exc:      # pragma: no cover - defensive
                print(f"    !! {key} failed: {type(exc).__name__}: {exc}")
    return out


def _with_extra_alloc_kwargs(config: BacktestConfig,
                             extra: Dict[str, Any]) -> BacktestConfig:
    """Return a copy of ``config`` with ``extra`` merged into ``alloc_kwargs``."""
    import dataclasses

    merged = dict(config.alloc_kwargs)
    merged.update(extra)
    return dataclasses.replace(config, alloc_kwargs=merged)
