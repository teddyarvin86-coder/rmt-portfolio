#!/usr/bin/env python
"""Experiment — ex-ante risk-forecast accuracy of each covariance estimator.

This is the sharpest available test of a *risk model*, and it is deliberately
separate from the return-based backtest.

Motivation
----------
A covariance estimator is a forecast.  Its job is to predict the variance of a
portfolio over the coming holding period.  Return-based metrics (Sharpe,
Calmar) confound the quality of that forecast with the realised return path,
which is dominated by noise over the few years of data available.  Risk-model
accuracy can instead be measured directly and almost noiselessly:

    for every rebalance date t and every strategy
        predicted_vol(t) = sqrt(w' Sigma_hat(t) w)
        realised_vol(t)  = sample std of w' r_{t+1 .. t+h}

Then compare the two with three statistics that need no expected-return model:

``MAE`` / ``RMSE``
    Absolute accuracy of the forecast in daily-volatility units.
``Bias``
    ``mean(predicted)/mean(realised) - 1``.  A **positive** bias means the
    model *over*-estimates risk (conservative); negative means it under-states
    risk, which is the dangerous direction for a risk manager.
``QLIKE``
    The quasi-likelihood loss of Patton (2011),
    ``mean(h_hat/h - log(h_hat/h) - 1)`` with ``h`` the variance.  This is a
    *robust* loss: it is the only standard loss function that ranks
    volatility forecasts correctly even when the volatility proxy (realised
    variance) is noisy.  Lower is better; 0 is perfect.

Why this matters for the RMT story
----------------------------------
The prediction from the theory is specific and testable: the sample
covariance is *over-dispersed*, so a portfolio that loads on its smallest
eigenvalues (minimum variance) will have its risk **under-estimated**, and the
error grows with ``q = N/T``.  Denoising should therefore cut the negative
bias and the QLIKE loss, and the effect should be **larger for minimum
variance than for HRP** — because HRP never inverts the matrix and so is not
exposed to the smallest eigenvalues.  This script tests exactly that
prediction.

Outputs
-------
``results/risk_forecast/<universe>_risk_forecast_<tag>.csv``
    One row per (date, estimator, allocator) with predicted and realised vol.
``results/risk_forecast/<universe>_risk_summary_<tag>.csv``
    Aggregated accuracy statistics per (estimator, allocator).
``results/risk_forecast/<universe>_risk_by_estimator_<tag>.csv``
    Averaged over allocators.
``results/risk_forecast/<universe>_risk_eigen_<tag>.csv``
    Regressions of the forecast bias on the eigenvalue dispersion: tests the
    theoretical mechanism directly.

Usage
-----
    python experiments/run_risk_forecast.py --universe us_sector_etf
    python experiments/run_risk_forecast.py --universe us_single_stock \
        --window 120 --tag highq
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rmt_portfolio.backtest.engine import FREQUENCY_DAYS
from rmt_portfolio.backtest.metrics import annualised_volatility
from rmt_portfolio.config import PROJECT_ROOT, load_config
from rmt_portfolio.data import load_returns
from rmt_portfolio.portfolio import ALLOCATORS, effective_number_of_bets
from rmt_portfolio.rmt import ESTIMATORS, eigen_decompose, sample_cov
from rmt_portfolio.utils import ensure_dir, get_logger, save_json, save_table, \
    set_global_seed, Timer

LOGGER = get_logger("experiments.risk_forecast")

TRADING_DAYS = 252.0


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def qlike(pred_var: np.ndarray, real_var: np.ndarray) -> float:
    """Patton (2011) QLIKE loss, robust to a noisy volatility proxy.

    .. math::

        L = \\frac{1}{T}\\sum_t \\left[
              \\frac{\\hat h_t}{h_t} - \\log\\frac{\\hat h_t}{h_t} - 1 \\right]

    where :math:`\\hat h` is the predicted variance and :math:`h` the realised
    variance.  The loss is non-negative, zero iff the forecast is perfect, and
    — crucially — it orders forecasts by their true accuracy even when
    :math:`h` is measured with error.
    """
    ratio = np.maximum(pred_var, 1e-300) / np.maximum(real_var, 1e-300)
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def mse_var(pred_var: np.ndarray, real_var: np.ndarray) -> float:
    return float(np.mean((pred_var - real_var) ** 2))


def rmse_var(pred_var: np.ndarray, real_var: np.ndarray) -> float:
    return float(np.sqrt(mse_var(pred_var, real_var)))


def mae_vol(pred_vol: np.ndarray, real_vol: np.ndarray) -> float:
    return float(np.mean(np.abs(pred_vol - real_vol)))


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def realised_variance(
    fwd: np.ndarray,
    w: np.ndarray,
    demean: bool = True,
    ddof: int = 1,
) -> float:
    """Realised variance of the portfolio ``w`` over the forward window.

    The portfolio is *held* over the window, so the correct realised variance
    is that of the buy-and-hold portfolio return series ``fwd @ w``.  We
    demean by default because the ex-ante estimator is centred too; a parallel
    run without demeaning (``--no-demean``) checks that the conclusion is not
    an artefact of that choice.
    """
    p = fwd @ w
    if demean:
        p = p - p.mean()
    return float(np.sum(p ** 2) / max(len(p) - ddof, 1))


def run_forecast_grid(
    returns: pd.DataFrame,
    estimators: List[str],
    allocators: List[str],
    window: int,
    holding_days: int,
    max_weight: Optional[float],
    long_only: bool = True,
    demean: bool = True,
    sigma_method: str = "median_bulk",
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk forward, recording predicted and realised risk for every cell."""
    R = np.asarray(returns.values, dtype=float)
    dates = returns.index
    n = R.shape[1]
    rows = []

    reb_dates = list(range(window, len(R) - holding_days, holding_days))
    if verbose:
        LOGGER.info("risk-forecast grid: %d rebalances x %d estimators x "
                    "%d allocators", len(reb_dates), len(estimators),
                    len(allocators))

    for k, t in enumerate(reb_dates):
        est = R[t - window:t]
        fwd = R[t:t + holding_days]
        cov_sample = sample_cov(est)
        dec = eigen_decompose(cov_sample, window)
        ev = dec.eigenvalues
        # spectral dispersion diagnostics recorded per date
        eig_diag = {
            "cond_number": float(ev[-1] / max(ev[0], 1e-300)),
            "eff_rank": dec.effective_rank(),
            "top1_share": float(ev[-1] / ev.sum()),
            "q": n / float(window),
            "lambda_max": float(ev[-1]),
            "lambda_min": float(ev[0]),
        }

        for meth in estimators:
            cov = ESTIMATORS[meth](est, n_obs=window, sigma_method=sigma_method)
            for alloc in allocators:
                try:
                    res = ALLOCATORS[alloc](cov=cov, long_only=long_only,
                                            max_weight=max_weight)
                except TypeError:
                    # allocators that do not accept long_only (e.g. hrp)
                    res = ALLOCATORS[alloc](cov=cov)
                w = np.asarray(res.weights, float)
                if not np.all(np.isfinite(w)) or w.sum() <= 0:
                    continue
                w = w / w.sum()

                pred_var = max(float(w @ cov @ w), 1e-300)
                real_var = max(realised_variance(fwd, w, demean=demean),
                               1e-300)
                rows.append({
                    "date": dates[t],
                    "estimator": meth,
                    "allocator": alloc,
                    "pred_vol": np.sqrt(pred_var),
                    "real_vol": np.sqrt(real_var),
                    "pred_var": pred_var,
                    "real_var": real_var,
                    "enb": effective_number_of_bets(w, cov),
                    "hhi": float(np.sum(w ** 2)),
                    **eig_diag,
                })
        if verbose and (k + 1) % 12 == 0:
            LOGGER.info("  rebalance %d/%d", k + 1, len(reb_dates))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy statistics per (estimator, allocator) cell."""
    rows = []
    for (meth, alloc), sub in df.groupby(["estimator", "allocator"]):
        pv = sub["pred_vol"].values
        rv = sub["real_vol"].values
        rows.append({
            "estimator": meth,
            "allocator": alloc,
            "pred_vol_mean": float(pv.mean()),
            "real_vol_mean": float(rv.mean()),
            "bias_pct": float(pv.mean() / rv.mean() - 1.0) * 100.0,
            "mae_vol": mae_vol(pv, rv),
            "rmse_vol": float(np.sqrt(np.mean((pv - rv) ** 2))),
            "qlike": qlike(sub["pred_var"].values, sub["real_var"].values),
            "mse_var": mse_var(sub["pred_var"].values, sub["real_var"].values),
            "corr_pred_real": float(np.corrcoef(pv, rv)[0, 1])
                              if pv.std() > 0 and rv.std() > 0 else np.nan,
            "avg_enb": float(sub["enb"].mean()),
            # `cond_number` / `eff_rank` are properties of the *date*, not of
            # the estimator, so they are identical across estimators at a given
            # date.  They are reported here only for reference (the mean over
            # the sample); they must not be read as an estimator attribute.
            "sample_avg_cond": float(sub["cond_number"].mean()),
            "sample_avg_eff_rank": float(sub["eff_rank"].mean()),
            "q": float(sub["q"].mean()),
            "n": int(len(sub)),
        })
    return pd.DataFrame(rows).set_index(["estimator", "allocator"])


def by_estimator(summary: pd.DataFrame) -> pd.DataFrame:
    agg = summary.groupby("estimator").agg(
        qlike=("qlike", "mean"),
        mae_vol=("mae_vol", "mean"),
        rmse_vol=("rmse_vol", "mean"),
        bias_pct=("bias_pct", "mean"),
        corr_pred_real=("corr_pred_real", "mean"),
        avg_enb=("avg_enb", "mean"),
        sample_avg_cond=("sample_avg_cond", "mean"),
    ).sort_values("qlike")
    return agg


def by_allocator(summary: pd.DataFrame) -> pd.DataFrame:
    agg = summary.groupby("allocator").agg(
        qlike=("qlike", "mean"),
        mae_vol=("mae_vol", "mean"),
        rmse_vol=("rmse_vol", "mean"),
        bias_pct=("bias_pct", "mean"),
        corr_pred_real=("corr_pred_real", "mean"),
        avg_enb=("avg_enb", "mean"),
    ).sort_values("qlike")
    return agg


def eigen_bias_regression(df: pd.DataFrame) -> pd.DataFrame:
    """Relate the forecast bias to the spectral dispersion across dates.

    For each (estimator, allocator) cell we regress the log forecast ratio
    ``log(pred_vol / real_vol)`` on the log condition number of the sample
    covariance.  The theory predicts a **positive** slope for the sample
    estimator under minimum variance: the more ill-conditioned the sample
    matrix, the more the optimiser exploits spurious low-variance directions
    and the more the risk is under-stated (so the ratio falls, giving a
    *negative* slope when the matrix is badly conditioned).  A denoised
    estimator should show a slope closer to zero.
    """
    rows = []
    for (meth, alloc), sub in df.groupby(["estimator", "allocator"]):
        x = np.log(np.maximum(sub["cond_number"].values, 1.0))
        y = np.log(np.maximum(sub["pred_vol"].values, 1e-300)
                   / np.maximum(sub["real_vol"].values, 1e-300))
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 8:
            continue
        xc, yc = x[mask], y[mask]
        xbar, ybar = float(xc.mean()), float(yc.mean())
        xd = xc - xbar
        yd = yc - ybar
        denom = float(np.sum(xd ** 2))
        if denom <= 0:
            continue
        slope = float(np.sum(xd * yd) / denom)
        intercept = float(ybar - slope * xbar)
        resid = yd - slope * xd                  # residuals about the fitted line
        ss_tot = float(np.sum(yd ** 2))
        r2 = 1.0 - float(np.sum(resid ** 2)) / max(ss_tot, 1e-300)
        rows.append({
            "estimator": meth, "allocator": alloc,
            "slope_logbias_vs_logcond": slope,
            "intercept": intercept, "r2": r2, "n": int(mask.sum()),
        })
    return pd.DataFrame(rows).set_index(["estimator", "allocator"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--frequency", default=None)
    ap.add_argument("--max-weight", default=None,
                    help="per-asset cap; 'none' for unconstrained")
    ap.add_argument("--sigma-method", default=None)
    ap.add_argument("--no-demean", action="store_true",
                    help="use raw (non-demeaned) realised variance")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--estimators", nargs="*", default=None)
    ap.add_argument("--allocators", nargs="*", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(int(cfg.project.seed))
    universe = args.universe or str(cfg.data.universe)
    window = int(args.window or cfg.backtest.window)
    frequency = args.frequency or str(cfg.backtest.frequency)
    sigma_method = args.sigma_method or str(cfg.rmt.sigma_method)

    if args.max_weight is None:
        mw = cfg.backtest.max_weight
        max_weight = float(mw) if mw else None
    elif str(args.max_weight).lower() in ("none", "null", ""):
        max_weight = None
    else:
        max_weight = float(args.max_weight)

    estimators = [e for e in (args.estimators or list(cfg.estimators))
                  if e in ESTIMATORS]
    allocators = [a for a in (args.allocators or list(cfg.allocators))
                  if a in ALLOCATORS]
    holding_days = FREQUENCY_DAYS[frequency]

    out_dir = ensure_dir(PROJECT_ROOT / "results" / "risk_forecast")
    LOGGER.info("=" * 78)
    LOGGER.info("EX-ANTE RISK-FORECAST ACCURACY | universe=%s window=%d "
                "freq=%s max_weight=%s", universe, window, frequency,
                max_weight)
    LOGGER.info("sigma_method=%s  demean=%s", sigma_method,
                not args.no_demean)
    LOGGER.info("=" * 78)

    with Timer("data load", LOGGER):
        rets, quality = load_returns(
            universe, start=str(cfg.data.start), end=str(cfg.data.end),
            method="simple", min_coverage=float(cfg.data.min_coverage),
            repair_splits=bool(cfg.data.repair_splits),
            align=str(cfg.data.align), winsorise=cfg.data.winsorise,
            verbose=False)
    LOGGER.info("panel: %d dates x %d assets  (%s .. %s)", rets.shape[0],
                rets.shape[1], rets.index.min().date(), rets.index.max().date())

    with Timer("forecast grid", LOGGER):
        df = run_forecast_grid(
            rets, estimators, allocators, window, holding_days, max_weight,
            demean=not args.no_demean, sigma_method=sigma_method, verbose=True)

    if df.empty:
        LOGGER.error("no forecast rows produced")
        return 1

    summary = summarise(df)
    est_tab = by_estimator(summary)
    alloc_tab = by_allocator(summary)
    eigen_tab = eigen_bias_regression(df)

    save_table(df, out_dir / f"{universe}_risk_forecast_{args.tag}.csv",
               index=False)
    save_table(summary, out_dir / f"{universe}_risk_summary_{args.tag}.csv")
    save_table(est_tab, out_dir / f"{universe}_risk_by_estimator_{args.tag}.csv")
    save_table(alloc_tab,
               out_dir / f"{universe}_risk_by_allocator_{args.tag}.csv")
    if not eigen_tab.empty:
        save_table(eigen_tab,
                   out_dir / f"{universe}_risk_eigen_{args.tag}.csv")

    LOGGER.info("-" * 78)
    LOGGER.info("RISK-FORECAST ACCURACY BY ESTIMATOR (averaged over allocators)")
    LOGGER.info("-" * 78)
    LOGGER.info("\n%s", est_tab.round(5).to_string())

    LOGGER.info("-" * 78)
    LOGGER.info("RISK-FORECAST ACCURACY BY ALLOCATOR (averaged over estimators)")
    LOGGER.info("-" * 78)
    LOGGER.info("\n%s", alloc_tab.round(5).to_string())

    LOGGER.info("-" * 78)
    LOGGER.info("FULL CELLS (sorted by QLIKE)")
    LOGGER.info("-" * 78)
    show = summary[["bias_pct", "mae_vol", "rmse_vol", "qlike",
                    "corr_pred_real", "avg_enb", "sample_avg_cond"]]
    LOGGER.info("\n%s", show.sort_values("qlike").round(5).to_string())

    if not eigen_tab.empty:
        LOGGER.info("-" * 78)
        LOGGER.info("BIAS-vs-CONDITION-NUMBER REGRESSION")
        LOGGER.info("slope > 0 => worse conditioning is associated with "
                    "under-stated risk")
        LOGGER.info("-" * 78)
        LOGGER.info("\n%s", eigen_tab.round(4).to_string())

    save_json({
        "universe": universe,
        "window": window, "frequency": frequency,
        "max_weight": max_weight, "sigma_method": sigma_method,
        "demean": not args.no_demean,
        "panel": {"n_dates": int(rets.shape[0]), "n_assets": int(rets.shape[1]),
                  "start": str(rets.index.min().date()),
                  "end": str(rets.index.max().date())},
        "best_by_qlike": summary["qlike"].idxmin(),
        "best_qlike": float(summary["qlike"].min()),
        "sample_minvar_qlike": float(
            summary.loc[("sample", "minimum_variance"), "qlike"])
            if ("sample", "minimum_variance") in summary.index else None,
    }, out_dir / f"{universe}_risk_summary_{args.tag}.json")

    LOGGER.info("risk-forecast outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
