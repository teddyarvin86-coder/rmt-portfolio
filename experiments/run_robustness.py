#!/usr/bin/env python
"""Experiment — robustness sweeps.

Every headline conclusion in a study like this must be shown to survive
reasonable changes in the design choices.  This script runs the five sweeps
that the literature regards as mandatory, plus one that is specific to RMT.

Sweeps
------
1. **Window length ``T``** — varies the aspect ratio :math:`q = N/T`
   (:math:`T \\in \\{120, 252, 500, 756\\}`).  This is the *key* RMT sweep:
   the theory predicts that the benefit of denoising grows with ``q``, so the
   ordering of estimators must change systematically along this axis, and the
   advantage of RMT must be largest at small ``T``.

2. **Rebalance frequency** — weekly / monthly / quarterly.  More frequent
   rebalancing means more turnover and, for a covariance estimator, less time
   for the error in the estimate to matter.

3. **Threshold multiplier** — :math:`\\lambda_+` multiplied by
   :math:`\\{0.85, 0.9, 1.0, 1.1, 1.15\\}`.  Tests whether the result is
   driven by the *exact* MP edge or is robust to mis-specifying it.  If the
   performance surface is flat in the neighbourhood of 1.0, the method is not
   relying on a lucky calibration.

4. **Transaction costs** — 0 / 5 / 10 / 20 / 40 bp.  Denoising changes
   turnover (usually reducing it), so the net effect must be measured across
   the cost spectrum.

5. **Noise-scale (sigma) estimation method** — ``median_bulk`` / ``mean_bulk``
   / ``max_eigen`` / ``cdf_fit``.  Four different ways of calibrating the MP
   law; the conclusion must not hinge on one of them.

6. **Universe** — repeated across the US sector ETFs, the Chinese sector
   ETFs and a 30-name US single-stock panel.  Different ``N``, different
   correlation structure, different market.

The output of each sweep is a tidy table with the Sharpe ratio, ex-ante risk
forecast quality, turnover and turnover-adjusted Sharpe for every
(estimator, allocator) cell, so that the *pattern* — not just a single number —
can be read off.

Usage
-----
    python experiments/run_robustness.py --sweep all
    python experiments/run_robustness.py --sweep window
    python experiments/run_robustness.py --sweep sigma --universe cn_sector_etf
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rmt_portfolio.backtest import BacktestConfig, run_matrix
from rmt_portfolio.backtest.engine import FREQUENCY_DAYS
from rmt_portfolio.config import PROJECT_ROOT, load_config
from rmt_portfolio.data import load_returns
from rmt_portfolio.portfolio import ALLOCATORS
from rmt_portfolio.rmt import ESTIMATORS
from rmt_portfolio.utils import ensure_dir, get_logger, save_json, save_table, \
    set_global_seed, Timer

LOGGER = get_logger("experiments.robustness")

SWEEPS = ("window", "frequency", "threshold", "cost", "sigma", "universe")

#: Which *column* each sweep varies.  ``sigma`` varies ``sigma_method``; the
#: threshold sweep labels rows with the multiplier, not the raw threshold.
SWEEP_KEY = {
    "window": "window",
    "frequency": "frequency",
    "threshold": "lambda_plus_mult",
    "cost": "cost_bps",
    "sigma": "sigma_method",
    "universe": "universe",
}


# ---------------------------------------------------------------------------
# One backtest -> one tidy row set
# ---------------------------------------------------------------------------


def _run_cell(
    rets: pd.DataFrame,
    universe: str,
    label_key: str,
    label_value,
    window: int,
    frequency: str,
    cost_bps: float,
    cost_model: str,
    max_weight: Optional[float],
    sigma_method: str,
    estimators: List[str],
    allocators: List[str],
    extra_cov: Optional[Dict] = None,
    extra_alloc: Optional[Dict] = None,
) -> List[dict]:
    """Run one (label, estimator x allocator) grid and flatten to rows."""
    cfg = BacktestConfig(
        window=window, frequency=frequency, cost_bps=cost_bps,
        cost_model=cost_model, max_weight=max_weight,
        cov_kwargs={"sigma_method": sigma_method, **(extra_cov or {})},
    )
    try:
        results = run_matrix(rets, cov_methods=estimators,
                             allocators=allocators, config=cfg,
                             extra_alloc_kwargs=extra_alloc, verbose=False)
    except Exception as exc:                      # pragma: no cover
        LOGGER.warning("cell failed (%s=%s): %s", label_key, label_value, exc)
        return []

    rows = []
    for key, res in results.items():
        m = res.metrics
        cm = m.extra.get("cov_method", "")
        al = m.extra.get("allocator", "")
        rows.append({
            label_key: label_value,
            "universe": universe,
            "estimator": cm,
            "allocator": al,
            "window": window,
            "frequency": frequency,
            "cost_bps": cost_bps,
            "sigma_method": sigma_method,
            "sharpe": m.sharpe,
            "sharpe_net": m.sharpe_net,
            "ann_return_net": m.ann_return_net,
            "ann_volatility": m.ann_volatility,
            "max_drawdown": m.max_drawdown,
            "calmar": m.calmar,
            "avg_turnover": m.avg_turnover,
            "avg_enb": m.extra.get("avg_enb", np.nan),
            "avg_dr": m.extra.get("avg_dr", np.nan),
            # turnover-adjusted Sharpe: penalise the Sharpe by the fraction of
            # annual return paid away in costs.  A strategy that "wins" only by
            # churning is exposed here.
            "sharpe_per_turnover": (m.sharpe_net / m.avg_turnover
                                    if m.avg_turnover and m.avg_turnover > 0
                                    else np.nan),
        })
    return rows


# ---------------------------------------------------------------------------
# Individual sweeps
# ---------------------------------------------------------------------------


def sweep_window(rets, universe, cfg, estimators, allocators, mw) -> pd.DataFrame:
    rows = []
    for T in list(cfg.robustness.windows):
        if T >= rets.shape[0] - 40:
            LOGGER.info("  skipping window=%d (panel too short)", T)
            continue
        LOGGER.info("  window T=%d  (q=%.3f)", T, rets.shape[1] / T)
        rows += _run_cell(rets, universe, "window", int(T), int(T),
                          str(cfg.backtest.frequency), float(cfg.backtest.cost_bps),
                          str(cfg.backtest.cost_model), mw,
                          str(cfg.rmt.sigma_method), estimators, allocators)
    return pd.DataFrame(rows)


def sweep_frequency(rets, universe, cfg, estimators, allocators, mw) -> pd.DataFrame:
    rows = []
    for f in list(cfg.robustness.frequencies):
        LOGGER.info("  frequency=%s", f)
        rows += _run_cell(rets, universe, "frequency", str(f),
                          int(cfg.backtest.window), str(f),
                          float(cfg.backtest.cost_bps),
                          str(cfg.backtest.cost_model), mw,
                          str(cfg.rmt.sigma_method), estimators, allocators)
    return pd.DataFrame(rows)


def sweep_threshold(rets, universe, cfg, estimators, allocators,
                    mw) -> pd.DataFrame:
    """Vary the multiplier applied to the fitted MP edge lambda_+.

    Only the hard- and soft-threshold estimators depend on this parameter, so
    we restrict the grid to them (plus the sample benchmark for reference).
    """
    rows = []
    ests = [e for e in estimators if e in ("sample", "rmt_hard", "rmt_soft")]
    ests = ests or ["sample", "rmt_hard", "rmt_soft"]
    for mult in list(cfg.robustness.lambda_plus_multipliers):
        LOGGER.info("  lambda_plus multiplier=%.2f", mult)
        rows += _run_cell(rets, universe, "lambda_plus_mult", float(mult),
                          int(cfg.backtest.window), str(cfg.backtest.frequency),
                          float(cfg.backtest.cost_bps),
                          str(cfg.backtest.cost_model), mw,
                          str(cfg.rmt.sigma_method),
                          ests, allocators,
                          extra_cov={"lambda_plus_scale": float(mult)})
    return pd.DataFrame(rows)


def sweep_cost(rets, universe, cfg, estimators, allocators, mw) -> pd.DataFrame:
    rows = []
    for c in list(cfg.robustness.cost_bps):
        LOGGER.info("  cost=%.1f bp", c)
        rows += _run_cell(rets, universe, "cost_bps", float(c),
                          int(cfg.backtest.window), str(cfg.backtest.frequency),
                          float(c), str(cfg.backtest.cost_model), mw,
                          str(cfg.rmt.sigma_method), estimators, allocators)
    return pd.DataFrame(rows)


def sweep_sigma(rets, universe, cfg, estimators, allocators, mw) -> pd.DataFrame:
    rows = []
    ests = [e for e in estimators
            if e in ("sample", "ledoit_wolf", "rmt_hard", "rmt_soft", "rmt_rie")]
    for sm in list(cfg.robustness.sigma_methods):
        LOGGER.info("  sigma_method=%s", sm)
        rows += _run_cell(rets, universe, "sigma_method", str(sm),
                          int(cfg.backtest.window), str(cfg.backtest.frequency),
                          float(cfg.backtest.cost_bps),
                          str(cfg.backtest.cost_model), mw,
                          str(sm), ests, allocators)
    return pd.DataFrame(rows)


def sweep_universe(cfg, estimators, allocators, mw,
                   windows: Dict[str, int]) -> pd.DataFrame:
    rows = []
    for uni in ("us_sector_etf", "cn_sector_etf", "us_single_stock"):
        T = windows.get(uni, int(cfg.backtest.window))
        LOGGER.info("  universe=%s (T=%d)", uni, T)
        rets, _ = load_returns(
            uni, start=str(cfg.data.start), end=str(cfg.data.end),
            method="simple", min_coverage=float(cfg.data.min_coverage),
            repair_splits=bool(cfg.data.repair_splits),
            align=str(cfg.data.align), winsorise=cfg.data.winsorise,
            verbose=False)
        if rets.shape[0] <= T + 40:
            LOGGER.info("    skipping %s (only %d rows)", uni, rets.shape[0])
            continue
        rows += _run_cell(rets, uni, "universe", uni, T,
                          str(cfg.backtest.frequency),
                          float(cfg.backtest.cost_bps),
                          str(cfg.backtest.cost_model), mw,
                          str(cfg.rmt.sigma_method), estimators, allocators)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def pivot_summary(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Pivot a sweep table to (key x estimator) mean Sharpe_net and QLIKE-proxy."""
    if df.empty:
        return df
    g = df.groupby([key, "estimator"]).agg(
        sharpe_net=("sharpe_net", "mean"),
        sharpe=("sharpe", "mean"),
        ann_volatility=("ann_volatility", "mean"),
        avg_turnover=("avg_turnover", "mean"),
        avg_enb=("avg_enb", "mean"),
    ).reset_index()
    return g


def relative_to_sample(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Delta Sharpe_net of every estimator vs ``sample`` within each key value."""
    if df.empty:
        return df
    g = df.groupby([key, "estimator"])["sharpe_net"].mean().unstack("estimator")
    if "sample" not in g.columns:
        return g
    return g.sub(g["sample"], axis=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sweep", default="all",
                    choices=["all", *SWEEPS])
    ap.add_argument("--universe", default=None)
    ap.add_argument("--estimators", nargs="*", default=None)
    ap.add_argument("--allocators", nargs="*", default=None)
    ap.add_argument("--max-weight", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(int(cfg.project.seed))
    universe = args.universe or str(cfg.data.universe)

    if args.max_weight is None:
        mw_cfg = cfg.backtest.max_weight
        mw = float(mw_cfg) if mw_cfg else None
    elif str(args.max_weight).lower() in ("none", "null", ""):
        mw = None
    else:
        mw = float(args.max_weight)

    estimators = [e for e in (args.estimators or list(cfg.estimators))
                  if e in ESTIMATORS]
    allocators = [a for a in (args.allocators or list(cfg.allocators))
                  if a in ALLOCATORS]

    out_dir = ensure_dir(PROJECT_ROOT / "results" / "robustness")
    LOGGER.info("=" * 78)
    LOGGER.info("ROBUSTNESS SWEEPS | sweep=%s universe=%s max_weight=%s",
                args.sweep, universe, mw)
    LOGGER.info("estimators: %s", ", ".join(estimators))
    LOGGER.info("allocators: %s", ", ".join(allocators))
    LOGGER.info("=" * 78)

    rets, _ = load_returns(
        universe, start=str(cfg.data.start), end=str(cfg.data.end),
        method="simple", min_coverage=float(cfg.data.min_coverage),
        repair_splits=bool(cfg.data.repair_splits),
        align=str(cfg.data.align), winsorise=cfg.data.winsorise, verbose=False)
    LOGGER.info("panel: %d dates x %d assets", rets.shape[0], rets.shape[1])

    todo = list(SWEEPS) if args.sweep == "all" else [args.sweep]
    # windows per universe for the cross-market sweep
    uni_windows = {"us_sector_etf": 252, "cn_sector_etf": 252,
                   "us_single_stock": 120}

    wanted = {
        "window": lambda: sweep_window(rets, universe, cfg, estimators,
                                       allocators, mw),
        "frequency": lambda: sweep_frequency(rets, universe, cfg, estimators,
                                             allocators, mw),
        "threshold": lambda: sweep_threshold(rets, universe, cfg, estimators,
                                             allocators, mw),
        "cost": lambda: sweep_cost(rets, universe, cfg, estimators,
                                   allocators, mw),
        "sigma": lambda: sweep_sigma(rets, universe, cfg, estimators,
                                     allocators, mw),
        "universe": lambda: sweep_universe(cfg, estimators, allocators, mw,
                                           uni_windows),
    }

    manifest = {"sweep": args.sweep, "universe": universe,
                "max_weight": mw, "sweeps_completed": []}

    for name in todo:
        LOGGER.info("-" * 78)
        LOGGER.info("SWEEP: %s", name)
        LOGGER.info("-" * 78)
        with Timer(f"sweep {name}", LOGGER):
            df = wanted[name]()
        if df.empty:
            LOGGER.warning("  sweep %s produced no rows", name)
            continue

        save_table(df, out_dir / f"{name}_{universe}.csv", index=False)
        # The swept dimension's *column* name does not always equal the sweep
        # name: the sigma sweep varies the "sigma_method" column (``sigma`` is
        # only the shorthand for the sweep itself), and the threshold sweep
        # labels rows with the multiplier rather than the raw threshold.
        key = SWEEP_KEY[name]
        piv = pivot_summary(df, key)
        save_table(piv, out_dir / f"{name}_{universe}_pivot.csv", index=False)
        rel = relative_to_sample(df, key)
        if not rel.empty:
            save_table(rel, out_dir / f"{name}_{universe}_vs_sample.csv")

        LOGGER.info("mean Sharpe_net by %s x estimator:", key)
        LOGGER.info("\n%s", piv.pivot(index=key, columns="estimator",
                                      values="sharpe_net").round(4).to_string())
        manifest["sweeps_completed"].append(name)

    save_json(manifest, out_dir / f"manifest_{args.sweep}_{universe}.json")
    LOGGER.info("robustness outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
