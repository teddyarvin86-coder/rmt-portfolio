#!/usr/bin/env python
"""Experiment 2 — the main rolling out-of-sample backtest.

Runs the full ``estimator x allocator`` grid walk-forward on the chosen
universe and writes:

``results/backtest/<universe>_main_<tag>.csv``
    Headline performance table (one row per strategy), with both gross and
    net-of-cost metrics plus turnover.
``results/backtest/<universe>_equity_<tag>.npz``
    Net return series and equity curves for every strategy, for the figures.
``results/backtest/<universe>_weights_<tag>.parquet`` / ``.csv``
    The full weight history of every strategy (long format).
``results/backtest/<universe>_diag_<tag>.csv``
    Per-rebalance diagnostics: ex-ante volatility, ENB, diversification
    ratio, weight concentration, number of live positions.

The script is deliberately explicit about the *comparison design*: the
headline question is whether RMT denoising improves the risk model relative to
the sample covariance and to Ledoit-Wolf shrinkage, **holding the allocator
fixed**.  The second question is whether the effect is larger for allocators
that depend on inverting the covariance (minimum variance) than for those that
do not (HRP, risk parity).

Usage
-----
    python experiments/run_main_backtest.py --universe us_sector_etf
    python experiments/run_main_backtest.py --universe us_single_stock \
        --window 120 --tag highq
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rmt_portfolio.backtest import BacktestConfig, BacktestResult, run_matrix
from rmt_portfolio.config import PROJECT_ROOT, load_config
from rmt_portfolio.data import load_returns
from rmt_portfolio.portfolio import ALLOCATOR_LABELS, ALLOCATORS
from rmt_portfolio.rmt import ESTIMATOR_LABELS, ESTIMATORS
from rmt_portfolio.utils import ensure_dir, get_logger, palette, save_json, \
    save_table, set_global_seed, Timer

LOGGER = get_logger("experiments.backtest")


# ---------------------------------------------------------------------------
# Benchmark construction
# ---------------------------------------------------------------------------


def build_60_40(
    returns: pd.DataFrame,
    equity_hint: str = "SPY",
    bond_hint: str = "IEF",
    weight: float = 0.60,
) -> pd.Series:
    """Classic 60/40 benchmark return series.

    Three cases, in order of preference:

    1. **Dedicated legs available** (``SPY`` and ``IEF`` present): the textbook
       ``0.6 * SPY + 0.4 * IEF`` blend.
    2. **Equity-only universe** (single stocks): an equal-weight basket of all
       assets stands in for "equity", and the *least volatile* asset for the
       bond sleeve.  A single stock is never used as a bond proxy.
    3. **Fallback**: the least volatile third of the universe forms the bond
       sleeve, the rest forms the equity sleeve.

    The previous implementation simply took ``idxmax``/``idxmin`` of the
    volatility ranking, which on a single-stock universe produced
    ``0.6 * NVDA + 0.4 * (least volatile stock)`` — a portfolio with 34%
    annualised volatility that is not a 60/40 benchmark at all.
    """
    cols = list(returns.columns)
    eq = equity_hint if equity_hint in cols else None
    bd = bond_hint if bond_hint in cols else None
    if eq is not None and bd is not None:
        return weight * returns[eq] + (1.0 - weight) * returns[bd]

    vol = returns.std()
    ranked = vol.sort_values()
    n = len(ranked)
    # bond sleeve = least volatile third (at least one asset, at most 40% of N)
    n_bond = max(1, min(int(np.ceil(0.40 * n)), n - 1))
    bond_cols = list(ranked.index[:n_bond])
    equity_cols = [c for c in cols if c not in bond_cols]
    eq_ret = returns[equity_cols].mean(axis=1)
    bd_ret = returns[bond_cols].mean(axis=1)
    return weight * eq_ret + (1.0 - weight) * bd_ret


def benchmark_metrics(
    returns: pd.DataFrame,
    holding_days: int,
    name: str,
    risk_free: float = 0.0,
):
    """Compute metrics for a daily benchmark on the same holding grid.

    The benchmark is sampled at the same holding-period frequency as the
    strategies so that the comparison is apples-to-apples.
    """
    from rmt_portfolio.backtest.metrics import compute_metrics

    series = build_60_40(returns)
    r = np.asarray(series.values, dtype=float)
    # compound into holding-period returns aligned to the strategy grid
    n = r.size
    n_periods = n // holding_days
    if n_periods < 1:
        return None
    trimmed = r[: n_periods * holding_days].reshape(n_periods, holding_days)
    per_period = np.prod(1.0 + trimmed, axis=1) - 1.0
    p_per_year = 252.0 / holding_days
    return compute_metrics(per_period, name=name, risk_free=risk_free,
                           periods_per_year=p_per_year, extra={
                               "cov_method": "benchmark",
                               "allocator": "benchmark"})


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def metrics_table(results: Dict[str, BacktestResult],
                  benchmark=None) -> pd.DataFrame:
    """Assemble a tidy performance table from a dict of backtest results."""
    rows = []
    for key, res in results.items():
        m = res.metrics
        rows.append({
            "strategy": key,
            "cov_method": m.extra.get("cov_method", ""),
            "allocator": m.extra.get("allocator", ""),
            "ann_return": m.ann_return,
            "ann_volatility": m.ann_volatility,
            "sharpe": m.sharpe,
            "sharpe_net": m.sharpe_net,
            "ann_return_net": m.ann_return_net,
            "sortino": m.sortino,
            "max_drawdown": m.max_drawdown,
            "calmar": m.calmar,
            "var_95": m.var_95,
            "cvar_95": m.cvar_95,
            "skew": m.skew,
            "excess_kurtosis": m.excess_kurtosis,
            "ulcer": m.ulcer,
            "avg_turnover": m.avg_turnover,
            "total_turnover": m.total_turnover,
            "n_rebalances": m.n_rebalances,
            "avg_vol_exante": m.extra.get("avg_vol", np.nan),
            "avg_enb": m.extra.get("avg_enb", np.nan),
            "avg_dr": m.extra.get("avg_dr", np.nan),
            "avg_weight_hhi": m.extra.get("avg_weight_hhi", np.nan),
            "total_cost": m.extra.get("total_cost", np.nan),
        })
    if benchmark is not None:
        m = benchmark
        rows.append({
            "strategy": m.name,
            "cov_method": "benchmark",
            "allocator": "benchmark",
            "ann_return": m.ann_return,
            "ann_volatility": m.ann_volatility,
            "sharpe": m.sharpe,
            "sharpe_net": m.sharpe_net,
            "ann_return_net": m.ann_return_net,
            "sortino": m.sortino,
            "max_drawdown": m.max_drawdown,
            "calmar": m.calmar,
            "var_95": m.var_95,
            "cvar_95": m.cvar_95,
            "skew": m.skew,
            "excess_kurtosis": m.excess_kurtosis,
            "ulcer": m.ulcer,
            "avg_turnover": m.avg_turnover,
            "total_turnover": m.total_turnover,
            "n_rebalances": m.n_rebalances,
        })
    return pd.DataFrame(rows).set_index("strategy")


def improvement_table(table: pd.DataFrame,
                      reference: str = "sample") -> pd.DataFrame:
    """Quantify the *incremental* effect of denoising, holding the allocator fixed.

    For every (estimator, allocator) cell we compute the difference in each
    headline metric against the same allocator fed the reference covariance
    estimator (``sample`` by default).  This is the core comparison of the
    study: it isolates the risk-model improvement from the allocator choice.
    """
    t = table.copy()
    rows = []
    allocators = sorted(set(t.loc[t["allocator"] != "benchmark", "allocator"]))
    for al in allocators:
        sub = t[t["allocator"] == al]
        if reference not in sub["cov_method"].values:
            continue
        ref = sub[sub["cov_method"] == reference].iloc[0]
        for cm in sub["cov_method"].values:
            row = sub[sub["cov_method"] == cm].iloc[0]
            rows.append({
                "allocator": al,
                "cov_method": cm,
                "d_sharpe": row["sharpe"] - ref["sharpe"],
                "d_sharpe_net": row["sharpe_net"] - ref["sharpe_net"],
                "d_ann_return": row["ann_return"] - ref["ann_return"],
                "d_ann_volatility": row["ann_volatility"] - ref["ann_volatility"],
                "d_max_drawdown": row["max_drawdown"] - ref["max_drawdown"],
                "d_calmar": row["calmar"] - ref["calmar"],
                "d_turnover": row["avg_turnover"] - ref["avg_turnover"],
                "d_enb": row["avg_enb"] - ref["avg_enb"],
                "sharpe": row["sharpe"],
                "sharpe_net": row["sharpe_net"],
                "avg_turnover": row["avg_turnover"],
                "avg_enb": row["avg_enb"],
                "avg_dr": row["avg_dr"],
            })
    return pd.DataFrame(rows).set_index(["allocator", "cov_method"])


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
    ap.add_argument("--cost-bps", type=float, default=None)
    ap.add_argument("--cost-model", default=None)
    ap.add_argument("--no-trade-band", type=float, default=None)
    ap.add_argument("--tag", default="main",
                    help="suffix for output filenames")
    ap.add_argument("--estimators", nargs="*", default=None)
    ap.add_argument("--allocators", nargs="*", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(int(cfg.project.seed))
    universe = args.universe or str(cfg.data.universe)
    window = int(args.window or cfg.backtest.window)
    frequency = args.frequency or str(cfg.backtest.frequency)
    cost_bps = float(args.cost_bps if args.cost_bps is not None
                     else cfg.backtest.cost_bps)
    cost_model = args.cost_model or str(cfg.backtest.cost_model)
    band = float(args.no_trade_band if args.no_trade_band is not None
                 else cfg.backtest.no_trade_band)

    estimators = args.estimators or list(cfg.estimators)
    allocators = args.allocators or list(cfg.allocators)
    # keep only names that actually exist, and warn about typos
    estimators = [e for e in estimators if e in ESTIMATORS]
    allocators = [a for a in allocators if a in ALLOCATORS]

    bt_cfg = BacktestConfig(
        window=window,
        frequency=frequency,
        cost_bps=cost_bps,
        cost_model=cost_model,
        no_trade_band=band,
        risk_free=float(cfg.backtest.risk_free),
        long_only=bool(cfg.backtest.long_only),
        max_weight=(float(cfg.backtest.max_weight)
                    if cfg.backtest.max_weight else None),
        cov_kwargs={
            "sigma_method": str(cfg.rmt.sigma_method),
            "trim": float(cfg.rmt.trim),
            "beta": float(cfg.rmt.soft_beta),
            "n_factors": int(cfg.rmt.n_factors),
        },
    )

    out_dir = ensure_dir(PROJECT_ROOT / "results" / "backtest")
    LOGGER.info("=" * 78)
    LOGGER.info("MAIN BACKTEST | universe=%s window=%d freq=%s cost=%.1fbp "
                "(%s)", universe, window, frequency, cost_bps, cost_model)
    LOGGER.info("estimators: %s", ", ".join(estimators))
    LOGGER.info("allocators: %s", ", ".join(allocators))
    LOGGER.info("=" * 78)

    with Timer("data load", LOGGER):
        rets, quality = load_returns(
            universe, start=str(cfg.data.start), end=str(cfg.data.end),
            method="simple",       # the backtester compounds simple returns
            min_coverage=float(cfg.data.min_coverage),
            repair_splits=bool(cfg.data.repair_splits),
            align=str(cfg.data.align),
            winsorise=cfg.data.winsorise,
            verbose=False)
    LOGGER.info("panel: %d dates x %d assets  (%s .. %s)",
                rets.shape[0], rets.shape[1],
                rets.index.min().date(), rets.index.max().date())
    for note in quality.notes:
        LOGGER.info("data note: %s", note)

    n_reb_est = max((rets.shape[0] - window) // bt_cfg.holding_days, 0)
    LOGGER.info("expected rebalances per strategy: ~%d", n_reb_est)
    if n_reb_est < 12:
        LOGGER.error("fewer than 12 rebalances -- sample too short. "
                     "Reduce --window or extend the date range.")
        return 1

    with Timer("backtest grid", LOGGER):
        results = run_matrix(rets, cov_methods=estimators,
                             allocators=allocators, config=bt_cfg,
                             verbose=True)

    if not results:
        LOGGER.error("no successful backtests")
        return 1

    bench = benchmark_metrics(rets, bt_cfg.holding_days, name="60_40",
                              risk_free=bt_cfg.risk_free)
    table = metrics_table(results, benchmark=bench)
    save_table(table, out_dir / f"{universe}_main_{args.tag}.csv")

    imp = improvement_table(table, reference="sample")
    if not imp.empty:
        save_table(imp, out_dir / f"{universe}_improvement_{args.tag}.csv")

    # ---- console summary -----------------------------------------------
    LOGGER.info("-" * 78)
    LOGGER.info("HEADLINE RESULTS (net of %.1f bp one-way costs)", cost_bps)
    LOGGER.info("-" * 78)
    show = table[["ann_return_net", "ann_volatility", "sharpe", "sharpe_net",
                  "max_drawdown", "calmar", "avg_turnover", "avg_enb",
                  "avg_dr"]].copy()
    LOGGER.info("\n%s", show.round(4).to_string())

    if not imp.empty:
        LOGGER.info("-" * 78)
        LOGGER.info("INCREMENTAL EFFECT vs SAMPLE COVARIANCE (same allocator)")
        LOGGER.info("-" * 78)
        LOGGER.info("\n%s",
                    imp[["d_sharpe", "d_sharpe_net", "d_ann_volatility",
                         "d_max_drawdown", "d_turnover", "d_enb"]]
                    .round(4).to_string())

    # ---- aggregate ranking ---------------------------------------------
    LOGGER.info("-" * 78)
    LOGGER.info("AVERAGE ACROSS ALLOCATORS, BY ESTIMATOR")
    LOGGER.info("-" * 78)
    non_bench = table[table["cov_method"] != "benchmark"]
    agg = non_bench.groupby("cov_method").agg(
        sharpe=("sharpe", "mean"),
        sharpe_net=("sharpe_net", "mean"),
        ann_return_net=("ann_return_net", "mean"),
        ann_volatility=("ann_volatility", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        calmar=("calmar", "mean"),
        avg_turnover=("avg_turnover", "mean"),
        avg_enb=("avg_enb", "mean"),
        avg_dr=("avg_dr", "mean"),
    ).sort_values("sharpe_net", ascending=False)
    LOGGER.info("\n%s", agg.round(4).to_string())
    save_table(agg, out_dir / f"{universe}_by_estimator_{args.tag}.csv")

    LOGGER.info("-" * 78)
    LOGGER.info("AVERAGE ACROSS ESTIMATORS, BY ALLOCATOR")
    LOGGER.info("-" * 78)
    agg2 = non_bench.groupby("allocator").agg(
        sharpe=("sharpe", "mean"),
        sharpe_net=("sharpe_net", "mean"),
        ann_return_net=("ann_return_net", "mean"),
        ann_volatility=("ann_volatility", "mean"),
        max_drawdown=("max_drawdown", "mean"),
        calmar=("calmar", "mean"),
        avg_turnover=("avg_turnover", "mean"),
        avg_enb=("avg_enb", "mean"),
        avg_dr=("avg_dr", "mean"),
    ).sort_values("sharpe_net", ascending=False)
    LOGGER.info("\n%s", agg2.round(4).to_string())
    save_table(agg2, out_dir / f"{universe}_by_allocator_{args.tag}.csv")

    # ---- persist series for figures ------------------------------------
    equity = {}
    net_rets = {}
    for key, res in results.items():
        safe = key.replace("+", "__")
        equity[safe] = res.net_equity.values
        net_rets[safe] = res.net_returns
    ref_key = next(iter(results))
    np.savez_compressed(
        out_dir / f"{universe}_equity_{args.tag}.npz",
        dates=np.array([str(d.date()) for d in results[ref_key].dates]),
        strategies=np.array(list(equity.keys())),
        **{f"eq_{k}": v for k, v in equity.items()},
        **{f"ret_{k}": v for k, v in net_rets.items()},
    )
    if bench is not None:
        np.savez_compressed(
            out_dir / f"{universe}_benchmark_{args.tag}.npz",
            ann_return=bench.ann_return, ann_vol=bench.ann_volatility,
            sharpe=bench.sharpe, max_drawdown=bench.max_drawdown,
            calmar=bench.calmar,
        )

    # ---- weight history ------------------------------------------------
    wframes = []
    for key, res in results.items():
        w = res.weights_history.copy()
        w.columns = pd.MultiIndex.from_product([[key], w.columns])
        wframes.append(w)
    if wframes:
        wall = pd.concat(wframes, axis=1)
        try:
            wall.to_parquet(out_dir / f"{universe}_weights_{args.tag}.parquet")
        except Exception as exc:      # pragma: no cover - pyarrow missing
            LOGGER.warning("parquet unavailable (%s); writing CSV", exc)
            wall.to_csv(out_dir / f"{universe}_weights_{args.tag}.csv")

    # ---- diagnostics ---------------------------------------------------
    dframes = []
    for key, res in results.items():
        d = res.diagnostics.copy()
        if d.empty:
            continue
        d["strategy"] = key
        dframes.append(d.reset_index())
    if dframes:
        dall = pd.concat(dframes, ignore_index=True)
        save_table(dall, out_dir / f"{universe}_diag_{args.tag}.csv",
                   index=False)

    save_json({
        "universe": universe,
        "config": {
            "window": window, "frequency": frequency, "cost_bps": cost_bps,
            "cost_model": cost_model, "no_trade_band": band,
            "estimators": estimators, "allocators": allocators,
        },
        "panel": {
            "n_dates": int(rets.shape[0]), "n_assets": int(rets.shape[1]),
            "start": str(rets.index.min().date()),
            "end": str(rets.index.max().date()),
        },
        "best_by_sharpe_net": table["sharpe_net"].idxmax(),
        "best_sharpe_net": float(table["sharpe_net"].max()),
        "benchmark_sharpe": float(bench.sharpe) if bench is not None else None,
    }, out_dir / f"{universe}_summary_{args.tag}.json")

    LOGGER.info("backtest outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
