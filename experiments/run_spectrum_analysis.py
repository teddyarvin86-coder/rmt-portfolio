#!/usr/bin/env python
"""Experiment 1 — Eigenvalue spectrum analysis and the Marchenko-Pastur law.

This script produces the empirical evidence that the RMT premise holds for
the chosen universe:

1. Fit the MP law to the sample correlation matrix over a representative
   estimation window and report ``q``, the fitted noise scale ``sigma`` and
   the noise band edges.
2. Compute the empirical eigenvalue density and overlay the MP prediction,
   quantifying how many eigenvalues fall outside the noise band and how much
   total variance those "signal" modes carry.
3. Repeat over a grid of ``N/T`` ratios (by varying the window length ``T``)
   to show the noise band widening as ``q`` grows — the mechanism through
   which denoising matters more in higher dimensions.
4. Report the stability of the eigenvalues over rolling windows: a genuine
   factor has a persistent eigenvalue, while a noise mode's eigenvalue
   fluctuates and its eigenvector rotates.

Outputs
-------
``results/spectrum/<universe>_spectrum.csv``
    Per-eigenvalue detail for the headline window.
``results/spectrum/<universe>_nt_grid.csv``
    Noise-band statistics across window lengths.
``results/spectrum/<universe>_stability.csv``
    Eigenvalue stability across rolling windows.
``figures/spectrum_*.png``
    The spectrum figures (drawn by ``experiments.make_figures``).

Usage
-----
    python -m experiments.run_spectrum_analysis --universe us_sector_etf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rmt_portfolio.config import PROJECT_ROOT, load_config
from rmt_portfolio.data import load_returns
from rmt_portfolio.rmt import (
    MPLaw,
    eigen_decompose,
    fit_mp_sigma,
    mp_density,
    mp_edge,
    sample_cov,
)
from rmt_portfolio.utils import ensure_dir, get_logger, save_json, save_table, \
    set_global_seed, Timer

LOGGER = get_logger("experiments.spectrum")


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------


def analyse_window(
    returns: np.ndarray,
    sigma_method: str = "median_bulk",
    trim: float = 0.2,
) -> Dict[str, object]:
    """Analyse one estimation window: MP fit + spectrum summary.

    Returns a dictionary with the fitted :class:`MPLaw`, the eigenvalues
    (ascending and descending), the implied MP density on a grid, and the
    headline statistics used in the paper's spectrum table.
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    cov = sample_cov(x)
    dec = eigen_decompose(cov, t)

    # Decompose the *correlation* matrix as well: this is the standard object
    # in the RMT literature, because it removes the (uninteresting) dispersion
    # of individual volatilities and leaves only the correlation structure.
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    dec_corr = eigen_decompose(corr, t)

    law = dec_corr.mp_law(sigma_method=sigma_method, trim=trim)

    ev = dec_corr.eigenvalues
    noise_mask = ev < law.lambda_plus
    n_signal = int((~noise_mask).sum())
    variance_share = float(ev[~noise_mask].sum() / ev.sum()) \
        if ev.sum() > 0 else np.nan

    # MP density on a grid spanning the observed spectrum
    grid = np.linspace(0.0, max(ev[-1] * 1.05, law.lambda_plus * 1.2), 800)
    density = mp_density(grid, law.sigma, law.q)

    # Compare the empirical bulk to the MP prediction with a KS statistic on
    # the eigenvalues identified as noise (the honest comparison: the MP law
    # describes the noise, not the factors).
    noise_ev = np.sort(ev[noise_mask])
    ks_stat = np.nan
    if noise_ev.size >= 5:
        emp_cdf = (np.arange(1, noise_ev.size + 1) - 0.5) / noise_ev.size
        from rmt_portfolio.rmt.mp_law import _mp_cdf
        theo_cdf = _mp_cdf(noise_ev, law.sigma, law.q)
        # renormalise the theoretical CDF to end at 1 on the noise block
        if theo_cdf[-1] > 0:
            theo_cdf = theo_cdf / theo_cdf[-1]
        ks_stat = float(np.max(np.abs(emp_cdf - theo_cdf)))

    return {
        "law": law,
        "dec": dec,
        "dec_corr": dec_corr,
        "eigenvalues": ev,
        "eigenvalues_desc": ev[::-1],
        "n_assets": n,
        "n_obs": t,
        "q": n / t,
        "n_signal": n_signal,
        "n_noise": int(noise_mask.sum()),
        "noise_fraction": float(noise_mask.sum() / ev.size),
        "signal_variance_share": variance_share,
        "noise_variance_share": 1.0 - variance_share if np.isfinite(
            variance_share) else np.nan,
        "lambda_max": float(ev[-1]),
        "lambda_min": float(ev[0]),
        "mean_eigenvalue": float(ev.mean()),
        "effective_rank": dec_corr.effective_rank(),
        "condition_number": dec_corr.condition_number,
        "top5_share": dec_corr.top_k_share(5),
        "top10_share": dec_corr.top_k_share(10),
        "ks_statistic_noise": ks_stat,
        "density_grid": grid,
        "density": density,
    }


def spectrum_table(res: Dict[str, object], top_k: int = 10) -> pd.DataFrame:
    """Per-eigenvalue detail table for the headline window."""
    ev = np.asarray(res["eigenvalues"], dtype=float)
    law: MPLaw = res["law"]          # type: ignore[assignment]
    total = ev.sum()
    order = np.argsort(ev)[::-1]
    rows = []
    for rank, i in enumerate(order[:top_k], start=1):
        rows.append({
            "rank": rank,
            "eigenvalue": float(ev[i]),
            "share_of_trace": float(ev[i] / total) if total else np.nan,
            "cumulative_share": float(np.sum(ev[order[:rank]]) / total)
                                if total else np.nan,
            "above_lambda_plus": bool(ev[i] >= law.lambda_plus),
            "ratio_to_lambda_plus": float(ev[i] / law.lambda_plus)
                                    if law.lambda_plus else np.nan,
        })
    return pd.DataFrame(rows).set_index("rank")


def analyse_nt_grid(
    returns: pd.DataFrame,
    windows: List[int],
    sigma_method: str = "median_bulk",
    trim: float = 0.2,
    step: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate the MP fit across a grid of window lengths (hence ``q``).

    For each window length we take the *most recent* full window and also the
    average across all rolling windows of that length, because a single window
    can be unrepresentative during crisis periods.
    """
    R = np.asarray(returns.values, dtype=float)
    n_obs_total, n = R.shape
    rows = []
    for w in windows:
        if w > n_obs_total:
            LOGGER.warning("window %d exceeds sample %d; skipped", w, n_obs_total)
            continue
        step_w = step or max(w // 4, 20)
        starts = list(range(0, n_obs_total - w + 1, step_w))
        if not starts:
            starts = [0]

        recs = []
        for s in starts:
            win = R[s:s + w]
            cov = sample_cov(win)
            sd = np.sqrt(np.diag(cov))
            corr = cov / np.outer(sd, sd)
            np.fill_diagonal(corr, 1.0)
            dec = eigen_decompose(corr, w)
            try:
                law = dec.mp_law(sigma_method=sigma_method, trim=trim)
            except Exception:
                continue
            ev = dec.eigenvalues
            noise = ev < law.lambda_plus
            recs.append({
                "sigma": law.sigma,
                "lambda_minus": law.lambda_minus,
                "lambda_plus": law.lambda_plus,
                "n_signal": int((~noise).sum()),
                "signal_variance_share": float(ev[~noise].sum() / ev.sum())
                    if ev.sum() > 0 else np.nan,
                "lambda_max": float(ev[-1]),
                "mean_noise_ev": float(ev[noise].mean()) if noise.any()
                    else np.nan,
                "effective_rank": dec.effective_rank(),
            })
        if not recs:
            continue
        df = pd.DataFrame(recs)
        row = {
            "window": w,
            "q": n / w,
            "n_assets": n,
            "n_windows": len(df),
        }
        for col in df.columns:
            row[f"{col}_mean"] = float(df[col].mean())
            row[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
            row[f"{col}_last"] = float(df[col].iloc[-1])
        rows.append(row)
    return pd.DataFrame(rows).set_index("window")


def eigenvalue_stability(
    returns: pd.DataFrame,
    window: int = 252,
    step: int = 21,
    n_track: int = 10,
    sigma_method: str = "median_bulk",
) -> pd.DataFrame:
    """Track the largest ``n_track`` eigenvalues over rolling windows.

    A genuine systematic factor produces an eigenvalue that stays large and
    stable, and whose eigenvector is persistent.  A noise mode has an
    eigenvalue that drifts within the MP band and an eigenvector that rotates
    from window to window.  We measure both:

    * the coefficient of variation of each ranked eigenvalue, and
    * the absolute overlap :math:`|\\langle v_i^{(t)}, v_i^{(t-1)}\\rangle|`
      between consecutive windows' eigenvectors at the same rank.
    """
    R = np.asarray(returns.values, dtype=float)
    n_obs, n = R.shape
    ev_hist: List[np.ndarray] = []
    vec_hist: List[np.ndarray] = []
    dates: List[pd.Timestamp] = []

    for s in range(0, n_obs - window + 1, step):
        win = R[s:s + window]
        cov = sample_cov(win)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        np.fill_diagonal(corr, 1.0)
        dec = eigen_decompose(corr, window)
        ev_desc = dec.eigenvalues[::-1]
        vec_desc = dec.eigenvectors[:, ::-1]
        ev_hist.append(ev_desc)
        vec_hist.append(vec_desc)
        dates.append(returns.index[s + window - 1])

    if len(ev_hist) < 2:
        return pd.DataFrame()

    ev_arr = np.vstack([e[:n_track] for e in ev_hist])
    # eigenvector overlap at the same rank between consecutive windows
    overlaps = np.zeros((len(vec_hist) - 1, n_track))
    for k in range(len(vec_hist) - 1):
        a = vec_hist[k][:, :n_track]
        b = vec_hist[k + 1][:, :n_track]
        for r in range(n_track):
            ov = abs(float(np.dot(a[:, r], b[:, r])))
            # sign ambiguity: eigenvectors are defined up to a sign
            overlaps[k, r] = min(abs(ov), 1.0)

    rows = []
    for r in range(n_track):
        series = ev_arr[:, r]
        rows.append({
            "rank": r + 1,
            "eigenvalue_mean": float(series.mean()),
            "eigenvalue_std": float(series.std(ddof=1)),
            "eigenvalue_cv": float(series.std(ddof=1) / series.mean())
                             if series.mean() != 0 else np.nan,
            "eigenvalue_min": float(series.min()),
            "eigenvalue_max": float(series.max()),
            "vector_overlap_mean": float(overlaps[:, r].mean()),
            "vector_overlap_min": float(overlaps[:, r].min()),
            "n_windows": len(ev_hist),
        })
    df = pd.DataFrame(rows).set_index("rank")
    df.attrs["dates"] = dates
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to config.yaml")
    ap.add_argument("--universe", default=None,
                    help="override data.universe")
    ap.add_argument("--window", type=int, default=None,
                    help="override the headline estimation window")
    ap.add_argument("--sigma-method", default=None,
                    help="override rmt.sigma_method")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(int(cfg.project.seed))
    if args.universe:
        cfg.data.universe = args.universe
    window = int(args.window or cfg.backtest.window)
    sigma_method = args.sigma_method or str(cfg.rmt.sigma_method)
    universe = str(cfg.data.universe)

    out_dir = ensure_dir(PROJECT_ROOT / "results" / "spectrum")
    LOGGER.info("=" * 70)
    LOGGER.info("Spectrum analysis | universe=%s window=%d sigma=%s",
                universe, window, sigma_method)
    LOGGER.info("=" * 70)

    with Timer("data load", LOGGER):
        rets, quality = load_returns(
            universe, start=str(cfg.data.start), end=str(cfg.data.end),
            method=str(cfg.data.return_type),
            min_coverage=float(cfg.data.min_coverage),
            repair_splits=bool(cfg.data.repair_splits),
            align=str(cfg.data.align),
            winsorise=cfg.data.winsorise,
            verbose=False,
        )
    LOGGER.info("panel: %d dates x %d assets  (%s .. %s)",
                rets.shape[0], rets.shape[1],
                rets.index.min().date(), rets.index.max().date())
    for note in quality.notes:
        LOGGER.info("data note: %s", note)

    if rets.shape[0] < window:
        LOGGER.error("sample (%d) shorter than window (%d)",
                     rets.shape[0], window)
        return 1

    # ---- 1. headline window: the most recent full window ---------------
    head = rets.iloc[-window:]
    with Timer("headline window analysis", LOGGER):
        res = analyse_window(head.values, sigma_method=sigma_method,
                             trim=float(cfg.rmt.trim))
    law: MPLaw = res["law"]      # type: ignore[assignment]
    LOGGER.info("-" * 70)
    LOGGER.info("HEADLINE WINDOW: %s .. %s",
                head.index.min().date(), head.index.max().date())
    LOGGER.info("q = N/T = %d/%d = %.4f", res["n_assets"], res["n_obs"],
                res["q"])
    LOGGER.info("fitted sigma = %.4f  (%s)", law.sigma, sigma_method)
    LOGGER.info("noise band = [%.4f, %.4f]  (sigma^2=1 units)",
                law.lambda_minus, law.lambda_plus)
    LOGGER.info("empirical lambda_max = %.4f  (%.2fx the noise edge)",
                res["lambda_max"],
                res["lambda_max"] / law.lambda_plus if law.lambda_plus else np.nan)
    LOGGER.info("eigenvalues above the noise edge: %d of %d (%.1f%%)",
                res["n_signal"], res["n_assets"],
                100.0 * res["n_signal"] / res["n_assets"])
    LOGGER.info("those %d modes carry %.2f%% of total variance",
                res["n_signal"], 100.0 * res["signal_variance_share"])
    LOGGER.info("KP-style KS distance on the noise block: %.4f",
                res["ks_statistic_noise"])
    LOGGER.info("effective rank = %.2f (vs N = %d)",
                res["effective_rank"], res["n_assets"])
    LOGGER.info("-" * 70)

    spec = spectrum_table(res, top_k=int(cfg.report.top_eigenvalues))
    save_table(spec, out_dir / f"{universe}_spectrum.csv")
    LOGGER.info("top-%d eigenvalues:\n%s",
                cfg.report.top_eigenvalues, spec.to_string())

    # ---- 2. N/T grid ---------------------------------------------------
    with Timer("N/T grid", LOGGER):
        nt = analyse_nt_grid(rets, windows=[int(w) for w in
                                            cfg.robustness.windows],
                             sigma_method=sigma_method,
                             trim=float(cfg.rmt.trim))
    save_table(nt, out_dir / f"{universe}_nt_grid.csv")
    LOGGER.info("N/T grid (mean across rolling windows):")
    show_cols = [c for c in ["q", "sigma_mean", "lambda_plus_mean",
                             "n_signal_mean", "signal_variance_share_mean",
                             "effective_rank_mean"] if c in nt.columns]
    LOGGER.info("\n%s", nt[show_cols].round(4).to_string())

    # ---- 3. eigenvalue stability ---------------------------------------
    with Timer("eigenvalue stability", LOGGER):
        stab = eigenvalue_stability(rets, window=window, step=max(window // 12, 10),
                                    n_track=int(cfg.report.top_eigenvalues),
                                    sigma_method=sigma_method)
    if not stab.empty:
        save_table(stab, out_dir / f"{universe}_stability.csv")
        LOGGER.info("eigenvalue stability (rank 1 = market mode):")
        LOGGER.info("\n%s", stab.round(4).to_string())

    # ---- 4. persist the headline summary as JSON -----------------------
    summary = {
        "universe": universe,
        "window": window,
        "sigma_method": sigma_method,
        "window_start": str(head.index.min().date()),
        "window_end": str(head.index.max().date()),
        "n_assets": int(res["n_assets"]),
        "n_obs": int(res["n_obs"]),
        "q": float(res["q"]),
        "sigma": float(law.sigma),
        "lambda_minus": float(law.lambda_minus),
        "lambda_plus": float(law.lambda_plus),
        "lambda_max": float(res["lambda_max"]),
        "n_signal": int(res["n_signal"]),
        "n_noise": int(res["n_noise"]),
        "signal_variance_share": float(res["signal_variance_share"]),
        "noise_variance_share": float(res["noise_variance_share"]),
        "effective_rank": float(res["effective_rank"]),
        "condition_number": float(res["condition_number"]),
        "top5_share": float(res["top5_share"]),
        "top10_share": float(res["top10_share"]),
        "ks_statistic_noise": (float(res["ks_statistic_noise"])
                               if np.isfinite(res["ks_statistic_noise"])
                               else None),
    }
    save_json(summary, out_dir / f"{universe}_summary.json")

    # cache the arrays needed by the figure script
    np.savez_compressed(
        out_dir / f"{universe}_spectrum_arrays.npz",
        eigenvalues=np.asarray(res["eigenvalues"], dtype=float),
        density_grid=np.asarray(res["density_grid"], dtype=float),
        density=np.asarray(res["density"], dtype=float),
        lambda_minus=np.array([law.lambda_minus]),
        lambda_plus=np.array([law.lambda_plus]),
        sigma=np.array([law.sigma]),
        q=np.array([law.q]),
    )
    LOGGER.info("spectrum outputs written to %s", out_dir)
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
