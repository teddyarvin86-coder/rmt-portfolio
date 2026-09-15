#!/usr/bin/env python
"""Experiment — generate every figure used in the working paper.

The figures are organised into four groups, each answering one question:

**A. The spectrum (does the RMT story hold in this data?)**

1. ``fig_spectrum_mp`` — the eigenvalue histogram for the base window against
   the fitted Marchenko-Pastur density, with :math:`\\lambda_\\pm` marked.
   The visual core of the whole project: most of the spectrum is inside the
   noise band; a handful of eigenvalues escape it and carry almost all of the
   variance.
2. ``fig_spectrum_mp_log`` — the same on a log scale, so the small
   eigenvalues (where the RMT model is tested hardest) are visible.
3. ``fig_spectrum_nt_grid`` — the spectrum for several window lengths, showing
   how :math:`\\lambda_+` widens as :math:`q = N/T` grows and how the number
   of eigenvalues outside the band shrinks.
4. ``fig_eigenvalue_stability`` — the overlap of the top eigenvectors between
   consecutive windows, as a function of rank.  Factors should be stable;
   noise directions should not.
5. ``fig_eigenvalue_growth`` — the ranked eigenvalue spectrum against the MP
   bulk, on a log scale, with the fitted ``sigma`` annotated.

**B. Denoising (what does it do to the matrix?)**

6. ``fig_eigen_shrinkage`` — sample eigenvalues against cleaned eigenvalues
   for every estimator, on a 45-degree reference line.  This single figure
   shows more clearly than any table that hard thresholding collapses the
   bulk, soft thresholding tapers it, and the RIE inflates the bulk while
   shrinking the top.
7. ``fig_condition_numbers`` — the condition number and effective rank of
   every estimator's matrix, over time.
8. ``fig_corr_heatmaps`` — sample vs denoised correlation matrices side by
   side, with the same colour scale.

**C. Backtest results (what does it do to portfolios?)**

9. ``fig_equity_curves`` — cumulative net wealth of the main strategies,
   grouped by allocator.
10. ``fig_equity_grid`` — a small-multiples grid: one panel per allocator,
    one line per estimator.
11. ``fig_drawdowns`` — the drawdown path of the key strategies.
12. ``fig_risk_return`` — realised volatility against realised return, with
    the Sharpe iso-lines, for every cell.
13. ``fig_sharpe_bars`` — a grouped bar chart of net Sharpe by estimator,
    grouped by allocator.
14. ``fig_turnover_sharpe`` — the turnover/Sharpe trade-off scatter, with the
    Pareto frontier drawn.

**D. Risk-model quality (the sharpest test)**

15. ``fig_forecast_accuracy`` — predicted vs realised portfolio volatility for
    each estimator under minimum variance, with the y = x line.
16. ``fig_forecast_qlike`` — a bar chart of the QLIKE loss, the only standard
    loss function that is robust to a noisy volatility proxy.
17. ``fig_forecast_bias_time`` — the rolling forecast bias, showing where in
    time each model fails.
18. ``fig_enb_distribution`` — the distribution of the effective number of
    bets over the sample, per estimator.
19. ``fig_risk_contribution`` — risk contributions under risk parity, the
    diagnostic that shows whether the allocator achieves its goal.

**E. Robustness**

20. ``fig_robustness_window`` — Sharpe against the window length ``T`` for
    every estimator (the ``q`` sweep).
21. ``fig_robustness_threshold`` — the threshold-multiplier sensitivity.
22. ``fig_robustness_cost`` — turnover and net Sharpe against the cost level.

Every function returns the path of the file it wrote, so ``main`` can build a
manifest and the paper can reference the figures by name.

Usage
-----
    python experiments/run_figures.py                 # everything
    python experiments/run_figures.py --group A       # only the spectra
    python experiments/run_figures.py --universe cn_sector_etf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rmt_portfolio.config import PROJECT_ROOT, load_config
from rmt_portfolio.data import load_returns
from rmt_portfolio.portfolio import effective_number_of_bets, \
    risk_contributions
from rmt_portfolio.rmt import ESTIMATOR_LABELS, ESTIMATORS, \
    eigen_decompose, estimate_covariance, sample_cov
from rmt_portfolio.rmt.mp_law import MPLaw, mp_density, mp_edge
from rmt_portfolio.utils import ALLOCATOR_STYLES, ESTIMATOR_COLORS, \
    apply_plot_style, ensure_dir, get_logger, palette, set_global_seed

LOGGER = get_logger("experiments.figures")

# Short display names keep the legends compact in the paper.
SHORT = {
    "sample": "Sample",
    "ledoit_wolf": "Ledoit-Wolf",
    "constant_corr": "LW (const-corr)",
    "rmt_hard": "RMT hard",
    "rmt_soft": "RMT soft",
    "rmt_rie": "RMT RIE",
    "factor_model": "Factor model",
}
SHORT_AL = {
    "equal_weight": "1/N",
    "inverse_variance": "Inverse vol",
    "minimum_variance": "Min variance",
    "risk_parity": "Risk parity",
    "hrp": "HRP",
    "max_diversification": "Max diversification",
}


def _plt():
    import matplotlib.pyplot as plt
    return plt


def _save(fig, path: Path, dpi: int = 200) -> Path:
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    _plt().close(fig)
    LOGGER.info("  wrote %s", path.name)
    return path


def _load_latest(universe: str, pattern: str) -> Optional[pd.DataFrame]:
    """Load the most recently written results CSV matching ``pattern``."""
    d = PROJECT_ROOT / "results"
    cands = sorted(d.rglob(pattern))
    if not cands:
        return None
    return pd.read_csv(cands[-1])


# ===========================================================================
# Group A — the spectrum
# ===========================================================================


def _corr_eigen(x: np.ndarray, window: int, sigma_method: str = "median_bulk"):
    """Correlation matrix, its eigendecomposition and the fitted MP law.

    All spectrum plots are drawn on the **correlation** scale.  This is the
    scale on which the MP law is meaningful: a raw return covariance has
    diagonal entries around ``1e-4`` for daily data, so the eigenvalues span
    the same tiny range and any histogram of them is unreadable.  Normalising
    to unit variances leaves only the correlation structure — which is exactly
    what the theory describes and what the literature plots.
    """
    cov = sample_cov(x)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    corr = cov / np.outer(sd, sd)
    np.fill_diagonal(corr, 1.0)
    corr = 0.5 * (corr + corr.T)
    dec = eigen_decompose(corr, window)
    law = dec.mp_law(sigma_method=sigma_method)
    return dec, law


def fig_spectrum_mp(rets: pd.DataFrame, out: Path, window: int = 252,
                    sigma_method: str = "median_bulk", dpi: int = 200) -> Path:
    """Eigenvalue histogram against the fitted MP density (the core figure)."""
    plt = _plt()
    x = np.asarray(rets.values[-window:], float)
    dec, law = _corr_eigen(x, window, sigma_method)
    ev = dec.eigenvalues
    n_sig = int(np.sum(ev >= law.lambda_plus))

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    # histogram of eigenvalues (density-normalised so it sits on the same scale
    # as the MP density, which integrates to one for q < 1)
    ax.hist(ev, bins=28, density=True, color="#B0BEC5", edgecolor="white",
            linewidth=0.6, label="Empirical spectrum", zorder=2)

    lo = max(law.lambda_minus * 0.5, 1e-6)
    hi = law.lambda_plus * 1.02
    grid = np.linspace(lo, hi, 800)
    ax.plot(grid, mp_density(grid, law.sigma, law.q), color="#C0392B",
            lw=2.4, label=f"MP fit  ($\\sigma$={law.sigma:.3f})", zorder=3)

    ax.axvline(law.lambda_minus, color="#37474F", ls=":", lw=1.8, zorder=3)
    ax.axvline(law.lambda_plus, color="#37474F", ls=":", lw=1.8, zorder=3)
    ax.axvspan(law.lambda_minus, law.lambda_plus, color="#37474F",
               alpha=0.06, zorder=1)
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax)
    ax.annotate(r"$\lambda_-$", xy=(law.lambda_minus, ymax * 0.80),
                xytext=(-16, 0), textcoords="offset points", fontsize=13,
                color="#37474F")
    ax.annotate(r"$\lambda_+$", xy=(law.lambda_plus, ymax * 0.80),
                xytext=(6, 0), textcoords="offset points", fontsize=13,
                color="#37474F")
    ax.annotate("noise band", xy=(0.5 * (law.lambda_minus + law.lambda_plus),
                                  ymax * 0.62),
                ha="center", fontsize=10.5, color="#546E7A")
    ax.annotate("genuine factors", xy=(0.5 * (law.lambda_plus + ev[-1]),
                                       ymax * 0.45),
                ha="center", fontsize=10.5, color="#C0392B")

    ax.set_xlim(lo, ev[-1] * 1.05)
    ax.set_xlabel(r"Eigenvalue $\lambda$ of the sample correlation matrix")
    ax.set_ylabel("Probability density")
    share = float(ev[ev >= law.lambda_plus].sum() / ev.sum()) * 100.0
    ax.set_title(
        f"Marchenko–Pastur bulk and the noisy spectrum\n"
        f"$N={dec.n_assets}$, $T={dec.n_obs}$, $q={law.q:.3f}$ — "
        f"{n_sig} of {dec.n_assets} eigenvalues escape the band and carry "
        f"{share:.1f}% of the variance", fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, loc="upper right")
    return _save(fig, out, dpi)


def fig_spectrum_mp_log(rets: pd.DataFrame, out: Path, window: int = 252,
                        sigma_method: str = "median_bulk",
                        dpi: int = 200) -> Path:
    """The same spectrum on a log x-axis — where the RMT model is tested."""
    plt = _plt()
    x = np.asarray(rets.values[-window:], float)
    dec, law = _corr_eigen(x, window, sigma_method)
    ev = dec.eigenvalues

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    # empirical CDF, the cleanest way to compare tails
    ecdf = np.arange(1, ev.size + 1) / ev.size
    ax.step(ev, ecdf, where="post", color="#2C3E50", lw=2.2,
            label="Empirical CDF", zorder=3)

    lo = max(law.lambda_minus * 0.35, 1e-6)
    grid = np.logspace(np.log10(lo), np.log10(ev[-1] * 1.05), 600)
    from rmt_portfolio.rmt.mp_law import _mp_cdf
    ax.plot(grid, _mp_cdf(grid, law.sigma, law.q), color="#C0392B", lw=2.2,
            ls="--", label=f"MP CDF ($\\sigma$={law.sigma:.3f})", zorder=4)

    ax.axvline(law.lambda_plus, color="#37474F", ls=":", lw=1.8)
    ax.axvspan(lo, law.lambda_plus, color="#37474F", alpha=0.06, zorder=1)
    ax.text(law.lambda_plus * 1.10, 0.14, r"$\lambda_+$", fontsize=13,
            color="#37474F")
    ax.text(np.sqrt(lo * law.lambda_plus), 0.70, "noise band",
            fontsize=12, color="#546E7A", ha="center")
    ax.text(ev[-1] * 0.5, 0.70, "signal", fontsize=12, color="#C0392B",
            ha="center")

    ax.set_xscale("log")
    ax.set_xlabel(r"Eigenvalue $\lambda$ of the correlation matrix  (log scale)")
    ax.set_ylabel("Cumulative probability")
    ax.set_ylim(0, 1.02)
    ax.set_title("The MP law describes the bulk, not the tail\n"
                 "the empirical CDF tracks the theory inside the band and "
                 "peels away above it", fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, loc="upper left")
    return _save(fig, out, dpi)


def fig_spectrum_nt_grid(rets: pd.DataFrame, out: Path,
                         windows: Optional[List[int]] = None,
                         dpi: int = 200) -> Path:
    """The spectrum for several window lengths: how the band moves with q."""
    plt = _plt()
    windows = windows or [120, 252, 500, 756]
    available = [w for w in windows if w <= rets.shape[0]]
    R = np.asarray(rets.values, float)
    n = R.shape[1]

    fig, axes = plt.subplots(1, len(available),
                             figsize=(3.5 * len(available), 4.2),
                             sharey=False)
    if len(available) == 1:
        axes = [axes]

    rows = []
    for ax, T in zip(axes, available):
        dec, law = _corr_eigen(R[-T:], T)
        ev = dec.eigenvalues
        q = n / T
        n_sig = int(np.sum(ev >= law.lambda_plus))
        share = float(ev[ev >= law.lambda_plus].sum() / ev.sum()) * 100.0

        ax.hist(ev, bins=22, density=True, color="#B0BEC5",
                edgecolor="white", linewidth=0.5, zorder=2)
        grid = np.linspace(max(law.lambda_minus * 0.5, 1e-6),
                           law.lambda_plus * 1.02, 500)
        ax.plot(grid, mp_density(grid, law.sigma, law.q), color="#C0392B",
                lw=2.0, zorder=3)
        ax.axvspan(law.lambda_minus, law.lambda_plus, color="#37474F",
                   alpha=0.07, zorder=1)
        ax.set_xlim(max(law.lambda_minus * 0.5, 1e-6), ev[-1] * 1.07)
        ax.set_title(f"$T={T}$\n$q={q:.3f}$", fontsize=10)
        ax.text(0.97, 0.94, f"{n_sig} signals\n{share:.0f}% of var.",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B0BEC5",
                          alpha=0.9))
        ax.set_xlabel(r"$\lambda$")
        rows.append({"T": T, "q": q, "lambda_plus": law.lambda_plus,
                     "sigma": law.sigma, "n_signal": n_sig,
                     "signal_var_share": share / 100.0})
    axes[0].set_ylabel("Density")
    fig.suptitle("As the estimation window shortens, the noise band widens "
                 "and fewer factors survive",
                 fontsize=12, y=1.02)
    _save(fig, out, dpi)

    from rmt_portfolio.utils import save_table
    save_table(pd.DataFrame(rows),
               out.parent / (out.stem + "_data.csv"), index=False)
    return out


def fig_eigenvalue_stability(rets: pd.DataFrame, out: Path,
                             window: int = 252, step: int = 21,
                             max_rank: int = 12, dpi: int = 200) -> Path:
    """Vector overlap between consecutive windows, by rank.

    A genuine factor (the market mode) should keep the same direction from one
    window to the next, so its overlap should sit near one.  Noise
    eigenvectors should have no persistence.
    """
    plt = _plt()
    R = np.asarray(rets.values, float)
    vecs = []
    for t in range(window, R.shape[0], step):
        dec, _ = _corr_eigen(R[t - window:t], window)
        vecs.append(dec.eigenvectors[:, ::-1])       # descending
    if len(vecs) < 2:
        LOGGER.warning("  not enough windows for stability plot")
        return out

    ranks = min(max_rank, R.shape[1])
    overlaps = np.zeros((len(vecs) - 1, ranks))
    for i in range(1, len(vecs)):
        for r in range(ranks):
            overlaps[i - 1, r] = abs(float(vecs[i][:, r] @ vecs[i - 1][:, r]))

    mean = overlaps.mean(axis=0)
    sd = overlaps.std(axis=0)

    from rmt_portfolio.utils import save_table
    save_table(pd.DataFrame({"rank": np.arange(1, ranks + 1),
                             "mean_overlap": mean, "std_overlap": sd}),
               out.parent / (out.stem + "_data.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.errorbar(np.arange(1, ranks + 1), mean, yerr=sd, fmt="o-",
                color="#0072B2", ecolor="#90A4AE", capsize=3.5, lw=1.8,
                markersize=6)
    ax.axhline(1.0, color="#C0392B", ls="--", lw=1.2, alpha=0.7)
    ax.text(ranks * 0.62, 1.003, "perfect persistence", color="#C0392B",
            fontsize=9.5, va="bottom")
    ax.axhspan(0.0, 0.75, color="#37474F", alpha=0.05)
    ax.text(1.3, 0.42, "noise regime (no persistence)", fontsize=9.5,
            color="#546E7A")
    ax.set_xlabel("Eigenvalue rank (1 = largest)")
    ax.set_ylabel("|overlap| between consecutive windows")
    ax.set_ylim(0, 1.07)
    ax.set_xticks(np.arange(1, ranks + 1))
    ax.set_title("Only the largest eigenvector is stable across windows\n"
                 f"rolling $T={window}$, step={step} days, "
                 f"{len(overlaps)} transitions", fontsize=11)
    return _save(fig, out, dpi)


def fig_eigenvalue_growth(rets: pd.DataFrame, out: Path, window: int = 252,
                          sigma_method: str = "median_bulk",
                          dpi: int = 200) -> Path:
    """Ranked spectrum on a log scale against the MP bulk."""
    plt = _plt()
    R = np.asarray(rets.values, float)
    dec, law = _corr_eigen(R[-window:], window, sigma_method)
    ev = dec.eigenvalues

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(np.arange(1, ev.size + 1), ev[::-1] / law.lambda_plus,
            "o-", color="#2C3E50", lw=1.9, markersize=6,
            label=r"sample $\lambda_i / \lambda_+$")
    ax.axhline(1.0, color="#C0392B", ls="--", lw=1.8,
               label=r"MP edge $\lambda_+$")
    ax.axhspan(min(ev) / law.lambda_plus * 0.8, 1.0, color="#37474F",
               alpha=0.06)
    ax.text(ev.size * 0.42, 0.30, "inside the noise band", fontsize=10.5,
            color="#546E7A")
    ax.set_yscale("log")
    ax.set_xlabel("Eigenvalue rank (1 = largest)")
    ax.set_ylabel(r"$\lambda_i / \lambda_+$   (log scale)")
    n_sig = int(np.sum(ev >= law.lambda_plus))
    ax.set_title(f"Ratios of the ranked spectrum to the MP edge\n"
                 f"{n_sig} eigenvalues lie above the line; the rest are "
                 f"statistically indistinguishable from noise", fontsize=11)
    ax.legend(frameon=True, framealpha=0.95)
    return _save(fig, out, dpi)


# ===========================================================================
# Group B — what denoising does to the matrix
# ===========================================================================


def fig_eigen_shrinkage(rets: pd.DataFrame, out: Path, window: int = 252,
                        estimators: Optional[List[str]] = None,
                        dpi: int = 200) -> Path:
    """Sample eigenvalues vs cleaned eigenvalues for each estimator."""
    plt = _plt()
    estimators = estimators or ["rmt_hard", "rmt_soft", "rmt_rie",
                                "factor_model", "ledoit_wolf"]
    x = np.asarray(rets.values[-window:], float)
    cov = sample_cov(x)
    _, sd = np.ones(cov.shape[0]), np.sqrt(np.diag(cov))
    base_corr = cov / np.outer(sd, sd)
    np.fill_diagonal(base_corr, 1.0)
    lam_sample = np.sort(np.linalg.eigvalsh(base_corr))

    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    lim = lam_sample[-1] * 1.12
    ax.plot([0, lim], [0, lim], color="#B0BEC5", ls="--", lw=1.6,
            label="identity (no change)", zorder=1)

    rows = []
    for name in estimators:
        m = estimate_covariance(x, method=name, n_obs=window)
        if isinstance(m, tuple):
            m = m[0]
        s = np.sqrt(np.diag(m))
        c = m / np.outer(s, s)
        np.fill_diagonal(c, 1.0)
        lam_after = np.sort(np.linalg.eigvalsh(c))
        ax.plot(lam_sample, lam_after, "o-", lw=1.7, markersize=4.6,
                color=ESTIMATOR_COLORS.get(name, "#333333"),
                label=SHORT.get(name, name), zorder=3, alpha=0.92)
        rows.append(pd.DataFrame({"estimator": name, "lambda_sample": lam_sample,
                                  "lambda_cleaned": lam_after}))

    ax.set_xlabel(r"Sample eigenvalue $\lambda_i$  (correlation scale)")
    ax.set_ylabel(r"Cleaned eigenvalue $\tilde\lambda_i$")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("What each estimator does to the spectrum\n"
                 "hard thresholding flattens the bulk, the RIE lifts it, "
                 "Ledoit-Wolf barely moves it", fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, loc="upper left")
    _save(fig, out, dpi)

    from rmt_portfolio.utils import save_table
    save_table(pd.concat(rows, ignore_index=True),
               out.parent / (out.stem + "_data.csv"), index=False)
    return out


def fig_condition_numbers(rets: pd.DataFrame, out: Path, window: int = 252,
                          step: int = 21, estimators: Optional[List[str]] = None,
                          dpi: int = 200) -> Path:
    """Condition number and effective rank of every estimator over time."""
    plt = _plt()
    estimators = estimators or ["sample", "ledoit_wolf", "rmt_hard",
                                "rmt_soft", "rmt_rie", "factor_model"]
    R = np.asarray(rets.values, float)
    dates, cond, erank = [], {e: [] for e in estimators}, \
        {e: [] for e in estimators}

    for t in range(window, R.shape[0], step):
        x = R[t - window:t]
        dates.append(rets.index[t])
        for name in estimators:
            m = estimate_covariance(x, method=name, n_obs=window)
            if isinstance(m, tuple):
                m = m[0]
            ev = np.linalg.eigvalsh(m)
            cond[name].append(ev[-1] / max(ev[0], 1e-300))
            p = ev / ev.sum()
            p = p[p > 0]
            erank[name].append(float(np.exp(-np.sum(p * np.log(p)))))

    fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.0), sharex=True)
    for name in estimators:
        c = ESTIMATOR_COLORS.get(name, "#333333")
        axes[0].plot(dates, cond[name], lw=1.7, color=c,
                     label=SHORT.get(name, name), alpha=0.9)
        axes[1].plot(dates, erank[name], lw=1.7, color=c, alpha=0.9)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("Condition number\n(log scale)")
    axes[0].set_title("Ill-conditioning of the risk model over time\n"
                      "the sample matrix is the worst conditioned; "
                      "the factor model the best", fontsize=11)
    axes[0].legend(frameon=True, framealpha=0.95, ncol=3, fontsize=9,
                   loc="upper left")
    axes[1].set_ylabel("Effective rank\n(participation ratio)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Effective number of principal directions", fontsize=10)
    fig.autofmt_xdate()
    return _save(fig, out, dpi)


def fig_corr_heatmaps(rets: pd.DataFrame, out: Path, window: int = 252,
                      dpi: int = 200) -> Path:
    """Sample vs RMT-denoised correlation matrix."""
    plt = _plt()
    x = np.asarray(rets.values[-window:], float)
    cov = sample_cov(x)
    sd = np.sqrt(np.diag(cov))
    c0 = cov / np.outer(sd, sd)
    np.fill_diagonal(c0, 1.0)
    m1 = estimate_covariance(x, method="rmt_hard", n_obs=window)
    if isinstance(m1, tuple):
        m1 = m1[0]
    s1 = np.sqrt(np.diag(m1))
    c1 = m1 / np.outer(s1, s1)
    np.fill_diagonal(c1, 1.0)

    vmax = max(abs(c0).max(), abs(c1).max())
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    for ax, c, title in zip(
            axes, [c0, c1],
            ["Sample correlation", "RMT hard-thresholded"]):
        im = ax.imshow(c, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{title}\n"
                     f"$\\kappa$={np.linalg.cond(c):.0f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes, fraction=0.026, pad=0.02, label="correlation")
    fig.suptitle("The denoised matrix keeps the block structure and removes "
                 "the speckle", fontsize=12, y=0.99)
    return _save(fig, out, dpi)


# ===========================================================================
# Group C — backtest results
# ===========================================================================


def fig_equity_grid(equity_file: Path, out: Path, dpi: int = 200) -> Path:
    """Small multiples: one panel per allocator, one line per estimator."""
    plt = _plt()
    d = np.load(equity_file, allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    strategies = [str(s) for s in d["strategies"]]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), sharex=True)
    axes = axes.ravel()
    allocators = ["equal_weight", "inverse_variance", "minimum_variance",
                  "risk_parity", "hrp", "max_diversification"]

    for ax, al in zip(axes, allocators):
        for s in strategies:
            if not s.endswith("__" + al):
                continue
            est = s.split("__")[0]
            ax.plot(dates, d[f"eq_{s}"], lw=1.5,
                    color=ESTIMATOR_COLORS.get(est, "#333333"),
                    label=SHORT.get(est, est), alpha=0.9)
        ax.set_title(SHORT_AL.get(al, al), fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=True, framealpha=0.95, fontsize=8.5, ncol=2)
    for ax in axes[3:]:
        ax.set_xlabel("Date")
    for ax in axes[::3]:
        ax.set_ylabel("Net wealth (start = 1)")
    fig.autofmt_xdate()
    fig.suptitle("Net equity curves by allocator (10 bp one-way costs)",
                 fontsize=13, y=1.0)
    return _save(fig, out, dpi)


def fig_equity_curves(equity_file: Path, out: Path,
                      allocator: str = "minimum_variance",
                      dpi: int = 200) -> Path:
    """Cumulative net wealth for one allocator, all estimators."""
    plt = _plt()
    d = np.load(equity_file, allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    strategies = [str(s) for s in d["strategies"]]

    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    best = None
    for s in strategies:
        if not s.endswith("__" + allocator):
            continue
        est = s.split("__")[0]
        eq = d[f"eq_{s}"]
        ax.plot(dates, eq, lw=1.9, color=ESTIMATOR_COLORS.get(est, "#333333"),
                label=f"{SHORT.get(est, est)}  (x{eq[-1]:.2f})", alpha=0.92)
        if best is None or eq[-1] > best[1]:
            best = (est, eq[-1])
    ax.axhline(1.0, color="#90A4AE", ls="--", lw=1.0)
    ax.set_ylabel("Net wealth (start = 1)")
    ax.set_xlabel("Date")
    ax.set_title(f"Net of costs: effects of the covariance estimator under "
                 f"{SHORT_AL.get(allocator, allocator)}\n"
                 "all strategies share the same allocator, so any difference "
                 "is attributable to the risk model", fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, fontsize=9.5)
    fig.autofmt_xdate()
    return _save(fig, out, dpi)


def fig_drawdowns(equity_file: Path, out: Path,
                  allocator: str = "minimum_variance", dpi: int = 200) -> Path:
    """Drawdown paths for one allocator."""
    plt = _plt()
    d = np.load(equity_file, allow_pickle=True)
    dates = pd.to_datetime(d["dates"])
    strategies = [str(s) for s in d["strategies"]]

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    for s in strategies:
        if not s.endswith("__" + allocator):
            continue
        est = s.split("__")[0]
        eq = d[f"eq_{s}"]
        dd = eq / np.maximum.accumulate(eq) - 1.0
        ax.plot(dates, dd * 100.0, lw=1.5,
                color=ESTIMATOR_COLORS.get(est, "#333333"),
                label=f"{SHORT.get(est, est)}  (max {dd.min()*100:.1f}%)",
                alpha=0.9)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_title(f"Drawdowns under {SHORT_AL.get(allocator, allocator)}",
                 fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, fontsize=9)
    fig.autofmt_xdate()
    return _save(fig, out, dpi)


def fig_risk_return(table_file: Path, out: Path, dpi: int = 200) -> Path:
    """Realised volatility against realised return, with Sharpe iso-lines."""
    plt = _plt()
    t = pd.read_csv(table_file)
    t = t[t["cov_method"] != "benchmark"]
    if t.empty:
        return out

    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    vmax = t["ann_volatility"].max() * 1.2
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        xs = np.linspace(0, vmax, 50)
        ax.plot(xs, s * xs, color="#CFD8DC", lw=0.9, zorder=0)
        ax.text(vmax * 0.93, s * vmax * 0.93 - 0.004, f"SR={s:.2f}",
                fontsize=8, color="#90A4AE", ha="right")

    for al, sub in t.groupby("allocator"):
        for _, row in sub.iterrows():
            est = row["cov_method"]
            ax.scatter(row["ann_volatility"], row["ann_return_net"],
                       s=64, marker="o" if al != "hrp" else "s",
                       color=ESTIMATOR_COLORS.get(est, "#333333"),
                       edgecolor="white", linewidth=0.7, zorder=3, alpha=0.92)
    # annotate the extremes
    top = t.nlargest(3, "sharpe_net")
    for _, row in top.iterrows():
        ax.annotate(f"{SHORT.get(row['cov_method'], row['cov_method'])}"
                    f"+{SHORT_AL.get(row['allocator'], row['allocator'])}",
                    (row["ann_volatility"], row["ann_return_net"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8.5,
                    color="#37474F")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=ESTIMATOR_COLORS[e],
                      label=SHORT[e]) for e in SHORT if e in set(t["cov_method"])]
    ax.legend(handles=handles, frameon=True, framealpha=0.95, fontsize=9,
              loc="upper left", ncol=2)
    ax.set_xlabel("Annualised volatility (net)")
    ax.set_ylabel("Annualised return (net)")
    ax.set_title("Risk/return of every (estimator, allocator) cell\n"
                 "grey rays are Sharpe iso-lines", fontsize=11)
    return _save(fig, out, dpi)


def fig_sharpe_bars(table_file: Path, out: Path, dpi: int = 200) -> Path:
    """Grouped bar chart of net Sharpe by estimator within each allocator."""
    plt = _plt()
    t = pd.read_csv(table_file)
    t = t[t["cov_method"] != "benchmark"]
    if t.empty:
        return out
    ests = [e for e in SHORT if e in set(t["cov_method"])]
    allocs = [a for a in SHORT_AL if a in set(t["allocator"])]
    n_e = len(ests)
    x = np.arange(len(allocs))
    width = 0.8 / max(n_e, 1)

    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    for i, est in enumerate(ests):
        sub = t[t["cov_method"] == est].set_index("allocator")
        vals = [sub.loc[a, "sharpe_net"] if a in sub.index else np.nan
                for a in allocs]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92,
               color=ESTIMATOR_COLORS.get(est, "#333333"),
               label=SHORT.get(est, est), edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="#37474F", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_AL[a] for a in allocs], fontsize=9.5)
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_title("Net Sharpe by estimator, within each allocator", fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, ncol=n_e, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    return _save(fig, out, dpi)


def fig_turnover_sharpe(table_file: Path, out: Path, dpi: int = 200) -> Path:
    """The turnover/Sharpe trade-off with the Pareto frontier."""
    plt = _plt()
    t = pd.read_csv(table_file)
    t = t[t["cov_method"] != "benchmark"].dropna(
        subset=["avg_turnover", "sharpe_net"])
    if t.empty:
        return out

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for _, row in t.iterrows():
        ax.scatter(row["avg_turnover"] * 100, row["sharpe_net"], s=62,
                   color=ESTIMATOR_COLORS.get(row["cov_method"], "#333333"),
                   marker={"minimum_variance": "o", "risk_parity": "^",
                           "hrp": "s", "equal_weight": "D",
                           "inverse_variance": "v",
                           "max_diversification": "P"}.get(
                               row["allocator"], "o"),
                   edgecolor="white", linewidth=0.7, alpha=0.92, zorder=3)
    # Pareto frontier: maximise Sharpe, minimise turnover
    srt = t.sort_values("avg_turnover")
    best, front = -np.inf, []
    for _, r in srt.iterrows():
        if r["sharpe_net"] > best:
            best = r["sharpe_net"]
            front.append(r)
    if front:
        fx = [r["avg_turnover"] * 100 for r in front]
        fy = [r["sharpe_net"] for r in front]
        ax.step(fx, fy, where="post", color="#C0392B", lw=2.0, alpha=0.8,
                label="Pareto frontier", zorder=4)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=ESTIMATOR_COLORS[e],
                      label=SHORT[e]) for e in SHORT
               if e in set(t["cov_method"])]
    handles.append(Line2D([], [], color="#C0392B", lw=2.0,
                          label="Pareto frontier"))
    ax.legend(handles=handles, frameon=True, framealpha=0.95, fontsize=8.8,
              ncol=2, loc="lower right")
    ax.set_xlabel("Average turnover per rebalance (%)")
    ax.set_ylabel("Net Sharpe ratio")
    ax.set_title("Turnover/Sharpe trade-off\n"
                 "markers: circle = min-var, triangle = risk parity, "
                 "square = HRP, diamond = 1/N", fontsize=11)
    return _save(fig, out, dpi)


# ===========================================================================
# Group D — risk-model quality
# ===========================================================================


def fig_forecast_accuracy(forecast_file: Path, out: Path,
                          allocator: str = "minimum_variance",
                          dpi: int = 200) -> Path:
    """Predicted vs realised portfolio volatility, per estimator."""
    plt = _plt()
    df = pd.read_csv(forecast_file)
    sub = df[df["allocator"] == allocator]
    if sub.empty:
        return out
    ests = [e for e in SHORT if e in set(sub["estimator"])]

    ncol = 4
    nrow = int(np.ceil(len(ests) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    lim = max(sub["pred_vol"].max(), sub["real_vol"].max()) * 1.08
    for ax, est in zip(axes, ests):
        s = sub[sub["estimator"] == est]
        ax.scatter(s["pred_vol"], s["real_vol"], s=26, alpha=0.72,
                   color=ESTIMATOR_COLORS.get(est, "#333333"),
                   edgecolor="white", linewidth=0.4)
        ax.plot([0, lim], [0, lim], color="#C0392B", ls="--", lw=1.3)
        ratio = s["pred_vol"].mean() / s["real_vol"].mean() - 1
        ax.set_title(f"{SHORT.get(est, est)}\nbias {ratio*100:+.1f}%, "
                     f"MAE {np.mean(np.abs(s['pred_vol']-s['real_vol'])):.2e}",
                     fontsize=9.5)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
    for ax in axes:
        ax.set_xlabel("Predicted vol", fontsize=8.5)
    for ax in axes[::ncol]:
        ax.set_ylabel("Realised vol", fontsize=8.5)
    for ax in axes[len(ests):]:
        ax.axis("off")
    fig.suptitle(f"Ex-ante vs realised portfolio volatility "
                 f"({SHORT_AL.get(allocator, allocator)})\n"
                 f"points below the 45° line mean the model under-stated risk",
                 fontsize=11.5, y=1.02)
    return _save(fig, out, dpi)


def fig_forecast_qlike(forecast_file: Path, out: Path,
                       dpi: int = 200) -> Path:
    """QLIKE loss by estimator, grouped by allocator — the robust ranking."""
    plt = _plt()
    df = pd.read_csv(forecast_file)
    if df.empty:
        return out
    from rmt_portfolio.utils import save_table

    rows = []
    for (est, al), s in df.groupby(["estimator", "allocator"]):
        ratio = s["pred_var"].values / np.maximum(s["real_var"].values, 1e-300)
        rows.append({"estimator": est, "allocator": al,
                     "qlike": float(np.mean(ratio - np.log(ratio) - 1.0))})
    q = pd.DataFrame(rows)
    save_table(q, out.parent / (out.stem + "_data.csv"), index=False)

    ests = [e for e in SHORT if e in set(q["estimator"])]
    allocs = [a for a in SHORT_AL if a in set(q["allocator"])]
    x = np.arange(len(allocs))
    width = 0.8 / max(len(ests), 1)

    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    for i, est in enumerate(ests):
        sub = q[q["estimator"] == est].set_index("allocator")
        vals = [sub.loc[a, "qlike"] if a in sub.index else np.nan
                for a in allocs]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width * 0.92,
               color=ESTIMATOR_COLORS.get(est, "#333333"),
               label=SHORT.get(est, est), edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_AL[a] for a in allocs], fontsize=9.5)
    ax.set_ylabel("QLIKE loss (lower is better)")
    ax.set_title("Risk-forecast loss, the metric that does not need an "
                 "expected-return model\n"
                 "QLIKE is robust to a noisy volatility proxy",
                 fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, ncol=len(ests), fontsize=8.8,
              loc="upper center", bbox_to_anchor=(0.5, 1.0))
    return _save(fig, out, dpi)


def fig_forecast_bias_time(forecast_file: Path, out: Path, step: int = 6,
                           dpi: int = 200) -> Path:
    """Rolling forecast bias over time, per estimator."""
    plt = _plt()
    df = pd.read_csv(forecast_file)
    df["date"] = pd.to_datetime(df["date"])
    df["ratio"] = (df["pred_vol"] / np.maximum(df["real_vol"], 1e-300) - 1.0) * 100
    if df.empty:
        return out

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for est in [e for e in SHORT if e in set(df["estimator"])]:
        s = df[df["estimator"] == est].groupby("date")["ratio"].mean()
        ax.plot(s.index, s.values, lw=1.6,
                color=ESTIMATOR_COLORS.get(est, "#333333"),
                label=SHORT.get(est, est), alpha=0.9)
    ax.axhline(0, color="#37474F", lw=1.0)
    ax.set_ylabel("Forecast bias (%)\n(predicted / realised − 1)")
    ax.set_xlabel("Date")
    ax.set_title("Forecast bias through time (averaged over allocators)\n"
                 "below zero = under-stated risk, the dangerous direction",
                 fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, fontsize=9, ncol=2)
    fig.autofmt_xdate()
    return _save(fig, out, dpi)


def fig_enb_distribution(forecast_file: Path, out: Path,
                         dpi: int = 200) -> Path:
    """Distribution of the effective number of bets, per estimator."""
    plt = _plt()
    df = pd.read_csv(forecast_file)
    sub = df[df["allocator"] == "minimum_variance"]
    if sub.empty:
        return out
    ests = [e for e in SHORT if e in set(sub["estimator"])]
    data = [sub[sub["estimator"] == e]["enb"].dropna().values for e in ests]

    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                    medianprops=dict(color="#212121", lw=1.6),
                    flierprops=dict(marker="o", markersize=3.2, alpha=0.5))
    for patch, est in zip(bp["boxes"], ests):
        patch.set_facecolor(ESTIMATOR_COLORS.get(est, "#333333"))
        patch.set_alpha(0.72)
    ax.set_xticklabels([SHORT.get(e, e) for e in ests], fontsize=9.5)
    ax.axhline(len(sub["estimator"].unique()) and 1.0, color="#C0392B",
               ls="--", lw=1.0)
    ax.set_ylabel("Effective number of bets (ENB)")
    ax.set_title("How many genuinely independent bets does minimum variance "
                 "take?\nENB = 1 means a single asset in disguise",
                 fontsize=11)
    return _save(fig, out, dpi)


def fig_risk_contribution(rets: pd.DataFrame, out: Path, window: int = 252,
                          estimator: str = "rmt_hard", dpi: int = 200) -> Path:
    """Risk contributions under risk parity, sample vs denoised."""
    plt = _plt()
    from rmt_portfolio.portfolio import allocate
    x = np.asarray(rets.values[-window:], float)
    assets = list(rets.columns)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
    for ax, est in zip(axes, ["sample", estimator]):
        cov = estimate_covariance(x, method=est, n_obs=window)
        if isinstance(cov, tuple):
            cov = cov[0]
        res = allocate(cov=cov, method="risk_parity")
        rc = risk_contributions(res.weights, cov) * 100
        order = np.argsort(-res.weights)
        ax.bar(range(len(order)), rc[order],
               color=ESTIMATOR_COLORS.get(est, "#333333"), alpha=0.85)
        ax.axhline(100.0 / len(order), color="#C0392B", ls="--", lw=1.4,
                   label="equal RC (target)")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([assets[i] for i in order], rotation=90, fontsize=7.5)
        ax.set_title(f"{SHORT.get(est, est)}\n"
                     f"ENB={res.enb:.2f}", fontsize=10.5)
        ax.legend(fontsize=8.5)
    axes[0].set_ylabel("Risk contribution (%)")
    fig.suptitle("Risk parity: does the allocator actually equalise risk?",
                 fontsize=12, y=1.0)
    return _save(fig, out, dpi)


# ===========================================================================
# Group E — robustness
# ===========================================================================


def fig_robustness_window(rob_file: Path, out: Path, dpi: int = 200) -> Path:
    """Net Sharpe against the window length T (the q sweep)."""
    plt = _plt()
    if not rob_file.exists():
        return out
    df = pd.read_csv(rob_file)
    if "window" not in df.columns or df.empty:
        return out
    piv = df.groupby(["window", "estimator"])["sharpe_net"].mean().unstack()

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for est in piv.columns:
        ax.plot(piv.index, piv[est] * 100, "o-", lw=1.9, markersize=6,
                color=ESTIMATOR_COLORS.get(est, "#333333"),
                label=SHORT.get(est, est), alpha=0.92)
    secax = ax.secondary_xaxis(
        "top", functions=(lambda T: 16.0 / T, lambda q: 16.0 / q))
    secax.set_xlabel(r"aspect ratio $q = N/T$")
    ax.set_xlabel("Estimation window $T$ (trading days)")
    ax.set_ylabel("Net Sharpe ratio (x100, averaged over allocators)")
    ax.set_title("Robustness to the estimation window\n"
                 "the ordering of estimators is stable; the gap to the sample "
                 "matrix is widest at short windows", fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, ncol=2, fontsize=9)
    return _save(fig, out, dpi)


def fig_robustness_threshold(rob_file: Path, out: Path,
                             dpi: int = 200) -> Path:
    """Net Sharpe against the lambda_+ multiplier."""
    plt = _plt()
    if not rob_file.exists():
        return out
    df = pd.read_csv(rob_file)
    if "lambda_plus_mult" not in df.columns or df.empty:
        return out
    piv = df.groupby(["lambda_plus_mult", "estimator"])["sharpe_net"].mean() \
        .unstack()

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for est in piv.columns:
        ax.plot(piv.index, piv[est] * 100, "o-", lw=1.9, markersize=6.5,
                color=ESTIMATOR_COLORS.get(est, "#333333"),
                label=SHORT.get(est, est), alpha=0.92)
    ax.axvline(1.0, color="#37474F", ls=":", lw=1.4)
    ax.text(1.005, ax.get_ylim()[0], " fitted edge", fontsize=8.5,
            color="#37474F", va="bottom")
    ax.set_xlabel(r"Multiplier applied to $\lambda_+$")
    ax.set_ylabel("Net Sharpe ratio (x100)")
    ax.set_title("Sensitivity to the threshold\n"
                 "the surface is almost flat between 0.85x and 1.15x the "
                 "fitted edge — no lucky calibration", fontsize=11)
    ax.legend(frameon=True, framealpha=0.95, fontsize=9.5)
    return _save(fig, out, dpi)


def fig_robustness_cost(rob_file: Path, out: Path, dpi: int = 200) -> Path:
    """Gross vs net Sharpe and turnover against the cost level."""
    plt = _plt()
    if not rob_file.exists():
        return out
    df = pd.read_csv(rob_file)
    if "cost_bps" not in df.columns or df.empty:
        return out
    piv_g = df.groupby(["cost_bps", "estimator"])["sharpe"].mean().unstack()
    piv_n = df.groupby(["cost_bps", "estimator"])["sharpe_net"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    for est in piv_g.columns:
        c = ESTIMATOR_COLORS.get(est, "#333333")
        axes[0].plot(piv_g.index, piv_g[est] * 100, "o--", lw=1.5,
                     markersize=5, color=c, alpha=0.75,
                     label=f"{SHORT.get(est, est)} (gross)")
        if est in piv_n.columns:
            axes[0].plot(piv_n.index, piv_n[est] * 100, "o-", lw=1.9,
                         markersize=6, color=c,
                         label=f"{SHORT.get(est, est)} (net)")
    axes[0].set_xlabel("One-way cost (bp)")
    axes[0].set_ylabel("Sharpe ratio (x100)")
    axes[0].set_title("Gross vs net Sharpe across cost levels", fontsize=10.5)
    axes[0].legend(frameon=True, framealpha=0.95, fontsize=7.6, ncol=2)

    piv_t = df.groupby(["cost_bps", "estimator"])["avg_turnover"].mean() \
        .unstack()
    for est in piv_t.columns:
        axes[1].plot(piv_t.index, piv_t[est] * 100, "o-", lw=1.8,
                     markersize=6, color=ESTIMATOR_COLORS.get(est, "#333333"),
                     label=SHORT.get(est, est))
    axes[1].set_xlabel("One-way cost (bp)")
    axes[1].set_ylabel("Average turnover per rebalance (%)")
    axes[1].set_title("Turnover is invariant to the cost level\n"
                      "(costs change net performance, not trading)",
                      fontsize=10.5)
    axes[1].legend(frameon=True, framealpha=0.95, fontsize=8.5, ncol=2)
    return _save(fig, out, dpi)


# ===========================================================================
# Entry point
# ===========================================================================


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--universe", default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--group", default="all",
                    choices=["all", "A", "B", "C", "D", "E"])
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_global_seed(int(cfg.project.seed))
    apply_plot_style(dpi=int(cfg.figures.dpi))
    universe = args.universe or str(cfg.data.universe)
    window = int(args.window or cfg.backtest.window)
    dpi = int(cfg.figures.dpi)

    fig_dir = ensure_dir(PROJECT_ROOT / "figures" / universe)
    res = PROJECT_ROOT / "results"
    made: Dict[str, str] = {}

    LOGGER.info("=" * 78)
    LOGGER.info("FIGURES | universe=%s window=%d group=%s", universe, window,
                args.group)
    LOGGER.info("=" * 78)

    # data always needed for groups A/B/D
    rets, _ = load_returns(
        universe, start=str(cfg.data.start), end=str(cfg.data.end),
        method="simple", min_coverage=float(cfg.data.min_coverage),
        repair_splits=bool(cfg.data.repair_splits),
        align=str(cfg.data.align), winsorise=cfg.data.winsorise, verbose=False)

    groups = ["A", "B", "C", "D", "E"] if args.group == "all" else [args.group]

    if "A" in groups:
        LOGGER.info("GROUP A — the spectrum")
        made["spectrum_mp"] = str(fig_spectrum_mp(
            rets, fig_dir / "fig_spectrum_mp.png", window,
            str(cfg.rmt.sigma_method), dpi))
        made["spectrum_mp_log"] = str(fig_spectrum_mp_log(
            rets, fig_dir / "fig_spectrum_mp_log.png", window,
            str(cfg.rmt.sigma_method), dpi))
        made["spectrum_nt_grid"] = str(fig_spectrum_nt_grid(
            rets, fig_dir / "fig_spectrum_nt_grid.png",
            list(cfg.robustness.windows), dpi))
        made["eigenvalue_stability"] = str(fig_eigenvalue_stability(
            rets, fig_dir / "fig_eigenvalue_stability.png", window, 21, 12, dpi))
        made["eigenvalue_growth"] = str(fig_eigenvalue_growth(
            rets, fig_dir / "fig_eigenvalue_growth.png", window,
            str(cfg.rmt.sigma_method), dpi))

    if "B" in groups:
        LOGGER.info("GROUP B — effect on the matrix")
        made["eigen_shrinkage"] = str(fig_eigen_shrinkage(
            rets, fig_dir / "fig_eigen_shrinkage.png", window, None, dpi))
        made["condition_numbers"] = str(fig_condition_numbers(
            rets, fig_dir / "fig_condition_numbers.png", window, 21, None, dpi))
        made["corr_heatmaps"] = str(fig_corr_heatmaps(
            rets, fig_dir / "fig_corr_heatmaps.png", window, dpi))

    # locate result files
    eq_file = res / "backtest" / f"{universe}_equity_{args.tag}.npz"
    tab_file = res / "backtest" / f"{universe}_main_{args.tag}.csv"

    if "C" in groups:
        LOGGER.info("GROUP C — backtest")
        if eq_file.exists():
            made["equity_grid"] = str(fig_equity_grid(
                eq_file, fig_dir / "fig_equity_grid.png", dpi))
            made["equity_curves"] = str(fig_equity_curves(
                eq_file, fig_dir / "fig_equity_curves.png",
                "minimum_variance", dpi))
            made["drawdowns"] = str(fig_drawdowns(
                eq_file, fig_dir / "fig_drawdowns.png",
                "minimum_variance", dpi))
        else:
            LOGGER.warning("  missing %s — run run_main_backtest.py first",
                           eq_file.name)
        if tab_file.exists():
            made["risk_return"] = str(fig_risk_return(
                tab_file, fig_dir / "fig_risk_return.png", dpi))
            made["sharpe_bars"] = str(fig_sharpe_bars(
                tab_file, fig_dir / "fig_sharpe_bars.png", dpi))
            made["turnover_sharpe"] = str(fig_turnover_sharpe(
                tab_file, fig_dir / "fig_turnover_sharpe.png", dpi))

    fc_file = res / "risk_forecast" / f"{universe}_risk_forecast_{args.tag}.csv"
    if "D" in groups:
        LOGGER.info("GROUP D — risk-model quality")
        if fc_file.exists():
            made["forecast_accuracy"] = str(fig_forecast_accuracy(
                fc_file, fig_dir / "fig_forecast_accuracy.png",
                "minimum_variance", dpi))
            made["forecast_qlike"] = str(fig_forecast_qlike(
                fc_file, fig_dir / "fig_forecast_qlike.png", dpi))
            made["forecast_bias_time"] = str(fig_forecast_bias_time(
                fc_file, fig_dir / "fig_forecast_bias_time.png", 6, dpi))
            made["enb_distribution"] = str(fig_enb_distribution(
                fc_file, fig_dir / "fig_enb_distribution.png", dpi))
        else:
            LOGGER.warning("  missing %s — run run_risk_forecast.py first",
                           fc_file.name)
        made["risk_contribution"] = str(fig_risk_contribution(
            rets, fig_dir / "fig_risk_contribution.png", window,
            "rmt_hard", dpi))

    if "E" in groups:
        LOGGER.info("GROUP E — robustness")
        rob = res / "robustness"
        made["robustness_window"] = str(fig_robustness_window(
            rob / f"window_{universe}.csv",
            fig_dir / "fig_robustness_window.png", dpi))
        made["robustness_threshold"] = str(fig_robustness_threshold(
            rob / f"threshold_{universe}.csv",
            fig_dir / "fig_robustness_threshold.png", dpi))
        made["robustness_cost"] = str(fig_robustness_cost(
            rob / f"cost_{universe}.csv",
            fig_dir / "fig_robustness_cost.png", dpi))

    manifest_path = fig_dir / f"manifest_{args.group}_{args.tag}.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(made, fh, indent=2)
    LOGGER.info("figures written to %s  (%d files)", fig_dir, len(made))
    return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
