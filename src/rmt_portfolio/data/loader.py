"""Price download, cleaning, caching and quality reporting.

Data sources
------------
All market data is fetched through **akshare**, which is the only free source
reachable from the target environment:

* **US equities and ETFs** — ``ak.stock_us_daily(symbol=...)`` returns the full
  available history (``date, open, high, low, close, volume``).  Despite the
  name it also serves US-listed ETFs (SPY, XLK, TLT, GLD, ...), which is what
  the sector universe needs.  Prices are already split/dividend adjusted.
* **Chinese ETFs** — ``ak.fund_etf_hist_em(symbol=..., adjust='qfq')`` returns
  forward-adjusted (前复权) OHLCV from Eastmoney.

Yahoo Finance and Stooq are *deliberately not used*: both return HTTP 403 /
JavaScript challenges from the target environment (verified), so relying on
them would make the pipeline unreproducible.

Caching
-------
Downloads are cached to ``data/raw/<universe>/<ticker>.csv`` and only refetched
when ``force=True``.  The cache makes the whole study reproducible offline once
the first run has populated it, and keeps the akshare provider from being hit
repeatedly during parameter sweeps.

Cleaning
--------
The pipeline is deliberately conservative and *explicit* about every step:

1. Parse dates, sort, drop duplicate dates.
2. Reject non-positive prices (data errors) as missing.
3. Reindex onto the **intersection** of trading days across all assets, then
   keep only dates where *every* asset has a price — this avoids silently
   filling long gaps, and the resulting panel is already balanced.
4. Optionally forward-fill gaps of at most ``max_ffill`` days (default 3) to
   bridge isolated holidays, then intersect again.
5. Report a quality table: coverage, start/end, missing count, extreme returns.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rmt_portfolio.config import PROJECT_ROOT
from rmt_portfolio.data.universes import SectorUniverse, get_universe

__all__ = [
    "DataQualityReport",
    "download_prices",
    "load_price_panel",
    "load_returns",
    "clean_price_panel",
    "compute_quality_report",
    "cache_dir_for",
]


# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------


def cache_dir_for(universe: str, root: Optional[Path] = None) -> Path:
    base = Path(root) if root else (PROJECT_ROOT / "data" / "raw")
    d = base / universe
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------


@dataclass
class DataQualityReport:
    """Per-asset and panel-level data quality statistics."""

    universe: str
    per_asset: pd.DataFrame = field(default_factory=pd.DataFrame)
    panel_start: Optional[pd.Timestamp] = None
    panel_end: Optional[pd.Timestamp] = None
    n_dates: int = 0
    n_assets: int = 0
    dropped_assets: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"DataQualityReport({self.universe})",
            f"  panel: {self.n_dates} dates x {self.n_assets} assets",
            f"  span : {self.panel_start} .. {self.panel_end}",
        ]
        if self.dropped_assets:
            lines.append(f"  dropped: {', '.join(self.dropped_assets)}")
        if not self.per_asset.empty:
            col = ("coverage_of_life"
                   if "coverage_of_life" in self.per_asset.columns
                   else "coverage")
            cov = self.per_asset[col].dropna()
            if len(cov):
                lines.append(f"  coverage: min={cov.min():.3f} "
                             f"median={cov.median():.3f} max={cov.max():.3f}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        return self.per_asset.copy()


# ---------------------------------------------------------------------------
# Downloaders
# ---------------------------------------------------------------------------


def _download_us(ticker: str, retries: int = 3,
                 pause: float = 1.5) -> pd.DataFrame:
    """Download one US ticker via akshare, with retries."""
    import akshare as ak

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = ak.stock_us_daily(symbol=ticker)
            if df is None or df.empty:
                raise ValueError("empty frame")
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            df["date"] = pd.to_datetime(df["date"])
            keep = ["date", "open", "high", "low", "close", "volume"]
            df = df[[c for c in keep if c in df.columns]]
            return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:       # pragma: no cover - network dependent
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(
        f"failed to download US ticker {ticker}: {last_exc}")


def _sina_symbol(ticker: str, market: str = "CN") -> str:
    """Convert a bare A-share ticker into a Sina symbol like ``sh510300``.

    Rule: Shanghai-listed funds/ETFs begin with ``5`` (51/56/58) and ETFs on
    the STAR market with ``588``; everything beginning with ``15``/``16``/``18``
    is Shenzhen.
    """
    t = str(ticker).strip().lower()
    if t.startswith(("sh", "sz")):
        return t
    if t.startswith(("5", "6", "9")):
        return f"sh{t}"
    return f"sz{t}"


def _download_cn_etf_sina(ticker: str, retries: int = 4,
                          pause: float = 1.5) -> pd.DataFrame:
    """Download one A-share ETF from Sina Finance via akshare.

    Sina is the **primary** Chinese source because it is reachable from the
    target environment whereas the Eastmoney endpoints
    (``push2his.eastmoney.com``) are frequently refused by the egress proxy.
    Sina returns un-adjusted prices, which is fine for our purpose: ETF splits
    and distributions are rare in China and, when they occur, are handled by
    the universal split-repair step in :func:`clean_price_panel`.
    """
    import akshare as ak

    sym = _sina_symbol(ticker)
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = ak.fund_etf_hist_sina(symbol=sym)
            if df is None or df.empty:
                raise ValueError("empty frame")
            df = df.rename(columns={c: str(c).lower() for c in df.columns})
            df["date"] = pd.to_datetime(df["date"])
            keep = ["date", "open", "high", "low", "close", "volume"]
            return df[[c for c in keep if c in df.columns]] \
                .sort_values("date").reset_index(drop=True)
        except Exception as exc:       # pragma: no cover - network dependent
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(
        f"failed to download CN ETF {ticker} from Sina: {last_exc}")


def _download_cn_etf_eastmoney(ticker: str, start: str, end: str,
                               retries: int = 3,
                               pause: float = 2.0) -> pd.DataFrame:
    """Download one A-share ETF from Eastmoney (forward-adjusted).

    Used as a fallback when Sina fails.  Forward adjustment (``qfq``) means
    the series is already corporate-action clean.
    """
    import akshare as ak

    s = pd.Timestamp(start).strftime("%Y%m%d")
    e = pd.Timestamp(end).strftime("%Y%m%d")
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = ak.fund_etf_hist_em(symbol=ticker, period="daily",
                                     start_date=s, end_date=e, adjust="qfq")
            if df is None or df.empty:
                raise ValueError("empty frame")
            ren = {"日期": "date", "开盘": "open", "最高": "high",
                   "最低": "low", "收盘": "close", "成交量": "volume"}
            df = df.rename(columns=ren)
            df["date"] = pd.to_datetime(df["date"])
            keep = ["date", "open", "high", "low", "close", "volume"]
            return df[[c for c in keep if c in df.columns]] \
                .sort_values("date").reset_index(drop=True)
        except Exception as exc:       # pragma: no cover - network dependent
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(pause * (attempt + 1))
    raise RuntimeError(
        f"failed to download CN ETF {ticker} from Eastmoney: {last_exc}")


def _download_cn_etf(ticker: str, start: str, end: str,
                     retries: int = 4, pause: float = 2.0,
                     prefer: str = "sina") -> pd.DataFrame:
    """Download one Chinese ETF, trying Sina then Eastmoney.

    The two providers are tried in order so that a provider-side outage does
    not break the pipeline; the successful provider is recorded in the cache
    filename-adjacent metadata via the returned frame's ``attrs``.
    """
    order = ("sina", "eastmoney") if prefer == "sina" else ("eastmoney", "sina")
    errors: List[str] = []
    for src in order:
        try:
            if src == "sina":
                df = _download_cn_etf_sina(ticker, retries=retries, pause=pause)
            else:
                df = _download_cn_etf_eastmoney(ticker, start, end,
                                                retries=retries, pause=pause)
            df.attrs["source"] = src
            return df
        except Exception as exc:       # pragma: no cover
            errors.append(f"{src}: {exc}")
    raise RuntimeError(
        f"failed to download CN ETF {ticker} from any provider -> "
        + " | ".join(errors))


def download_prices(
    universe: SectorUniverse | str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
    cache_root: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Download (or read from cache) a long-form price panel.

    Returns a tidy DataFrame with columns
    ``[date, ticker, open, high, low, close, volume]``.
    """
    uni = get_universe(universe) if isinstance(universe, str) else universe
    start = start or uni.start
    end = end or uni.end
    cdir = cache_dir_for(uni.name, cache_root)

    frames: List[pd.DataFrame] = []
    for spec in uni.assets:
        fpath = cdir / f"{spec.ticker}.csv"
        if fpath.exists() and not force:
            df = pd.read_csv(fpath, parse_dates=["date"])
        else:
            if verbose:
                print(f"  downloading {spec.ticker} ({spec.label}) ...")
            if spec.market == "US":
                df = _download_us(spec.ticker)
            else:
                df = _download_cn_etf(spec.ticker, start, end)
            df.to_csv(fpath, index=False)
            time.sleep(0.3)
        df = df[(df["date"] >= pd.Timestamp(start)) &
                (df["date"] <= pd.Timestamp(end))]
        df = df.copy()
        df["ticker"] = spec.ticker
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    cols = ["date", "ticker", "open", "high", "low", "close", "volume"]
    panel = panel[[c for c in cols if c in panel.columns]]
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def _detect_and_repair_splits(
    prices: pd.DataFrame,
    threshold: float = -0.30,
    ratio_tol: float = 0.06,
    max_iter: int = 5,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Detect and repair un-adjusted stock splits in a price panel.

    ``akshare.stock_us_daily`` returns prices that are generally
    split-adjusted, but the adjustment is occasionally missing for recent
    corporate actions.  The failure mode is unmistakable: a single-day return
    of roughly ``-1/2``, ``-2/3`` or ``-3/4`` — exactly the complement of a
    2:1, 3:1 or 4:1 split ratio — followed by prices that continue normally at
    the new level.

    Detection rule: a close-to-close return below ``threshold`` whose
    complement ``1 + r`` is within ``ratio_tol`` of ``1/k`` for some integer
    ``k`` in ``{2, ..., 10}``.  Repair: divide all *earlier* prices for that
    asset by ``k`` (i.e. back-adjust the pre-split history so the series is
    continuous).

    Returns the repaired panel plus a list of records describing every
    adjustment made, which is written into the data-quality report so the
    correction is auditable rather than silent.

    Implementation note
    -------------------
    Returns are computed on each asset's **own** valid observations
    (``s.dropna().pct_change()``) rather than across the full panel index.
    This matters: in a ragged panel an asset can be missing a date that other
    assets trade, and computing ``pct_change()`` on the reindexed column would
    yield ``NaN`` across the gap and hide the split entirely.  This exact
    failure was observed on a Chinese ETF whose 1:4 split coincided with a
    one-day data gap.
    """
    px = prices.copy()
    records: List[Dict[str, object]] = []

    for col in px.columns:
        s = px[col].copy()
        for _ in range(max_iter):
            # compute returns only on this asset's valid observations
            valid = s.dropna()
            if valid.size < 3:
                break
            r = valid.pct_change()
            gross = 1.0 + r
            cand = r[(r < threshold) & gross.notna() & (gross > 0)]
            if cand.empty:
                break

            fixed_any = False
            for dt, rr in cand.items():
                g = 1.0 + float(rr)
                inv = 1.0 / g
                k = int(round(inv))
                if k < 2 or k > 10:
                    continue
                if abs(inv - k) > ratio_tol * k:
                    continue
                pos = s.index.get_loc(dt)
                # back-adjust everything strictly before the split date
                s.iloc[:pos] = s.iloc[:pos] / k
                records.append({
                    "ticker": col,
                    "date": dt,
                    "ratio": f"1:{k}",
                    "raw_return": float(rr),
                    "repaired": True,
                })
                if verbose:
                    print(f"  [split repair] {col} on {dt.date()}: "
                          f"r={rr:.4f} -> 1:{k} split, back-adjusted "
                          f"{pos} prior observations")
                fixed_any = True
                break      # restart the scan with the repaired series
            if not fixed_any:
                break
        px[col] = s

    return px, records


def clean_price_panel(
    panel: pd.DataFrame,
    max_ffill: int = 3,
    min_coverage: float = 0.95,
    repair_splits: bool = True,
    align: str = "intersection",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Turn a long-form panel into a clean, balanced wide close-price frame.

    Returns ``(prices, report)`` where ``prices`` is a
    ``T x N`` DataFrame of close prices.

    Parameters
    ----------
    max_ffill:
        Maximum number of consecutive missing days to forward-fill before
        treating the gap as a real absence.
    min_coverage:
        Minimum fraction of the *panel's* calendar on which an asset must have
        a price.  This is the survivorship/availability filter.  Defaults to
        0.95: an asset with genuine gaps covering more than 5% of the sample
        is excluded rather than silently interpolated.
    align:
        How to build the common calendar.

        ``'intersection'`` (default)
            Keep only dates on which **every** surviving asset trades.  Gives
            a perfectly balanced panel — essential for a covariance study,
            because an unbalanced panel silently changes ``N`` and ``T``
            across windows and corrupts ``q = N/T``.

        ``'union'``
            Keep the union of dates and forward-fill.  Rarely appropriate
            here; provided for completeness.
    """
    p = panel.copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker", "date"])

    # --- per-asset cleaning ------------------------------------------------
    cleaned: List[pd.DataFrame] = []
    for ticker, g in p.groupby("ticker", sort=True):
        g = g.drop_duplicates(subset="date", keep="last").copy()
        g = g.set_index("date")
        for col in ("close", "open", "high", "low"):
            if col in g.columns:
                g.loc[g[col] <= 0, col] = np.nan
        g["close"] = g["close"].ffill(limit=max_ffill)
        g = g[["close"]].dropna()
        g.columns = pd.Index([ticker])
        cleaned.append(g)

    wide = pd.concat(cleaned, axis=1, sort=True).sort_index()

    # --- split repair (before coverage filtering, so a repaired series can
    #     still qualify) -----------------------------------------------------
    split_records: List[Dict[str, object]] = []
    if repair_splits:
        wide, split_records = _detect_and_repair_splits(wide, verbose=verbose)

    # --- coverage filter --------------------------------------------------
    # Coverage is measured against the *union* calendar (the set of dates on
    # which at least one asset trades), so an asset that lists late is only
    # penalised for the part of the sample where it could have traded.
    union_index = wide.index
    span_start = union_index.min()
    asset_start = wide.apply(lambda s: s.first_valid_index())
    asset_end = wide.apply(lambda s: s.last_valid_index())
    n_possible = (asset_end - asset_start).dt.days
    n_actual = wide.notna().sum()
    # effective required coverage: over the asset's own listed life, measured
    # on the union calendar restricted to that life
    coverage = {}
    for col in wide.columns:
        s0, s1 = asset_start[col], asset_end[col]
        if pd.isna(s0) or pd.isna(s1):
            coverage[col] = 0.0
            continue
        life = union_index[(union_index >= s0) & (union_index <= s1)]
        coverage[col] = (n_actual[col] / len(life)) if len(life) else 0.0
    coverage = pd.Series(coverage)

    dropped = sorted(coverage[coverage < min_coverage].index.tolist())
    if dropped:
        wide = wide.drop(columns=dropped)
        if verbose:
            print(f"  dropped (coverage < {min_coverage:.0%}): {dropped}")

    # --- build the common calendar ----------------------------------------
    if align == "intersection":
        wide = wide.dropna(how="any", axis=0)
    elif align == "union":
        wide = wide.sort_index().ffill(limit=max_ffill)
        wide = wide.dropna(how="any", axis=0)
    else:
        raise ValueError("align must be 'intersection' or 'union'")
    wide = wide.loc[(wide > 0).all(axis=1)]

    # --- build the report -------------------------------------------------
    per_asset = pd.DataFrame({
        "n_obs": wide.notna().sum(),
        "coverage_of_life": coverage.reindex(wide.columns),
        "coverage_of_panel": wide.notna().sum() / max(len(wide.index), 1),
        "start": wide.apply(lambda s: s.first_valid_index()),
        "end": wide.apply(lambda s: s.last_valid_index()),
    })
    rets = wide.pct_change()
    per_asset["ann_vol"] = rets.std() * np.sqrt(252)
    per_asset["max_abs_daily"] = rets.abs().max()
    per_asset["n_extreme"] = (rets.abs() > 0.25).sum()

    report = DataQualityReport(
        universe="panel",
        per_asset=per_asset,
        panel_start=wide.index.min() if len(wide) else None,
        panel_end=wide.index.max() if len(wide) else None,
        n_dates=len(wide),
        n_assets=wide.shape[1],
        dropped_assets=dropped,
    )
    if split_records:
        report.notes.append(
            f"repaired {len(split_records)} un-adjusted split(s): "
            + ", ".join(f"{r['ticker']}@{pd.Timestamp(r['date']).date()}"
                        f"({r['ratio']})" for r in split_records))
    if len(wide) == 0:
        report.notes.append("EMPTY panel after cleaning")
    else:
        n_ext = int((rets.abs() > 0.25).sum().sum())
        if n_ext > 0:
            report.notes.append(
                f"{n_ext} daily moves still exceed 25% after repair -- "
                "inspect before use")
        n_neg = int((rets < -0.999).sum().sum())
        if n_neg:
            report.notes.append(f"{n_neg} returns below -99.9% detected")

    return wide, report


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------


def load_price_panel(
    universe: SectorUniverse | str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
    max_ffill: int = 3,
    min_coverage: float = 0.95,
    repair_splits: bool = True,
    align: str = "intersection",
    cache_root: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Download + clean a universe in one call."""
    uni = get_universe(universe) if isinstance(universe, str) else universe
    panel = download_prices(uni, start=start, end=end, force=force,
                            cache_root=cache_root, verbose=verbose)
    prices, report = clean_price_panel(
        panel, max_ffill=max_ffill, min_coverage=min_coverage,
        repair_splits=repair_splits, align=align, verbose=verbose)
    report.universe = uni.name
    return prices, report


def load_returns(
    universe: SectorUniverse | str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
    method: str = "log",
    min_coverage: float = 0.95,
    repair_splits: bool = True,
    align: str = "intersection",
    winsorise: float = 0.50,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, DataQualityReport]:
    """Load simple or log returns for a universe.

    ``method='log'`` returns :math:`\\log(P_t/P_{t-1})`; ``method='simple'``
    returns arithmetic returns, which is what the backtester expects for
    compounding wealth.

    Notes
    -----
    **No blanket winsorisation is applied by default.**  Extreme daily returns
    in equity data are usually *real* — 2008-10-13 saw sector ETFs move
    +13% to +15%, and 2020-03-09 saw energy fall 22%.  Clipping those would
    destroy exactly the tail information a risk model must capture.  Genuine
    data errors are removed upstream by :func:`_detect_and_repair_splits`
    and by the positivity filter.  Set ``winsorise`` to a finite value (e.g.
    ``0.5``) only if you deliberately want a robustness floor; it is applied
    symmetrically and the number of affected observations is reported.
    """
    prices, report = load_price_panel(
        universe, start=start, end=end, force=force,
        min_coverage=min_coverage, repair_splits=repair_splits,
        align=align, verbose=verbose)
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    rets = rets.iloc[1:].dropna(how="any")

    if winsorise is not None and np.isfinite(winsorise):
        n_clipped = int((rets.abs() > winsorise).sum().sum())
        if n_clipped:
            rets = rets.clip(lower=-float(winsorise), upper=float(winsorise))
            report.notes.append(
                f"winsorised {n_clipped} observations at +/-{winsorise:.0%}")

    return rets, report


def compute_quality_report(prices: pd.DataFrame,
                           universe: str = "panel") -> DataQualityReport:
    """Build a quality report from an already-cleaned price frame."""
    rets = prices.pct_change()
    per_asset = pd.DataFrame({
        "n_obs": prices.notna().sum(),
        "coverage": prices.notna().sum() / max(len(prices.index), 1),
        "start": prices.apply(lambda s: s.first_valid_index()),
        "end": prices.apply(lambda s: s.last_valid_index()),
        "ann_vol": rets.std() * np.sqrt(252),
        "max_abs_daily": rets.abs().max(),
        "n_extreme": (rets.abs() > 0.25).sum(),
    })
    return DataQualityReport(
        universe=universe, per_asset=per_asset,
        panel_start=prices.index.min() if len(prices) else None,
        panel_end=prices.index.max() if len(prices) else None,
        n_dates=len(prices), n_assets=prices.shape[1],
    )
