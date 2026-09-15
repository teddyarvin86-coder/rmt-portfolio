"""Shared utilities: logging, reproducibility, IO and plotting style."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from rmt_portfolio.config import PROJECT_ROOT, Config

__all__ = [
    "get_logger",
    "set_global_seed",
    "Timer",
    "ensure_dir",
    "save_table",
    "save_json",
    "load_json",
    "apply_plot_style",
    "palette",
    "cv_summary",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_ROOT_LOGGER_NAME = "rmt_portfolio"
_LOG_CONFIGURED = False


def get_logger(name: str = _ROOT_LOGGER_NAME, level: int = logging.INFO
               ) -> logging.Logger:
    """Return a configured logger, correctly namespaced under the package.

    A bare name such as ``"spectrum"`` is automatically prefixed to
    ``"rmt_portfolio.spectrum"`` so that it inherits the package logger's
    level and handlers.  (Without the prefix the child would sit under the
    stdlib root logger at WARNING and silently swallow every INFO record --
    a subtle failure mode worth avoiding.)
    """
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S"))
        pkg_logger = logging.getLogger(_ROOT_LOGGER_NAME)
        pkg_logger.handlers[:] = [handler]
        pkg_logger.setLevel(level)
        pkg_logger.propagate = False
        _LOG_CONFIGURED = True

    full = name if name.startswith(_ROOT_LOGGER_NAME) else \
        f"{_ROOT_LOGGER_NAME}.{name}"
    logger = logging.getLogger(full)
    logger.setLevel(logging.NOTSET)          # inherit from the package logger
    return logger


def set_log_level(level: int | str) -> None:
    """Change the verbosity of the whole package at runtime."""
    get_logger()
    logging.getLogger(_ROOT_LOGGER_NAME).setLevel(level)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int = 20240915) -> None:
    """Seed every RNG that the pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class Timer:
    """Context manager that logs elapsed wall-clock time.

    >>> with Timer("eigen sweep"):
    ...     pass                                       # doctest: +SKIP
    """

    def __init__(self, label: str, logger: Optional[logging.Logger] = None,
                 level: int = logging.INFO) -> None:
        self.label = label
        self.logger = logger or get_logger()
        self.level = level
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._t0
        self.logger.log(self.level, "%s finished in %.2fs", self.label,
                        self.elapsed)


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    t0 = time.perf_counter()
    yield
    (logger or get_logger()).info("%s finished in %.2fs", label,
                                  time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def ensure_dir(path: str | os.PathLike | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_table(
    df: pd.DataFrame,
    path: str | os.PathLike | Path,
    index: bool = True,
    float_format: str = "%.6f",
) -> Path:
    """Save a DataFrame as CSV and, alongside it, a Markdown version.

    The companion ``.md`` file uses ``DataFrame.to_markdown`` when the
    optional ``tabulate`` dependency is importable, otherwise a hand-rolled
    pipe table is written.  Having the Markdown next to the CSV makes it
    trivial to paste results tables straight into the working paper.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=index, float_format=float_format)
        md_path = p.with_suffix(".md")
        try:
            md_path.write_text(df.to_markdown(index=index),
                               encoding="utf-8")
        except Exception:
            md_path.write_text(_to_pipe_table(df, index=index),
                               encoding="utf-8")
    elif p.suffix.lower() in (".md", ".markdown"):
        p.write_text(_to_pipe_table(df, index=index), encoding="utf-8")
    elif p.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(p, index=index)
    else:
        df.to_csv(p, index=index, float_format=float_format)
    return p


def _to_pipe_table(df: pd.DataFrame, index: bool = True) -> str:
    """Minimal Markdown pipe-table renderer (no external dependency)."""
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in d.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.6f}" if np.isfinite(v) else "")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_json(obj: Any, path: str | os.PathLike | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return o.isoformat()
        if isinstance(o, Path):
            return str(o)
        if isinstance(o, Config):
            return o.to_dict()
        return str(o)

    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                            default=default), encoding="utf-8")
    return p


def load_json(path: str | os.PathLike | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------

#: A colour-blind-safe qualitative palette (Okabe-Ito plus extensions),
#: chosen so that up to twelve strategies stay distinguishable in print.
_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
    "#8C564B",  # brown
    "#7F7F7F",  # grey
    "#17BECF",  # cyan
    "#BCBD22",  # olive
]


def palette(n: Optional[int] = None) -> list:
    """Return ``n`` colours from the project palette (cycling if needed)."""
    if n is None:
        return list(_PALETTE)
    out = []
    for i in range(n):
        out.append(_PALETTE[i % len(_PALETTE)])
    return out


#: Consistent colour per estimator, so the same strategy keeps the same
#: colour in every figure.  This is essential for a paper with many plots.
ESTIMATOR_COLORS: Dict[str, str] = {
    "sample": "#7F7F7F",
    "ledoit_wolf": "#009E73",
    "constant_corr": "#56B4E9",
    "rmt_hard": "#D55E00",
    "rmt_soft": "#E69F00",
    "rmt_rie": "#0072B2",
    "factor_model": "#CC79A7",
}

#: Consistent line style per allocator.
ALLOCATOR_STYLES: Dict[str, str] = {
    "equal_weight": ":",
    "inverse_variance": "-.",
    "minimum_variance": "-",
    "risk_parity": "--",
    "hrp": "-",
    "max_diversification": (0, (3, 1, 1, 1)),
}


def apply_plot_style(style: str = "seaborn-v0_8-whitegrid",
                     font_scale: float = 1.05,
                     dpi: int = 200) -> None:
    """Apply the project's Matplotlib defaults.

    Falls back gracefully if the requested style is unavailable in the
    installed Matplotlib version.
    """
    import matplotlib
    matplotlib.use("Agg")            # headless-safe
    import matplotlib.pyplot as plt

    try:
        plt.style.use(style)
    except Exception:
        try:
            plt.style.use("seaborn-whitegrid")
        except Exception:
            plt.style.use("default")

    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10 * font_scale,
        "axes.titlesize": 12 * font_scale,
        "axes.labelsize": 11 * font_scale,
        "legend.fontsize": 9 * font_scale,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.autolayout": False,
    })


# ---------------------------------------------------------------------------
# Small statistics helpers
# ---------------------------------------------------------------------------


def cv_summary(series: np.ndarray) -> Dict[str, float]:
    """Return mean/std/CV/min/max for a numeric series."""
    s = np.asarray(series, dtype=float).ravel()
    s = s[np.isfinite(s)]
    if s.size == 0:
        return {"mean": np.nan, "std": np.nan, "cv": np.nan,
                "min": np.nan, "max": np.nan, "n": 0}
    m = float(np.mean(s))
    sd = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
    return {
        "mean": m,
        "std": sd,
        "cv": (sd / abs(m)) if m != 0 else np.nan,
        "min": float(np.min(s)),
        "max": float(np.max(s)),
        "n": int(s.size),
    }
