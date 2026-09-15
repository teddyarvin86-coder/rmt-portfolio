"""Configuration objects for the whole pipeline.

The project is driven by a single YAML file (``config/config.yaml``) that is
parsed into a nested, attribute-accessible :class:`Config` object.  Every
experiment therefore becomes fully reproducible from the config plus the
cached price data.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# ---------------------------------------------------------------------------
# Attribute-accessible nested config
# ---------------------------------------------------------------------------


class Config(dict):
    """A dict subclass that allows attribute access and nested merging.

    ``cfg.backtest.window`` works as well as ``cfg["backtest"]["window"]``.
    Nested dictionaries are recursively converted to :class:`Config`.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        for key, value in list(self.items()):
            self[key] = self._wrap(value)

    # -- helpers ----------------------------------------------------------

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, Config):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(v) for v in value]
        return value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(f"Config has no key '{item}'") from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def __delattr__(self, item: str) -> None:  # pragma: no cover
        try:
            del self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` (deep) copy of the configuration."""
        return copy.deepcopy(_plain(self))

    def dump(self, path: Optional[os.PathLike | str] = None) -> None:
        """Write the configuration back to YAML (for run provenance)."""
        target = Path(path) if path else DEFAULT_CONFIG_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, allow_unicode=True,
                           sort_keys=False, default_flow_style=False)

    # -- resolution helpers ----------------------------------------------

    def resolve(self, *parts: str) -> Path:
        """Resolve a path relative to the project root."""
        rel = os.path.join(*parts) if parts else ""
        p = Path(rel)
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def path_attr(self, dotted: str) -> Path:
        """Resolve ``a.b.c`` from the config into an absolute path."""
        node: Any = self
        for part in dotted.split("."):
            node = node[part]
        p = Path(str(node))
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def merge(self, other: Dict[str, Any]) -> "Config":
        """Recursively merge ``other`` into a copy of this config."""
        merged = _deep_merge(self.to_dict(), _plain(other))
        return Config(merged)


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: Optional[os.PathLike | str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load the YAML config, optionally applying ``overrides``.

    Parameters
    ----------
    path:
        Path to the YAML file.  Defaults to ``config/config.yaml``.
    overrides:
        Nested dictionary merged on top of the file contents.  Useful for
        sweeping parameters inside experiment scripts.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    cfg = Config(data)
    if overrides:
        cfg = cfg.merge(overrides)
    return cfg


# ---------------------------------------------------------------------------
# Universe definition (a convenience dataclass for asset metadata)
# ---------------------------------------------------------------------------


@dataclass
class Universe:
    """A named basket of tradable tickers."""

    name: str
    tickers: List[str]
    market: str = "US"
    start: str = "2005-01-01"
    end: str = "2024-12-31"
    description: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "tickers": list(self.tickers),
            "market": self.market,
            "start": self.start,
            "end": self.end,
            "description": self.description,
        }


__all__ = ["Config", "Universe", "load_config", "PROJECT_ROOT",
           "DEFAULT_CONFIG_PATH"]
