"""Universe definitions: curated baskets of liquid, sector-diverse assets.

Two universes are shipped:

``us_sector_etf``
    The eleven SPDR Select Sector ETFs (XLK, XLF, XLE, XLV, XLI, XLP, XLU,
    XLY, XLB, XLRE, XLC) plus two broad-market/defensive complements (SPY,
    TLT) and a real-asset sleeve (GLD, IWM).  Sixteen assets with deep
    liquidity and no survivorship issues; the natural first test bed because
    the sector taxonomy is designed to be low-correlation *between* groups.

``cn_sector_etf``
    Liquid Chinese sector and style ETFs traded on the Shanghai/Shenzhen
    exchanges, covering broad index, large/mid/small cap, tech, healthcare,
    consumer, financials, new energy, defence, and dividends.  Intended as the
    second market for the cross-market robustness test.  Data comes from
    Eastmoney via ``akshare``, so it starts in the mid-2010s when these ETFs
    reached reliable trading volume.

``us_single_stock``
    A high-dimensional stress test: 30 large-cap US single names spanning all
    eleven sectors.  With ``N = 30`` and ``T = 252`` this gives ``q ≈ 0.12``;
    shrinking ``T`` to 120 pushes ``q`` to ``0.25``, which is where the noise
    band is wide enough to materially affect optimisation.  Used for the
    ``N/T`` sensitivity analysis.

Each universe records both the ticker and a human label, so plots and tables
can be readable without a separate lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = [
    "AssetSpec",
    "SectorUniverse",
    "US_SECTOR_ETF",
    "CN_SECTOR_ETF",
    "US_SINGLE_STOCK",
    "UNIVERSES",
    "get_universe",
]


@dataclass(frozen=True)
class AssetSpec:
    """A single tradable instrument plus the metadata needed for grouping."""

    ticker: str
    label: str
    sector: str = ""
    market: str = "US"

    def __str__(self) -> str:      # pragma: no cover - cosmetic
        return f"{self.ticker} ({self.label})"


@dataclass
class SectorUniverse:
    """A named collection of assets with grouping metadata."""

    name: str
    assets: List[AssetSpec]
    start: str = "2005-01-01"
    end: str = "2025-12-31"
    description: str = ""
    source: str = "akshare"

    @property
    def tickers(self) -> List[str]:
        return [a.ticker for a in self.assets]

    @property
    def labels(self) -> List[str]:
        return [f"{a.ticker}" for a in self.assets]

    @property
    def sector_map(self) -> Dict[str, str]:
        return {a.ticker: a.sector for a in self.assets}

    @property
    def label_map(self) -> Dict[str, str]:
        return {a.ticker: a.label for a in self.assets}

    @property
    def groups(self) -> Dict[str, List[str]]:
        """Map sector name -> list of tickers."""
        out: Dict[str, List[str]] = {}
        for a in self.assets:
            out.setdefault(a.sector or "other", []).append(a.ticker)
        return out

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def __len__(self) -> int:
        return len(self.assets)

    def describe(self) -> str:
        return (f"{self.name}: {self.n_assets} assets, "
                f"{len(self.groups)} groups, {self.start}..{self.end} "
                f"({self.source})")


# ---------------------------------------------------------------------------
# US sector ETF universe
# ---------------------------------------------------------------------------

US_SECTOR_ETF = SectorUniverse(
    name="us_sector_etf",
    description=(
        "Eleven SPDR Select Sector ETFs plus broad-market, rates and "
        "real-asset complements. The sector taxonomy is deliberately built to "
        "have low between-group correlation, which makes the hierarchical "
        "structure that HRP exploits easy to identify."
    ),
    start="2005-01-01",
    end="2025-12-31",
    source="akshare",
    assets=[
        AssetSpec("SPY", "S&P 500 broad market", "broad_market"),
        AssetSpec("XLK", "Technology", "technology"),
        AssetSpec("XLF", "Financials", "financials"),
        AssetSpec("XLE", "Energy", "energy"),
        AssetSpec("XLV", "Health Care", "healthcare"),
        AssetSpec("XLI", "Industrials", "industrials"),
        AssetSpec("XLP", "Consumer Staples", "staples"),
        AssetSpec("XLU", "Utilities", "utilities"),
        AssetSpec("XLY", "Consumer Discretionary", "discretionary"),
        AssetSpec("XLB", "Materials", "materials"),
        AssetSpec("XLRE", "Real Estate", "real_estate"),
        AssetSpec("XLC", "Communication Services", "communication"),
        AssetSpec("TLT", "20+ Year Treasuries", "rates"),
        AssetSpec("IEF", "7-10 Year Treasuries", "rates"),
        AssetSpec("GLD", "Gold", "commodities"),
        AssetSpec("IWM", "Russell 2000 small cap", "broad_market"),
    ],
)


# ---------------------------------------------------------------------------
# China sector ETF universe
# ---------------------------------------------------------------------------

CN_SECTOR_ETF = SectorUniverse(
    name="cn_sector_etf",
    description=(
        "Liquid Chinese sector, style and thematic ETFs listed in Shanghai "
        "and Shenzhen. Data is sourced from Eastmoney (akshare) with forward "
        "adjustment ('qfq'). The universe spans broad index, size styles, "
        "technology, healthcare, consumption, financials, new energy, "
        "defence and dividend strategies."
    ),
    start="2015-01-01",
    end="2025-12-31",
    source="akshare",
    assets=[
        AssetSpec("510300", "CSI 300 (Huatai-PB)", "broad_index", "CN"),
        AssetSpec("510500", "CSI 500 (Southern)", "mid_small", "CN"),
        AssetSpec("159915", "ChiNext (E Fund)", "growth", "CN"),
        AssetSpec("588000", "STAR 50 (Bosera)", "technology", "CN"),
        AssetSpec("512760", "Semiconductor (Guolian'an)", "technology", "CN"),
        AssetSpec("515000", "Technology Leaders (Huabao)", "technology", "CN"),
        AssetSpec("512010", "Pharma & Biotech (E Fund)", "healthcare", "CN"),
        AssetSpec("159928", "Consumer Staples (Huabao)", "staples", "CN"),
        AssetSpec("159996", "Household Appliances (Tianhong)", "discretionary", "CN"),
        AssetSpec("512800", "Banking (Huabao)", "financials", "CN"),
        AssetSpec("512880", "Securities (Guotai)", "financials", "CN"),
        AssetSpec("512580", "New Energy (Guotai)", "energy", "CN"),
        AssetSpec("515790", "Photovoltaics (Huatai-PB)", "energy", "CN"),
        AssetSpec("512660", "Defence & Military (Guotai)", "industrials", "CN"),
        AssetSpec("512400", "Non-ferrous Metals (Southern)", "materials", "CN"),
        AssetSpec("515180", "CSI Dividend (E Fund)", "dividend", "CN"),
    ],
)


# ---------------------------------------------------------------------------
# US single-stock stress-test universe
# ---------------------------------------------------------------------------

US_SINGLE_STOCK = SectorUniverse(
    name="us_single_stock",
    description=(
        "Thirty large-cap US single names spanning all eleven GICS sectors. "
        "Used for the high-dimensional stress test: with N = 30 a 252-day "
        "window gives q = 0.12, and shortening the window to 120 days pushes "
        "q to 0.25 -- squarely in the regime where random-matrix effects "
        "dominate the small eigenvalues."
    ),
    start="2005-01-01",
    end="2025-12-31",
    source="akshare",
    assets=[
        # technology
        AssetSpec("AAPL", "Apple", "technology"),
        AssetSpec("MSFT", "Microsoft", "technology"),
        AssetSpec("NVDA", "NVIDIA", "technology"),
        AssetSpec("ORCL", "Oracle", "technology"),
        # communication
        AssetSpec("GOOGL", "Alphabet", "communication"),
        AssetSpec("META", "Meta Platforms", "communication"),
        # discretionary
        AssetSpec("AMZN", "Amazon", "discretionary"),
        AssetSpec("HD", "Home Depot", "discretionary"),
        AssetSpec("MCD", "McDonald's", "discretionary"),
        # staples
        AssetSpec("PG", "Procter & Gamble", "staples"),
        AssetSpec("KO", "Coca-Cola", "staples"),
        AssetSpec("WMT", "Walmart", "staples"),
        # financials
        AssetSpec("JPM", "JPMorgan Chase", "financials"),
        AssetSpec("BAC", "Bank of America", "financials"),
        AssetSpec("GS", "Goldman Sachs", "financials"),
        # healthcare
        AssetSpec("JNJ", "Johnson & Johnson", "healthcare"),
        AssetSpec("UNH", "UnitedHealth", "healthcare"),
        AssetSpec("PFE", "Pfizer", "healthcare"),
        # energy
        AssetSpec("XOM", "Exxon Mobil", "energy"),
        AssetSpec("CVX", "Chevron", "energy"),
        # industrials
        AssetSpec("CAT", "Caterpillar", "industrials"),
        AssetSpec("GE", "GE Aerospace", "industrials"),
        AssetSpec("UPS", "United Parcel Service", "industrials"),
        # materials
        AssetSpec("LIN", "Linde", "materials"),
        AssetSpec("FCX", "Freeport-McMoRan", "materials"),
        # utilities
        AssetSpec("NEE", "NextEra Energy", "utilities"),
        AssetSpec("DUK", "Duke Energy", "utilities"),
        # real estate
        AssetSpec("PLD", "Prologis", "real_estate"),
        AssetSpec("SPG", "Simon Property Group", "real_estate"),
        # tech/semis
        AssetSpec("AMD", "Advanced Micro Devices", "technology"),
    ],
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

UNIVERSES: Dict[str, SectorUniverse] = {
    u.name: u for u in (US_SECTOR_ETF, CN_SECTOR_ETF, US_SINGLE_STOCK)
}


def get_universe(name: str) -> SectorUniverse:
    """Look up a universe by name."""
    if name not in UNIVERSES:
        raise KeyError(
            f"unknown universe {name!r}; available: {sorted(UNIVERSES)}")
    return UNIVERSES[name]
