"""Central configuration: paths, asset universe, and keyword -> asset routing.

Everything the pipeline needs to know about *which* headlines map to *which*
market lives here, so the routing logic is auditable in one place.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = ROOT / "sample_data"

MACRO_EVENTS_CSV = DATA_DIR / "macro_events.csv"
SAMPLE_HEADLINES_CSV = SAMPLE_DIR / "sample_headlines.csv"

# Pipeline outputs (written by scripts/run_pipeline.py)
SCORED_HEADLINES_CSV = PROCESSED_DIR / "scored_headlines.csv"
DAILY_SENTIMENT_CSV = PROCESSED_DIR / "daily_sentiment.csv"
PRICES_CSV = PROCESSED_DIR / "prices.csv"
EVENT_STUDY_CSV = PROCESSED_DIR / "event_study.csv"
REACTION_SUMMARY_CSV = PROCESSED_DIR / "reaction_summary.csv"

for _d in (RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"

# ---------------------------------------------------------------------------
# Asset universe.  yfinance tickers for the futures the doc calls out, plus an
# ETF proxy that is more reliably available on the free yfinance endpoint.
# ---------------------------------------------------------------------------
ASSETS = {
    "crude": {
        "name": "WTI Crude Oil",
        "ticker": "CL=F",       # WTI crude futures
        "proxy": "USO",         # United States Oil Fund ETF (fallback)
    },
    "gold": {
        "name": "Gold",
        "ticker": "GC=F",       # COMEX gold futures
        "proxy": "GLD",         # SPDR Gold Shares ETF (fallback)
    },
    "bonds": {
        "name": "US 10Y Treasuries",
        "ticker": "ZN=F",       # 10-Year T-Note futures
        "proxy": "TLT",         # 20+ Year Treasury ETF (fallback)
    },
}

# ---------------------------------------------------------------------------
# Keyword -> asset routing.  A headline is routed to an asset if it contains any
# of that asset's keywords (case-insensitive, word-ish match).  A single
# headline can map to several assets (e.g. a Fed decision moves gold AND bonds).
# ---------------------------------------------------------------------------
ASSET_KEYWORDS = {
    "crude": [
        "opec", "opec+", "crude", "oil", "wti", "brent", "barrel",
        "petroleum", "shale", "refinery", "energy prices",
    ],
    "gold": [
        "gold", "bullion", "xau", "safe haven", "safe-haven", "precious metal",
    ],
    "bonds": [
        "treasur", "bond", "yield", "10-year", "10 year", "coupon",
        "fixed income",
    ],
}

# Macro themes that move gold AND bonds even when the metal/bond isn't named.
# These fire off the *event*, not the instrument.
MACRO_THEME_KEYWORDS = {
    "monetary_policy": [
        "fed", "fomc", "federal reserve", "powell", "rate hike", "rate cut",
        "interest rate", "hawkish", "dovish", "tightening", "easing",
        "quantitative", "basis points", "bps",
    ],
    "inflation": [
        "cpi", "inflation", "ppi", "pce", "core prices", "disinflation",
    ],
}
# Themes -> the assets they are assumed to move.
THEME_TO_ASSETS = {
    "monetary_policy": ["gold", "bonds"],
    "inflation": ["gold", "bonds"],
}

# ---------------------------------------------------------------------------
# Event-study parameters
# ---------------------------------------------------------------------------
FORWARD_WINDOWS = [1, 3, 5]          # trading days after the event
HIGH_CONVICTION_THRESHOLD = 0.60     # |sentiment_score| above this = high conviction
