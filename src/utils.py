"""Small shared helpers: logging, headline loading, and keyword routing."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from . import config


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a sensible default format (configured once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


log = get_logger(__name__)


def load_headlines(path: Path | str | None = None) -> pd.DataFrame:
    """Load a headlines CSV into a tidy DataFrame.

    Falls back to the bundled sample data so the repo runs out of the box.
    Expected columns: ``date``, ``headline``.  ``source`` is optional.
    """
    if path is None:
        path = config.SAMPLE_HEADLINES_CSV
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Headlines file not found: {path}. "
            f"Run scripts to fetch news, or use the bundled sample_data/."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"date", "headline"} - set(df.columns)
    if missing:
        raise ValueError(f"Headlines file is missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 0].copy()
    if "source" not in df.columns:
        df["source"] = "unknown"
    df = df.dropna(subset=["date", "headline"]).reset_index(drop=True)
    log.info("Loaded %d headlines from %s (%s to %s)",
             len(df), path.name, df["date"].min().date(), df["date"].max().date())
    return df


def _contains_any(text: str, keywords: list[str]) -> bool:
    """Case-insensitive substring match with a light word boundary on short tokens."""
    text = text.lower()
    for kw in keywords:
        kw = kw.lower()
        # For very short tokens (<= 3 chars) require a word boundary to avoid
        # matching e.g. "oil" inside "toil".  Longer tokens use plain substring.
        if len(kw) <= 3:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                return True
        elif kw in text:
            return True
    return False


def route_headline(headline: str) -> list[str]:
    """Map a single headline to the list of assets it plausibly moves.

    Combines direct instrument keywords (``ASSET_KEYWORDS``) with macro-theme
    routing (``THEME_TO_ASSETS``) so a Fed headline hits gold + bonds even when
    neither is named explicitly.
    """
    assets: set[str] = set()

    for asset, keywords in config.ASSET_KEYWORDS.items():
        if _contains_any(headline, keywords):
            assets.add(asset)

    for theme, keywords in config.MACRO_THEME_KEYWORDS.items():
        if _contains_any(headline, keywords):
            assets.update(config.THEME_TO_ASSETS[theme])

    return sorted(assets)


def explode_by_asset(df: pd.DataFrame) -> pd.DataFrame:
    """Add an ``asset`` column, one row per (headline, matched asset).

    Headlines that match no asset are dropped (they carry no tradeable signal
    for our three markets).
    """
    df = df.copy()
    df["assets"] = df["headline"].apply(route_headline)
    df = df[df["assets"].map(len) > 0].copy()
    df = df.explode("assets").rename(columns={"assets": "asset"})
    return df.reset_index(drop=True)
