"""Market price data via yfinance, with a local cache and ETF fallback.

We only need daily closes for the three assets to run the event study, so this
module stays deliberately thin: fetch, cache to CSV, hand back a tidy frame.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from . import config
from .utils import get_logger

log = get_logger(__name__)


def _download_one(ticker: str, start: str, end: str) -> pd.Series | None:
    """Download one ticker's daily close. Returns None on empty/failed pull."""
    import yfinance as yf

    try:
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as exc:  # network / ticker errors -> caller tries the proxy
        log.warning("yfinance failed for %s: %s", ticker, exc)
        return None
    if raw is None or raw.empty:
        return None
    close = raw["Close"]
    # yfinance sometimes returns a single-column DataFrame instead of a Series.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    return close if not close.empty else None


def get_prices(
    start: str,
    end: str,
    assets: dict | None = None,
    cache_path: Path | str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a long-format price frame: columns ``date``, ``asset``, ``close``.

    For each asset we try the futures ticker first and fall back to the ETF
    proxy if the futures pull comes back empty (common on the free endpoint).
    """
    assets = assets or config.ASSETS
    cache_path = Path(cache_path) if cache_path else config.PRICES_CSV

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        have = set(cached["asset"].unique())
        if set(assets) <= have:
            lo, hi = cached["date"].min(), cached["date"].max()
            if lo <= pd.Timestamp(start) and hi >= pd.Timestamp(end) - pd.Timedelta(days=5):
                log.info("Using cached prices (%s)", cache_path.name)
                return cached

    frames = []
    for asset, meta in assets.items():
        series = _download_one(meta["ticker"], start, end)
        used = meta["ticker"]
        if series is None and meta.get("proxy"):
            log.info("Falling back to proxy %s for %s", meta["proxy"], asset)
            series = _download_one(meta["proxy"], start, end)
            used = meta["proxy"]
        if series is None:
            log.warning("No price data for %s (tried %s / %s)", asset,
                        meta["ticker"], meta.get("proxy"))
            continue
        df = series.rename("close").reset_index()
        df.columns = ["date", "close"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df["asset"] = asset
        df["ticker_used"] = used
        frames.append(df)
        log.info("Fetched %d closes for %s via %s", len(df), asset, used)

    if not frames:
        raise RuntimeError("Could not fetch any price data. Check your connection.")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["asset", "date"]).reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(cache_path, index=False)
    log.info("Cached prices -> %s", cache_path)
    return prices


def default_date_range(headlines: pd.DataFrame, pad_days: int = 10) -> tuple[str, str]:
    """Derive a fetch window that covers the headlines plus a forward pad."""
    start = (headlines["date"].min() - pd.Timedelta(days=pad_days)).date()
    end = (headlines["date"].max() + pd.Timedelta(days=pad_days)).date()
    # Never ask yfinance for the future.
    today = dt.date.today()
    end = min(end, today)
    return start.isoformat(), end.isoformat()
