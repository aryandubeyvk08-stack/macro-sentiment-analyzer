"""Event study: did the market move the way the headline's sentiment implied?

Given (a) headlines scored by FinBERT and routed to assets, and (b) daily
closes per asset, we measure forward returns over 1/3/5 trading-day windows and
compute the directional hit-rate on high-conviction events.

Methodology note (say this in the interview): a headline stamped on day *t* is
matched to the last available close on-or-before *t*, and the forward return is
measured from that close to the close *N trading days later*. Using close-to-
close (not the pre-headline open) keeps the entry realistic — you can't trade a
headline before it prints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .utils import get_logger

log = get_logger(__name__)


def _forward_returns_for_asset(events: pd.DataFrame, prices: pd.DataFrame,
                               windows: list[int]) -> pd.DataFrame:
    """Attach forward returns to each event for a single asset.

    ``events`` and ``prices`` are already filtered to one asset.
    """
    prices = prices.sort_values("date").reset_index(drop=True)
    price_dates = prices["date"].values
    closes = prices["close"].values

    out_rows = []
    for _, ev in events.iterrows():
        # Index of the last trading day on-or-before the headline date.
        pos = np.searchsorted(price_dates, np.datetime64(ev["date"]), side="right") - 1
        if pos < 0:
            continue  # headline predates our price history
        base_close = closes[pos]
        row = ev.to_dict()
        row["entry_date"] = pd.Timestamp(price_dates[pos])
        row["entry_close"] = float(base_close)
        for w in windows:
            fwd = pos + w
            if fwd < len(closes):
                row[f"ret_{w}d"] = float(closes[fwd] / base_close - 1.0)
            else:
                row[f"ret_{w}d"] = np.nan
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def run_event_study(scored: pd.DataFrame, prices: pd.DataFrame,
                    windows: list[int] | None = None) -> pd.DataFrame:
    """Compute forward returns for every (headline, asset) event.

    Parameters
    ----------
    scored : DataFrame with columns date, asset, sentiment_score, label, headline
    prices : long-format price frame from :func:`market_data.get_prices`
    """
    windows = windows or config.FORWARD_WINDOWS
    frames = []
    for asset in scored["asset"].unique():
        ev = scored[scored["asset"] == asset]
        px = prices[prices["asset"] == asset]
        if px.empty:
            log.warning("No prices for asset %s; skipping %d events", asset, len(ev))
            continue
        frames.append(_forward_returns_for_asset(ev, px, windows))

    if not frames:
        raise RuntimeError("Event study produced no rows — no price/asset overlap.")

    study = pd.concat(frames, ignore_index=True)

    # Directional hypothesis: positive sentiment -> price up, negative -> down.
    # `hit_Nd` is 1.0 when the realized move agrees with the sentiment sign,
    # 0.0 when it disagrees, and NaN when the event can't be scored (neutral
    # sentiment or no forward price).  Floats let hit_rate = mean(hit) directly.
    for w in windows:
        agree = np.sign(study[f"ret_{w}d"]) == np.sign(study["sentiment_score"])
        hit = agree.astype(float)
        invalid = (study["sentiment_score"].abs() < 1e-9) | study[f"ret_{w}d"].isna()
        hit[invalid] = np.nan
        study[f"hit_{w}d"] = hit

    study["conviction"] = study["sentiment_score"].abs()
    study["high_conviction"] = study["conviction"] >= config.HIGH_CONVICTION_THRESHOLD
    return study


def _bucket(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def summarize(study: pd.DataFrame, windows: list[int] | None = None) -> pd.DataFrame:
    """Aggregate the event study into a reaction table.

    One row per (asset, sentiment_bucket, window) with average forward move,
    hit-rate, event count, and a one-sample t-stat on the mean return.
    """
    windows = windows or config.FORWARD_WINDOWS
    study = study.copy()
    study["bucket"] = study["sentiment_score"].apply(_bucket)

    rows = []
    for asset, g_asset in study.groupby("asset"):
        for bucket, g in g_asset.groupby("bucket"):
            for w in windows:
                rets = g[f"ret_{w}d"].dropna()
                hits = g[f"hit_{w}d"].dropna()
                if len(rets) == 0:
                    continue
                tstat, pval = (np.nan, np.nan)
                if len(rets) >= 3 and rets.std(ddof=1) > 0:
                    tstat, pval = stats.ttest_1samp(rets, 0.0)
                rows.append({
                    "asset": asset,
                    "sentiment_bucket": bucket,
                    "window_days": w,
                    "n_events": int(len(rets)),
                    "avg_return": float(rets.mean()),
                    "median_return": float(rets.median()),
                    "hit_rate": float(hits.mean()) if len(hits) else np.nan,
                    "t_stat": float(tstat),
                    "p_value": float(pval),
                })
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["asset", "window_days", "sentiment_bucket"])
    return summary.reset_index(drop=True)


def high_conviction_hit_rate(study: pd.DataFrame, window: int = 1) -> dict:
    """Headline metric for the résumé line: hit-rate on high-conviction events."""
    hc = study[study["high_conviction"]]
    hits = hc[f"hit_{window}d"].dropna()
    return {
        "window_days": window,
        "n_high_conviction": int(len(hits)),
        "hit_rate": float(hits.mean()) if len(hits) else float("nan"),
    }
