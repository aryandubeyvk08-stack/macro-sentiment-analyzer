"""Headline collection.

The repo ships with a curated sample (``sample_data/sample_headlines.csv``) so
everything runs offline.  For fresh data this module can pull from:

* **NewsAPI** (free developer tier) — needs ``NEWSAPI_KEY`` in the environment.
* **GDELT 2.0 Doc API** — fully free, no key, good for macro/global coverage.

Kaggle datasets (e.g. *Financial PhraseBank*, historical Reuters/Bloomberg
headline sets) are the recommended bulk source; download them into
``data/raw/`` and point the pipeline at the CSV with ``--headlines``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from . import config
from .utils import get_logger

log = get_logger(__name__)

# Query terms that surface macro / commodity / rates headlines.
DEFAULT_QUERIES = [
    "Federal Reserve rate decision",
    "OPEC oil production",
    "US CPI inflation",
    "crude oil price",
    "gold price safe haven",
    "Treasury yields",
]


def fetch_newsapi(queries: list[str] | None = None, days_back: int = 28,
                  api_key: str | None = None) -> pd.DataFrame:
    """Fetch recent headlines from NewsAPI. Free tier ~ last 30 days only."""
    import requests

    api_key = api_key or os.environ.get("NEWSAPI_KEY")
    if not api_key:
        raise RuntimeError(
            "Set NEWSAPI_KEY in your environment to use NewsAPI "
            "(free key at https://newsapi.org)."
        )
    queries = queries or DEFAULT_QUERIES
    from datetime import date, timedelta
    # NOTE: `date.today()` is used only for the request window, not for analysis.
    frm = (date.today() - timedelta(days=days_back)).isoformat()

    rows = []
    for q in queries:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": q, "from": frm, "language": "en", "sortBy": "publishedAt",
                    "pageSize": 100, "apiKey": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        for art in resp.json().get("articles", []):
            rows.append({
                "date": art["publishedAt"][:10],
                "headline": art["title"],
                "source": art.get("source", {}).get("name", "newsapi"),
                "query": q,
            })
        log.info("NewsAPI '%s' -> %d articles", q, len(rows))
    return _dedupe(pd.DataFrame(rows))


def fetch_gdelt(queries: list[str] | None = None, days_back: int = 28) -> pd.DataFrame:
    """Fetch headlines from GDELT's free Doc 2.0 API (no key required)."""
    import requests

    queries = queries or DEFAULT_QUERIES
    rows = []
    for q in queries:
        resp = requests.get(
            "https://api.gdeltproject.org/api/v2/doc/doc",
            params={"query": q, "mode": "artlist", "maxrecords": 100,
                    "timespan": f"{days_back}d", "format": "json", "sort": "datedesc"},
            timeout=30,
        )
        resp.raise_for_status()
        try:
            articles = resp.json().get("articles", [])
        except ValueError:
            log.warning("GDELT returned non-JSON for '%s' (rate limit?)", q)
            continue
        for art in articles:
            seendate = art.get("seendate", "")
            rows.append({
                "date": f"{seendate[:4]}-{seendate[4:6]}-{seendate[6:8]}"
                        if len(seendate) >= 8 else seendate,
                "headline": art.get("title", ""),
                "source": art.get("domain", "gdelt"),
                "query": q,
            })
        log.info("GDELT '%s' -> %d articles", q, len(articles))
    return _dedupe(pd.DataFrame(rows))


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 0]
    df = df.drop_duplicates(subset="headline").reset_index(drop=True)
    return df


def save(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("Saved %d headlines -> %s", len(df), path)
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fetch fresh macro/finance headlines.")
    p.add_argument("--source", choices=["newsapi", "gdelt"], default="gdelt")
    p.add_argument("--days-back", type=int, default=28)
    p.add_argument("--out", default=str(config.RAW_DIR / "headlines_live.csv"))
    args = p.parse_args()

    fetcher = fetch_newsapi if args.source == "newsapi" else fetch_gdelt
    df = fetcher(days_back=args.days_back)
    save(df, args.out)
    print(f"\nFetched {len(df)} headlines. Run the pipeline with:\n"
          f"  python -m scripts.run_pipeline --headlines {args.out}")
