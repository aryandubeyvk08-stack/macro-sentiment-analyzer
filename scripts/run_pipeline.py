"""End-to-end pipeline: headlines -> FinBERT sentiment -> event study -> tables.

Run from the repo root:

    python -m scripts.run_pipeline                 # uses bundled sample data
    python -m scripts.run_pipeline --headlines data/raw/headlines_live.csv

Outputs land in ``data/processed/`` and feed the Streamlit dashboard.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import config
from src.event_study import (high_conviction_hit_rate, run_event_study, summarize)
from src.market_data import default_date_range, get_prices
from src.sentiment import FinBERTScorer
from src.utils import explode_by_asset, get_logger, load_headlines

log = get_logger("pipeline")


def daily_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Aggregate headline-level scores to a per-day, per-asset series."""
    agg = (scored.groupby(["date", "asset"])
           .agg(mean_sentiment=("sentiment_score", "mean"),
                n_headlines=("headline", "count"),
                n_positive=("label", lambda s: (s == "positive").sum()),
                n_negative=("label", lambda s: (s == "negative").sum()))
           .reset_index())
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headlines", default=None,
                    help="CSV with date,headline[,source]. Defaults to sample data.")
    ap.add_argument("--no-cache", action="store_true",
                    help="Force a fresh yfinance pull instead of the cached prices.")
    args = ap.parse_args()

    # 1. Load + route headlines to assets ----------------------------------
    headlines = load_headlines(args.headlines)
    routed = explode_by_asset(headlines)
    log.info("Routed to %d (headline, asset) events across %d assets",
             len(routed), routed["asset"].nunique())

    # 2. Score sentiment with FinBERT --------------------------------------
    scorer = FinBERTScorer(model_name=config.FINBERT_MODEL)
    scored = scorer.score_dataframe(routed, text_col="headline")
    config.SCORED_HEADLINES_CSV.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(config.SCORED_HEADLINES_CSV, index=False)
    log.info("Wrote %s", config.SCORED_HEADLINES_CSV)

    daily = daily_sentiment(scored)
    daily.to_csv(config.DAILY_SENTIMENT_CSV, index=False)

    # 3. Fetch prices + run the event study --------------------------------
    start, end = default_date_range(headlines)
    prices = get_prices(start, end, use_cache=not args.no_cache)

    study = run_event_study(scored, prices)
    study.to_csv(config.EVENT_STUDY_CSV, index=False)

    summary = summarize(study)
    summary.to_csv(config.REACTION_SUMMARY_CSV, index=False)

    # 4. Print the headline finding ----------------------------------------
    hc = high_conviction_hit_rate(study, window=1)
    log.info("=" * 64)
    log.info("High-conviction (|score| >= %.2f) 1-day directional hit-rate: "
             "%.1f%% over %d events",
             config.HIGH_CONVICTION_THRESHOLD,
             100 * hc["hit_rate"], hc["n_high_conviction"])
    log.info("Reaction summary written to %s", config.REACTION_SUMMARY_CSV)
    log.info("Launch the dashboard with:  streamlit run app/dashboard.py")
    log.info("=" * 64)

    print("\n=== Reaction summary (avg forward move by sentiment bucket) ===")
    with pd.option_context("display.width", 120, "display.max_rows", 60):
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
