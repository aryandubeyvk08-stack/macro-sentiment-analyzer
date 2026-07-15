"""Unit tests for routing + the event-study maths.

These run without network or the FinBERT model, so CI stays fast and offline.
    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_study import run_event_study, summarize
from src.utils import route_headline


def test_route_headline_maps_opec_to_crude():
    assets = route_headline("OPEC+ announces surprise oil output cut")
    assert "crude" in assets


def test_route_headline_fed_hits_gold_and_bonds():
    assets = route_headline("Fed signals hawkish hold on interest rates")
    assert set(assets) == {"gold", "bonds"}


def test_route_headline_no_match_returns_empty():
    assert route_headline("Local bakery wins regional pastry award") == []


def test_forward_returns_and_hit_flags():
    # Two events on one asset with a hand-built, monotonically rising price path.
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04",
                                "2023-01-05", "2023-01-06"]),
        "asset": "crude",
        "close": [100.0, 101.0, 102.0, 103.0, 104.0],
    })
    scored = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02", "2023-01-03"]),
        "asset": "crude",
        "headline": ["bullish oil", "bearish oil"],
        "label": ["positive", "negative"],
        "sentiment_score": [0.8, -0.7],
    })

    study = run_event_study(scored, prices, windows=[1])
    # Event 1: 100 -> 101 = +1%.  Positive sentiment + up move => hit.
    row1 = study[study["headline"] == "bullish oil"].iloc[0]
    assert np.isclose(row1["ret_1d"], 0.01)
    assert bool(row1["hit_1d"]) is True
    # Event 2: 101 -> 102 = +1% up, but sentiment negative => miss.
    row2 = study[study["headline"] == "bearish oil"].iloc[0]
    assert np.isclose(row2["ret_1d"], 101.0 / 101.0 * (102.0 / 101.0) - 1.0, atol=1e-9) \
        or np.isclose(row2["ret_1d"], 102.0 / 101.0 - 1.0)
    assert bool(row2["hit_1d"]) is False


def test_summarize_shapes():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]),
        "asset": "gold",
        "close": [1800.0, 1810.0, 1795.0, 1820.0],
    })
    scored = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-03"]),
        "asset": "gold",
        "headline": ["a", "b", "c"],
        "label": ["positive", "negative", "positive"],
        "sentiment_score": [0.5, -0.4, 0.3],
    })
    study = run_event_study(scored, prices, windows=[1])
    summary = summarize(study, windows=[1])
    assert {"asset", "sentiment_bucket", "window_days", "avg_return",
            "hit_rate", "n_events"} <= set(summary.columns)
    assert (summary["window_days"] == 1).all()
