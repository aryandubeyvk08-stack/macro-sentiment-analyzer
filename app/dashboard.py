"""Streamlit dashboard: sentiment timeline vs price, event markers, reaction table.

Run from the repo root:
    streamlit run app/dashboard.py

If the processed CSVs don't exist yet, run the pipeline first:
    python -m scripts.run_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Make `src` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

st.set_page_config(page_title="Macro Sentiment & Market Reaction", layout="wide",
                   page_icon="📈")


@st.cache_data
def load_processed():
    """Load the pipeline outputs; return None for any file that's missing."""
    def _read(path, **kw):
        return pd.read_csv(path, **kw) if Path(path).exists() else None

    scored = _read(config.SCORED_HEADLINES_CSV, parse_dates=["date"])
    daily = _read(config.DAILY_SENTIMENT_CSV, parse_dates=["date"])
    prices = _read(config.PRICES_CSV, parse_dates=["date"])
    study = _read(config.EVENT_STUDY_CSV, parse_dates=["date", "entry_date"])
    summary = _read(config.REACTION_SUMMARY_CSV)
    return scored, daily, prices, study, summary


scored, daily, prices, study, summary = load_processed()

st.title("📈 Macro Event Sentiment & Market Reaction Analyzer")
st.caption("FinBERT sentiment on financial headlines vs. how crude oil, gold, and "
           "bond futures actually reacted afterwards.")

if scored is None or prices is None:
    st.warning("No processed data found. Run the pipeline first:\n\n"
               "```\npython -m scripts.run_pipeline\n```")
    st.stop()

# --- Sidebar controls ------------------------------------------------------
assets = sorted(prices["asset"].unique())
asset_names = {k: config.ASSETS.get(k, {}).get("name", k) for k in assets}
asset = st.sidebar.selectbox("Asset", assets, format_func=lambda a: asset_names[a])
window = st.sidebar.selectbox("Forward window (trading days)", config.FORWARD_WINDOWS, index=0)
st.sidebar.markdown("---")
st.sidebar.caption("High-conviction threshold: "
                   f"|sentiment| ≥ {config.HIGH_CONVICTION_THRESHOLD}")

# --- KPI row ---------------------------------------------------------------
a_study = study[study["asset"] == asset].copy()
hc = a_study[a_study["conviction"] >= config.HIGH_CONVICTION_THRESHOLD]
hits = hc[f"hit_{window}d"].dropna()
avg_move_pos = a_study.loc[a_study["sentiment_score"] >= 0.15, f"ret_{window}d"].mean()
avg_move_neg = a_study.loc[a_study["sentiment_score"] <= -0.15, f"ret_{window}d"].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Events (headlines)", len(a_study))
c2.metric(f"High-conviction {window}d hit-rate",
          f"{100 * hits.mean():.0f}%" if len(hits) else "n/a",
          help=f"Over {len(hits)} high-conviction events")
c3.metric(f"Avg {window}d move · positive news",
          f"{100 * avg_move_pos:+.2f}%" if pd.notna(avg_move_pos) else "n/a")
c4.metric(f"Avg {window}d move · negative news",
          f"{100 * avg_move_neg:+.2f}%" if pd.notna(avg_move_neg) else "n/a")

# --- Sentiment vs price chart ---------------------------------------------
st.subheader(f"{asset_names[asset]} — price vs. daily sentiment")

a_prices = prices[prices["asset"] == asset].sort_values("date")
a_daily = (daily[daily["asset"] == asset].sort_values("date")
           if daily is not None else pd.DataFrame())

fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=a_prices["date"], y=a_prices["close"], name="Close",
                         line=dict(color="#1f77b4", width=2)), secondary_y=False)
if not a_daily.empty:
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in a_daily["mean_sentiment"]]
    fig.add_trace(go.Bar(x=a_daily["date"], y=a_daily["mean_sentiment"],
                         name="Daily sentiment", marker_color=colors, opacity=0.5),
                  secondary_y=True)

# Event markers for high-conviction headlines.
hc_events = a_study[a_study["conviction"] >= config.HIGH_CONVICTION_THRESHOLD]
if not hc_events.empty:
    merged = hc_events.merge(a_prices[["date", "close"]], left_on="entry_date",
                             right_on="date", how="left", suffixes=("", "_px"))
    up = merged[merged["sentiment_score"] > 0]
    dn = merged[merged["sentiment_score"] < 0]
    fig.add_trace(go.Scatter(
        x=up["entry_date"], y=up["close"], mode="markers", name="Positive headline",
        marker=dict(symbol="triangle-up", size=11, color="#2ca02c"),
        text=up["headline"], hovertemplate="%{text}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=dn["entry_date"], y=dn["close"], mode="markers", name="Negative headline",
        marker=dict(symbol="triangle-down", size=11, color="#d62728"),
        text=dn["headline"], hovertemplate="%{text}<extra></extra>"), secondary_y=False)

fig.update_layout(height=460, hovermode="x unified",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
fig.update_yaxes(title_text="Price", secondary_y=False)
fig.update_yaxes(title_text="Mean sentiment", secondary_y=True, range=[-1, 1])
st.plotly_chart(fig, use_container_width=True)

# --- Reaction summary table -----------------------------------------------
st.subheader("Reaction study — average forward move by sentiment bucket")
if summary is not None and not summary.empty:
    tbl = summary[(summary["asset"] == asset) & (summary["window_days"] == window)].copy()
    if not tbl.empty:
        show = tbl[["sentiment_bucket", "n_events", "avg_return", "median_return",
                    "hit_rate", "t_stat", "p_value"]].copy()
        show["avg_return"] = (100 * show["avg_return"]).round(2).astype(str) + "%"
        show["median_return"] = (100 * show["median_return"]).round(2).astype(str) + "%"
        show["hit_rate"] = (100 * show["hit_rate"]).round(0).astype("Int64").astype(str) + "%"
        show["t_stat"] = show["t_stat"].round(2)
        show["p_value"] = show["p_value"].round(3)
        show.columns = ["Sentiment", "Events", f"Avg {window}d move",
                        f"Median {window}d move", "Hit-rate", "t-stat", "p-value"]
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("A |t-stat| > ~2 with p < 0.05 suggests the average move is "
                   "statistically distinguishable from zero for that bucket.")

# --- Top headlines table ---------------------------------------------------
st.subheader("Highest-conviction headlines")
if not a_study.empty:
    top = a_study.reindex(a_study["conviction"].sort_values(ascending=False).index)
    cols = ["date", "headline", "label", "sentiment_score", f"ret_{window}d",
            f"hit_{window}d", "source"]
    cols = [c for c in cols if c in top.columns]
    disp = top[cols].head(20).copy()
    disp["sentiment_score"] = disp["sentiment_score"].round(3)
    disp[f"ret_{window}d"] = (100 * disp[f"ret_{window}d"]).round(2).astype(str) + "%"
    st.dataframe(disp, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Built with FinBERT · yfinance · Streamlit. Sentiment ≠ causation — "
           "see FINDINGS.md for the honest interpretation.")
