# 📈 Macro Event Sentiment & Market Reaction Analyzer
App--> https://macro-sentiment-analyzer-tyvcixtqox6fkuty9uhske.streamlit.app/

> An NLP pipeline that scores financial-news sentiment with **FinBERT** and runs an **event study** on how crude oil, gold, and US Treasury futures *actually* reacted afterwards — visualized in an interactive Streamlit dashboard.

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="FinBERT" src="https://img.shields.io/badge/model-FinBERT-orange">
  <img alt="Streamlit" src="https://img.shields.io/badge/dashboard-Streamlit-red">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## What it does

Financial headlines carry sentiment — but does that sentiment actually *predict* how markets move? This project builds the full pipeline to answer that empirically:

1. **Collect** macro & commodity headlines (bundled sample + live NewsAPI/GDELT fetchers).
2. **Score** each headline's sentiment with FinBERT (a finance-tuned BERT), producing a signed conviction score in `[-1, +1]`.
3. **Route** every headline to the market it plausibly moves (OPEC/oil → crude; Fed/CPI → gold & bonds).
4. **Measure** the realized 1-, 3-, and 5-day forward returns of the relevant futures after each event — a classic **event study**.
5. **Report** directional hit-rates, average moves by sentiment bucket, and statistical significance — all in a Streamlit dashboard.

The honest headline finding (see [`FINDINGS.md`](FINDINGS.md)): **sentiment alone is a weak, mostly-priced-in signal** — which is exactly the nuanced, defensible story you want to tell in an interview.

## Architecture

```
headlines (CSV / NewsAPI / GDELT)
        │
        ▼
  keyword routing ──►  crude / gold / bonds
        │
        ▼
   FinBERT scoring  ──►  prob_pos / prob_neg / sentiment_score
        │
        ▼
   yfinance prices (CL=F, GC=F, ZN=F)
        │
        ▼
   event study  ──►  forward returns · hit-rate · t-stats
        │
        ▼
   Streamlit dashboard  +  FINDINGS.md
```

```
macro-sentiment-analyzer/
├── src/
│   ├── config.py            # paths, asset universe, keyword routing
│   ├── collect_headlines.py # NewsAPI + GDELT fetchers
│   ├── sentiment.py         # FinBERT scorer
│   ├── market_data.py       # yfinance prices (+ ETF fallback, caching)
│   ├── event_study.py       # forward returns, hit-rates, t-stats
│   └── utils.py             # loading + headline→asset routing
├── scripts/run_pipeline.py  # end-to-end: headlines → scores → event study
├── app/dashboard.py         # Streamlit dashboard
├── data/macro_events.csv    # hand-built FOMC / OPEC / CPI calendar
├── sample_data/             # curated headlines so it runs out of the box
└── tests/                   # offline unit tests (routing + event-study maths)
```

---

## Quickstart

```bash
# 1. Clone and enter
git clone https://github.com/<your-username>/macro-sentiment-analyzer.git
cd macro-sentiment-analyzer

# 2. Create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the full pipeline on the bundled sample data
#    (first run downloads FinBERT, ~440 MB)
python -m scripts.run_pipeline

# 5. Launch the dashboard
streamlit run app/dashboard.py
```

That's it — the repo ships with a curated headline set and a macro-event calendar, so **step 4 works with no API keys and no manual downloads.**

---

## Using fresh / live data

**GDELT (free, no key):**
```bash
python -m src.collect_headlines --source gdelt --days-back 28 --out data/raw/headlines_live.csv
python -m scripts.run_pipeline --headlines data/raw/headlines_live.csv
```

**NewsAPI (free developer key from https://newsapi.org):**
```bash
export NEWSAPI_KEY=your_key_here        # Windows: setx NEWSAPI_KEY your_key_here
python -m src.collect_headlines --source newsapi --days-back 28 --out data/raw/headlines_live.csv
python -m scripts.run_pipeline --headlines data/raw/headlines_live.csv
```

**Kaggle bulk datasets** (recommended for volume — e.g. *Financial PhraseBank*, historical Reuters/Bloomberg headline sets): download into `data/raw/` and point the pipeline at any CSV with `date` and `headline` columns.

---

## Methodology notes (the interview-defensible part)

- **Entry realism.** A headline stamped on day *t* is matched to the last close on-or-before *t*; forward return is measured close-to-close over *N* trading days. You can't trade a headline before it prints, so we never use the pre-headline price.
- **Directional hit-rate.** For each event we test the naive hypothesis *positive sentiment → price up, negative → price down*, and count how often the realized move agrees. **High-conviction** events are those with `|sentiment| ≥ 0.60`.
- **Significance.** Each (asset, sentiment-bucket, window) cell reports a one-sample t-stat / p-value on the mean forward return, so you can see whether a bucket's average move is distinguishable from zero.
- **Why the naive relationship is weak.** Markets often price the news *before* the headline prints, and the sentiment→direction mapping inverts by asset (hawkish-Fed news is *negative* for bonds but the headline reads "positive/decisive"). Measuring reaction windows instead of assuming causation is the whole point.

---

## Assets & tickers

| Signal source | Market | Futures (primary) | ETF proxy (fallback) |
|---|---|---|---|
| OPEC, crude, oil, WTI, Brent | WTI Crude Oil | `CL=F` | `USO` |
| Fed, CPI, inflation, safe-haven | Gold | `GC=F` | `GLD` |
| Fed, CPI, yields, Treasuries | US 10Y Treasuries | `ZN=F` | `TLT` |

---

## Tests

```bash
pytest -q
```
The tests cover keyword routing and the event-study maths — they run fully offline (no model download, no network).

---

## Talking points this project gives you

- *"I found sentiment alone isn't tradeable — markets often price in news before headlines appear. That's why I measured reaction windows instead of assuming causation."*
- Demonstrates the end-to-end loop: **data → events → market reaction → dashboard**, applying transformer NLP to a new domain.
- Bridges an ML/automation background into finance: *"I applied my existing ML skills to markets"* rather than *"I have no finance background."*

---

## Tech stack

Python · HuggingFace Transformers (FinBERT) · PyTorch · pandas / NumPy / SciPy · yfinance · Streamlit · Plotly · NewsAPI / GDELT

## License

MIT — see [`LICENSE`](LICENSE). Update the copyright line with your name before publishing.

> **Disclaimer:** research/educational project only. Nothing here is investment advice.
Developer --> Aryan Vijay Dubey 
