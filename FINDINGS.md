# Findings — Macro Event Sentiment & Market Reaction

Results from running `python -m scripts.run_pipeline` on the bundled sample
(59 headlines → 108 `(headline, asset)` events, Jan 2023 – Jun 2024), scored
with FinBERT against `CL=F` (crude), `GC=F` (gold), and `ZN=F` (10Y Treasury
futures). Reproduce with the one-command pipeline; full table lives in
`data/processed/reaction_summary.csv`.

---

## TL;DR — three concrete findings

1. **Sentiment is a coin flip next-day.** On high-conviction events
   (`|sentiment| ≥ 0.60`), the **1-day directional hit-rate was 47.6% over 42
   events** — statistically indistinguishable from random. Headline tone alone
   does *not* predict the next day's move, which is the expected result for
   liquid, heavily-watched markets that price news on impact.

2. **The sentiment→price sign *flips* for rate-sensitive assets — and it's
   significant.** "Positive"-toned macro headlines preceded a **−0.46% 3-day move
   in 10-year Treasury futures (t = −2.71, p = 0.017)** and **−0.70% in gold over
   5 days**. FinBERT reads a decisive/hawkish Fed as *positive tone*, but hawkish
   news is *bearish* for bonds and gold — so a naive sentiment-sign rule isn't
   just weak here, it's **backwards**. This is the whole reason to measure
   reaction windows instead of assuming causation.

3. **Crude reacts to supply-tone, but noisily.** Positive-toned oil headlines
   preceded an average **−1.4% next-day crude move** (n = 5, hit-rate 20%) —
   the same sign-flip, since "OPEC cut extended / decisive" tone tends to land
   when prices are *already* elevated and mean-revert. Small sample: directional,
   not conclusive.

> **Interview one-liner:** *"'Positive'-toned macro headlines preceded an average
> −0.5% move in 10-year Treasury futures over the next three days (t = −2.7) —
> hawkish-Fed news reads as positive tone but is bearish for bonds. That's exactly
> why I measured reaction windows instead of trusting the sentiment sign."*

---

## Headline metric

| Metric | Value |
|---|---|
| High-conviction events (`\|score\| ≥ 0.60`) | 42 |
| **1-day directional hit-rate** | **47.6%** |
| Interpretation | ≈ coin flip → tone alone not tradeable next-day |

---

## Reaction summary (from `data/processed/reaction_summary.csv`)

### US 10Y Treasury futures (`ZN=F`)

| Sentiment | Window | n | Avg move | Hit-rate | t-stat | p-value |
|---|---|---|---|---|---|---|
| positive | 1d | 15 | +0.08% | 60% | 0.66 | 0.52 |
| negative | 1d | 23 | +0.25% | 39% | 1.71 | 0.10 |
| **positive** | **3d** | **15** | **−0.46%** | **33%** | **−2.71** | **0.017** ✅ |
| negative | 3d | 23 | −0.10% | 48% | −0.62 | 0.54 |
| positive | 5d | 15 | −0.37% | 33% | −1.58 | 0.14 |

### Gold (`GC=F`)

| Sentiment | Window | n | Avg move | Hit-rate | t-stat | p-value |
|---|---|---|---|---|---|---|
| positive | 1d | 15 | −0.04% | 53% | −0.16 | 0.87 |
| negative | 1d | 23 | +0.29% | 48% | 1.16 | 0.26 |
| positive | 3d | 15 | −0.35% | 33% | −1.12 | 0.28 |
| positive | 5d | 15 | −0.70% | 47% | −1.67 | 0.12 |
| negative | 5d | 23 | +0.33% | 35% | 0.83 | 0.42 |

### WTI Crude (`CL=F`)

| Sentiment | Window | n | Avg move | Hit-rate | t-stat | p-value |
|---|---|---|---|---|---|---|
| positive | 1d | 5 | −1.42% | 20% | −1.98 | 0.12 |
| negative | 1d | 6 | +0.24% | 67% | 0.19 | 0.86 |
| positive | 3d | 5 | −0.83% | 40% | −0.79 | 0.47 |
| negative | 3d | 6 | +1.62% | 33% | 0.98 | 0.37 |
| positive | 5d | 5 | −3.08% | 20% | −1.34 | 0.25 |

✅ = the one cell where the average move is statistically distinguishable from
zero at p < 0.05. Note it's **negative for positive-tone** — the sign flip.

---

## Interpretation

- **Efficient-market read.** A ~48% next-day hit-rate on high-conviction news is
  what you'd expect if the market has already moved by the time the headline
  prints. The tradeable signal, if any, lives in **surprise** (actual vs.
  consensus), not tone.
- **Tone ≠ direction for macro assets.** FinBERT scores linguistic sentiment.
  "Fed holds firm / decisive Powell" is positive *language* but hawkish *policy*,
  which pushes bond and gold prices **down**. The only statistically significant
  cell in the whole study (bonds, positive, 3d, p = 0.017) is negative — a clean,
  defensible demonstration that you must not map sentiment sign to price direction.
- **Commodities are more mechanical but noisier here.** Crude's reaction is larger
  in magnitude but built on tiny samples (n = 5–6); treat as directional only.

---

## What would strengthen the edge (next iterations)

1. **Surprise features** — CPI actual vs. consensus, Fed dot-plot shift — instead
   of raw tone. Surprise is what markets price.
2. **Intraday windows** — most of the reaction happens in the first hours; daily
   close-to-close washes it out.
3. **De-clustering** — collapse multiple same-day headlines about one event so a
   single FOMC decision isn't counted 3×.
4. **Bigger, unbiased sample** — re-run on a large NewsAPI/GDELT/Kaggle pull; the
   bundled set is curated for coverage, not a random draw.

---

## Limitations

- Daily close-to-close windows miss the intraday move where most reaction occurs.
- Sample is curated (not survivorship-free); numbers above are illustrative of the
  *method*, not a robust backtested edge.
- FinBERT scores tone, not market-relevant surprise — and the two diverge exactly
  when it matters most (the sign-flip finding above).
