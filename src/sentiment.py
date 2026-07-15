"""FinBERT sentiment scoring.

Wraps the ``ProsusAI/finbert`` model in a small batched scorer that turns each
headline into class probabilities and a single signed ``sentiment_score`` in
[-1, 1] (positive prob minus negative prob).
"""

from __future__ import annotations

import pandas as pd

from .utils import get_logger

log = get_logger(__name__)

_LABELS = ["positive", "negative", "neutral"]


class FinBERTScorer:
    """Lazy-loaded FinBERT scorer.

    The model (~440 MB) is downloaded on first use and cached by HuggingFace.
    Kept as a class so the (expensive) load happens once and is reused across
    the whole pipeline / a Streamlit session.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None,
                 batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._tokenizer = None

    # -- lazy loading -------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading FinBERT (%s) on %s ...", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        # Map the model's own id2label ordering to our canonical label names.
        self._id2label = {int(k): v.lower() for k, v in self._model.config.id2label.items()}
        log.info("FinBERT ready. Label order: %s", self._id2label)

    # -- scoring ------------------------------------------------------------
    def score_texts(self, texts: list[str]) -> pd.DataFrame:
        """Score a list of texts. Returns one row per text with prob_* + score."""
        import torch

        self._ensure_loaded()
        records: list[dict] = []
        texts = [str(t) for t in texts]

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            enc = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=128
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                scores = {self._id2label[i]: float(p) for i, p in enumerate(row)}
                pos = scores.get("positive", 0.0)
                neg = scores.get("negative", 0.0)
                neu = scores.get("neutral", 0.0)
                label = max(scores, key=scores.get)
                records.append({
                    "prob_positive": pos,
                    "prob_negative": neg,
                    "prob_neutral": neu,
                    "label": label,
                    # Signed conviction: +1 fully positive, -1 fully negative.
                    "sentiment_score": pos - neg,
                })
            log.info("Scored %d / %d headlines", min(start + self.batch_size, len(texts)),
                     len(texts))

        return pd.DataFrame.from_records(records)

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "headline") -> pd.DataFrame:
        """Score a DataFrame's ``text_col`` and concat the score columns back on."""
        scores = self.score_texts(df[text_col].tolist())
        scores.index = df.index
        return pd.concat([df, scores], axis=1)
