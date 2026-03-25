"""
Text-feature extraction: FinBERT sentiment + Loughran-McDonald lexicon features.

Includes an abstract ``LLMAdapterInterface`` that shows how to swap FinBERT
for any LLM API in a future version.
"""

from __future__ import annotations

import abc
import logging
import re
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loughran-McDonald lexicon word lists (embedded — no external download)
# ---------------------------------------------------------------------------

# Representative subsets of the Loughran-McDonald (2011) financial word lists.
# Full lists contain hundreds of words; we use the most impactful terms.

_UNCERTAINTY_WORDS: set[str] = {
    "approximate", "approximation", "assume", "assumed", "assumption",
    "assumptions", "believe", "believed", "contingency", "contingent",
    "could", "depend", "dependent", "depending", "depends", "doubt",
    "doubtful", "estimate", "estimated", "estimates", "estimation",
    "eventual", "eventually", "exposure", "fluctuate", "fluctuation",
    "fluctuations", "indefinite", "indefinitely", "indicate", "indicating",
    "inherent", "inherently", "may", "might", "nearly", "occasionally",
    "pending", "perhaps", "possible", "possibly", "predict", "predicted",
    "predicting", "prediction", "predictions", "preliminary", "presumably",
    "probable", "probably", "projected", "projecting", "projection",
    "projections", "random", "randomly", "reassess", "reassessment",
    "recalculate", "reconsider", "reestimate", "reexamine", "reinterpret",
    "revise", "revised", "revision", "risk", "risky", "roughly",
    "speculate", "speculative", "suggest", "suggested", "suggesting",
    "susceptible", "tending", "tentative", "tentatively", "uncertain",
    "uncertainty", "unclear", "undecided", "undefined", "undetermined",
    "unforeseeable", "unknown", "unlikely", "unpredictable", "unproven",
    "unquantifiable", "unsettled", "unspecified", "untested", "variable",
    "variability", "vary", "varying", "volatile", "volatility",
}

_HEDGING_WORDS: set[str] = {
    "almost", "apparent", "apparently", "appear", "appeared", "appears",
    "conceivable", "conceivably", "conditional", "conditionally",
    "fairly", "generally", "hopefully", "largely", "likelihood",
    "moderately", "mostly", "partially", "partly", "possibility",
    "possibly", "potentially", "practically", "presumably", "probably",
    "relatively", "seemingly", "somewhat", "substantially", "suggests",
    "typically", "usually",
}

_FORWARD_GUIDANCE_WORDS: set[str] = {
    "anticipate", "anticipated", "anticipates", "anticipating",
    "expect", "expected", "expects", "expecting",
    "forecast", "forecasted", "forecasting", "forecasts",
    "guidance", "guide", "guided", "guides", "guiding",
    "intend", "intended", "intending", "intends",
    "objective", "objectives",
    "outlook",
    "plan", "planned", "planning", "plans",
    "project", "projected", "projecting", "projects",
    "strategy", "strategic",
    "target", "targeted", "targeting", "targets",
    "will",
}

_LITIGIOUS_WORDS: set[str] = {
    "adjudicate", "adjudicated", "adjudication", "allegation", "allegations",
    "allege", "alleged", "alleging", "arbitrate", "arbitration", "claim",
    "claims", "claimant", "claimants", "complain", "complainant", "complaint",
    "complaints", "conviction", "convictions", "counsel", "counterclaim",
    "court", "courts", "decree", "defendant", "defendants", "defense",
    "deposition", "depositions", "dispute", "disputed", "disputes",
    "enjoin", "enjoined", "guilty", "hearing", "hearings", "indict",
    "indicted", "indictment", "infraction", "injunction", "jury",
    "law", "laws", "lawsuit", "lawsuits", "lawyer", "lawyers", "legal",
    "legislate", "legislation", "liable", "liability", "liabilities",
    "litigate", "litigated", "litigation", "magistrate", "motion",
    "motions", "petition", "plaintiff", "plaintiffs", "plead", "pleaded",
    "pleading", "prosecution", "prosecute", "prosecuted", "prosecutor",
    "ruling", "rulings", "settlement", "settlements", "statute", "statutes",
    "statutory", "subpoena", "subpoenas", "sue", "sued", "suit", "suits",
    "testimony", "testify", "trial", "trials", "tribunal", "verdict",
    "verdicts", "violation", "violations",
}


def extract_lexicon_features(text: str) -> dict[str, float]:
    """
    Compute Loughran-McDonald-style lexicon ratios from *text*.

    Returns
    -------
    dict with keys: uncertainty_ratio, hedging_ratio,
    forward_guidance_strength, litigious_ratio.
    """
    words = re.findall(r"[a-z]+", text.lower())
    n = max(len(words), 1)

    def _ratio(wordset: set[str]) -> float:
        return sum(1 for w in words if w in wordset) / n

    return {
        "uncertainty_ratio": _ratio(_UNCERTAINTY_WORDS),
        "hedging_ratio": _ratio(_HEDGING_WORDS),
        "forward_guidance_strength": _ratio(_FORWARD_GUIDANCE_WORDS),
        "litigious_ratio": _ratio(_LITIGIOUS_WORDS),
    }


# ---------------------------------------------------------------------------
# FinBERT sentiment extractor
# ---------------------------------------------------------------------------

class FinBERTExtractor:
    """
    Wraps the ``ProsusAI/finbert`` model for CPU-only inference.

    Long texts are split into chunks of ``config.FINBERT_MAX_TOKENS`` tokens.
    Chunk-level predictions are averaged to produce document-level sentiment.
    """

    _instance: FinBERTExtractor | None = None  # singleton

    def __init__(self) -> None:
        logger.info("Loading FinBERT model (%s) …", config.FINBERT_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.FINBERT_MODEL,
        )
        self.model.eval()
        # Label mapping for ProsusAI/finbert
        self._labels: list[str] = ["positive", "negative", "neutral"]

    @classmethod
    def get_instance(cls) -> FinBERTExtractor:
        """Return a lazily-initialised singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def extract_sentiment(self, text: str) -> dict[str, float]:
        """
        Return document-level sentiment scores.

        Returns
        -------
        dict with keys: sentiment_score (positive - negative),
        positive_prob, negative_prob, neutral_prob.
        """
        chunks = self._chunk_text(text)
        if not chunks:
            return {
                "sentiment_score": 0.0,
                "positive_prob": 0.0,
                "negative_prob": 0.0,
                "neutral_prob": 0.0,
            }

        all_probs: list[np.ndarray] = []
        for i in range(0, len(chunks), config.FINBERT_BATCH_SIZE):
            batch = chunks[i : i + config.FINBERT_BATCH_SIZE]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.FINBERT_MAX_TOKENS,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()
            all_probs.append(probs)

        avg = np.vstack(all_probs).mean(axis=0)
        pos, neg, neu = float(avg[0]), float(avg[1]), float(avg[2])

        return {
            "sentiment_score": pos - neg,
            "positive_prob": pos,
            "negative_prob": neg,
            "neutral_prob": neu,
        }

    # ------------------------------------------------------------------
    def _chunk_text(self, text: str) -> list[str]:
        """Split *text* into token-sized chunks that fit FinBERT's context."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        max_len = config.FINBERT_MAX_TOKENS - 2  # room for [CLS] and [SEP]
        chunks: list[str] = []
        for start in range(0, len(tokens), max_len):
            chunk_ids = tokens[start : start + max_len]
            chunks.append(self.tokenizer.decode(chunk_ids, skip_special_tokens=True))
        return chunks


# ---------------------------------------------------------------------------
# Combined extraction
# ---------------------------------------------------------------------------

def extract_all_text_features(text: str) -> dict[str, float]:
    """
    Extract and merge FinBERT sentiment + lexicon features from *text*.

    Returns a flat dict with all ``config.TEXT_FEATURES`` keys.
    """
    finbert = FinBERTExtractor.get_instance()
    features: dict[str, float] = {}
    features.update(finbert.extract_sentiment(text))
    features.update(extract_lexicon_features(text))
    return features


# ---------------------------------------------------------------------------
# LLM Adapter Interface (upgrade path)
# ---------------------------------------------------------------------------

class LLMAdapterInterface(abc.ABC):
    """
    Abstract interface for replacing FinBERT with any LLM.

    To integrate an LLM (e.g. OpenAI, Anthropic, local Llama):
      1. Subclass this interface.
      2. Implement ``extract_sentiment`` and ``extract_tone``.
      3. Register the adapter in ``extract_all_text_features`` or config.

    Example (pseudo-code)::

        class OpenAIAdapter(LLMAdapterInterface):
            def extract_sentiment(self, text: str) -> dict[str, float]:
                response = openai.chat(messages=[
                    {"role": "system", "content": SENTIMENT_PROMPT},
                    {"role": "user", "content": text[:4000]},
                ])
                return parse_json(response)

            def extract_tone(self, text: str) -> dict[str, float]:
                response = openai.chat(messages=[
                    {"role": "system", "content": TONE_PROMPT},
                    {"role": "user", "content": text[:4000]},
                ])
                return parse_json(response)
    """

    @abc.abstractmethod
    def extract_sentiment(self, text: str) -> dict[str, float]:
        """Return sentiment scores: sentiment_score, positive/negative/neutral probs."""
        ...

    @abc.abstractmethod
    def extract_tone(self, text: str) -> dict[str, float]:
        """Return tone features: uncertainty, hedging, guidance strength, litigious."""
        ...
