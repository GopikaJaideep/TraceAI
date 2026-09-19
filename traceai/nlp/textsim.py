"""Semantic similarity between a missing-person description and a witness report."""
from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from traceai import config

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _model():
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(config.TEXT_EMBED_MODEL)
    except Exception as exc:  # offline first run, missing weights, ...
        log.warning("Sentence-transformer unavailable (%s); using TF-IDF fallback", exc)
        return None


def similarity(a: str, b: str) -> float:
    """Cosine similarity in [0, 1] between two texts."""
    model = _model()
    if model is not None:
        va, vb = model.encode([a, b], normalize_embeddings=True)
        return float(max(0.0, np.dot(va, vb)))
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf = TfidfVectorizer(stop_words="english").fit_transform([a, b])
    return float(max(0.0, (tfidf[0] @ tfidf[1].T).toarray()[0, 0]))
