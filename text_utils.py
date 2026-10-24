"""NLP preprocessing shared by training, notebooks and the web app.

This module is deliberately free of heavy imports (no TensorFlow) because the
saved scikit-learn pipelines reference ``clean_texts`` by import path. Joblib can
only reload them if this module is importable under the same name.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Negations and intensity/recurrence words carry sentiment or urgency, e.g.
# "NOT satisfied" or "the problem is STILL not fixed". Standard stop-word lists
# remove them, so we keep them on purpose.
_KEEP_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "cannot", "neither", "nobody",
    "very", "too", "enough", "more", "less", "only", "again", "still", "already",
    "yet", "always", "every", "much", "many", "few",
}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - _KEEP_WORDS

_CONTRACTIONS = [
    (re.compile(r"\bcan['’]t\b"), "cannot"),
    (re.compile(r"\bwon['’]t\b"), "will not"),
    (re.compile(r"n['’]t\b"), " not"),
    (re.compile(r"['’]re\b"), " are"),
    (re.compile(r"['’]ve\b"), " have"),
    (re.compile(r"['’]ll\b"), " will"),
    (re.compile(r"['’]m\b"), " am"),
    (re.compile(r"['’]s\b"), ""),
]
_NON_ALNUM = re.compile(r"[^a-z0-9\s]")
_SPACES = re.compile(r"\s+")


def clean_text(text: object) -> str:
    """Lower-case, expand contractions, strip punctuation/emoji, drop stop words."""
    if text is None:
        return ""
    return _clean_cached(str(text))


@lru_cache(maxsize=8192)
def _clean_cached(raw: str) -> str:
    # The dataset repeats ~2,100 distinct texts thousands of times; caching makes training fast.
    s = raw.lower()
    for pattern, repl in _CONTRACTIONS:
        s = pattern.sub(repl, s)
    s = _NON_ALNUM.sub(" ", s)
    tokens = [t for t in _SPACES.split(s.strip()) if t and t not in STOP_WORDS]
    return " ".join(tokens)


def clean_texts(texts: Iterable[object]) -> List[str]:
    """Vectorised wrapper used inside scikit-learn pipelines (FunctionTransformer)."""
    return [clean_text(t) for t in texts]


def normalise_for_grouping(text: object) -> str:
    """Key used to detect identical feedback texts when building leak-free splits."""
    return _SPACES.sub(" ", str(text).strip().lower())
