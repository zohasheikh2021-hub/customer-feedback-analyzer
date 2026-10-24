"""Inference layer: loads the saved artifacts once and turns one feedback text into three predictions.

No training happens here. The module is independent of Streamlit so it can be unit-tested and reused.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np

import config as cfg
from text_utils import clean_text

log = logging.getLogger("predictor")

REQUIRED_FILES = [
    cfg.SENTIMENT_MODEL_PATH, cfg.COMPLAINT_MODEL_PATH, cfg.PRIORITY_ANN_PATH,
    cfg.PRIORITY_PREPROCESSOR_PATH, cfg.PRIORITY_BASELINE_PATH, cfg.METADATA_PATH,
]
LOW_COVERAGE_THRESHOLD = 0.5


class ModelNotFoundError(RuntimeError):
    """Raised when trained artifacts are missing from the models/ folder."""


class ModelLoadError(RuntimeError):
    """Raised when artifacts exist but cannot be loaded or are inconsistent."""


class InvalidInputError(ValueError):
    """Raised for feedback text that cannot be analysed (message is safe to show to the user)."""


class InferenceError(RuntimeError):
    """Raised when a loaded model fails or returns something unexpected."""


@dataclass
class TaskPrediction:
    label: str
    confidence: Optional[float]                      # None -> "Confidence unavailable"
    probabilities: Optional[Dict[str, float]]        # None when the model has no probability estimates


@dataclass
class FeedbackAnalysis:
    text: str
    cleaned_text: str
    sentiment: TaskPrediction
    complaint_type: TaskPrediction
    priority: TaskPrediction
    baseline_priority: TaskPrediction
    vocabulary_coverage: float
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _task_from_proba(classes: List[str], proba: np.ndarray) -> TaskPrediction:
    proba = np.asarray(proba, dtype=float).ravel()
    if proba.shape[0] != len(classes) or not np.all(np.isfinite(proba)):
        raise InferenceError("Model returned an unexpected probability vector.")
    if abs(proba.sum() - 1.0) > 1e-3:
        raise InferenceError("Model probabilities do not sum to 1.")
    idx = int(proba.argmax())
    return TaskPrediction(label=str(classes[idx]), confidence=float(proba[idx]),
                          probabilities={str(c): float(p) for c, p in zip(classes, proba)})


class FeedbackPredictor:
    """Loads every saved model once; ``analyze`` is cheap and safe to call per request."""

    def __init__(self, models_dir: Path = cfg.MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._load()

    # ------------------------------------------------------------------ loading
    def _paths(self) -> List[Path]:
        return [self.models_dir / p.name for p in REQUIRED_FILES]

    def _load(self) -> None:
        missing = [p.name for p in self._paths() if not p.exists()]
        if missing:
            log.error("Missing model artifacts: %s", missing)
            raise ModelNotFoundError(
                "Trained model files are missing (" + ", ".join(missing) + "). "
                "Run `python train_models.py` once from the project folder, then restart the app.")
        try:
            d = self.models_dir
            self.metadata = json.loads((d / cfg.METADATA_PATH.name).read_text(encoding="utf-8"))
            self._sentiment = joblib.load(d / cfg.SENTIMENT_MODEL_PATH.name)
            self._complaint = joblib.load(d / cfg.COMPLAINT_MODEL_PATH.name)
            self._priority_pre = joblib.load(d / cfg.PRIORITY_PREPROCESSOR_PATH.name)
            self._priority_base = joblib.load(d / cfg.PRIORITY_BASELINE_PATH.name)
            import tensorflow as tf  # imported here so that a missing-files error needs no TensorFlow
            self._ann = tf.keras.models.load_model(d / cfg.PRIORITY_ANN_PATH.name)
        except Exception as exc:  # noqa: BLE001 - re-raised as a clear, user-safe error
            log.exception("Failed to load model artifacts")
            raise ModelLoadError(
                "The model files could not be loaded. They may be corrupted or built with different library "
                "versions. Re-run `python train_models.py` in this environment.") from exc

        labels = self.metadata["labels"]
        self.sentiment_classes = list(self._sentiment.classes_)
        self.complaint_classes = list(self._complaint.classes_)
        self.priority_classes = list(labels["priority"])
        if (self.sentiment_classes != sorted(labels["sentiment"]) or self.complaint_classes != sorted(labels["complaint_type"])
                or list(self._priority_base.classes_) != sorted(self.priority_classes)):
            raise ModelLoadError("Saved models and metadata disagree about the class labels. Re-run `python train_models.py`.")
        if self._ann.output_shape[-1] != len(self.priority_classes):
            raise ModelLoadError("The ANN output size does not match the priority classes. Re-run `python train_models.py`.")
        for name, pipe in (("sentiment", self._sentiment), ("complaint", self._complaint), ("priority baseline", self._priority_base)):
            if not hasattr(pipe, "predict_proba"):
                raise ModelLoadError(f"The {name} model does not provide probabilities. Re-run `python train_models.py`.")
        vocab = self._sentiment.named_steps["tfidf"].vocabulary_
        self._vocab_unigrams = {t for t in vocab if " " not in t}
        self.max_chars = int(self.metadata.get("max_input_chars", cfg.MAX_INPUT_CHARS))
        log.info("Models loaded from %s", d.name)

    # ------------------------------------------------------------------ validation
    def validate(self, text: object) -> str:
        if not isinstance(text, str):
            raise InvalidInputError("Feedback must be text.")
        text = text.replace("\x00", " ").strip()
        if not text:
            raise InvalidInputError("Please enter some customer feedback before analyzing.")
        if len(text) > self.max_chars:
            raise InvalidInputError(f"Feedback is too long ({len(text):,} characters). Please keep it under {self.max_chars:,} characters.")
        if not clean_text(text):
            raise InvalidInputError("No readable words were found. Please include a sentence made of words, not only punctuation, numbers or emoji.")
        return text

    # ------------------------------------------------------------------ inference
    def analyze(self, text: object) -> FeedbackAnalysis:
        text = self.validate(text)
        cleaned = clean_text(text)
        try:
            sent = _task_from_proba(self.sentiment_classes, self._sentiment.predict_proba([text])[0])
            comp = _task_from_proba(self.complaint_classes, self._complaint.predict_proba([text])[0])
            base = _task_from_proba(list(self._priority_base.classes_), self._priority_base.predict_proba([text])[0])
            feats = np.asarray(self._priority_pre.transform([text]), dtype="float32")
            ann_out = np.asarray(self._ann(feats, training=False))[0]
            prio = _task_from_proba(self.priority_classes, ann_out)
        except InferenceError:
            log.exception("Inference produced invalid output")
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("Inference failed")
            raise InferenceError("Something went wrong while analyzing this feedback. Please try again.") from exc

        tokens = cleaned.split()
        coverage = float(np.mean([t in self._vocab_unigrams for t in tokens])) if tokens else 0.0
        warnings: List[str] = []
        if coverage < LOW_COVERAGE_THRESHOLD:
            warnings.append(
                f"Only {coverage:.0%} of the meaningful words in this text appear in the models' training vocabulary, "
                "so the predictions below are especially uncertain.")
        if len(tokens) < 3:
            warnings.append("Very short feedback gives the models little to work with.")
        return FeedbackAnalysis(text=text, cleaned_text=cleaned, sentiment=sent, complaint_type=comp, priority=prio,
                                baseline_priority=base, vocabulary_coverage=coverage, warnings=warnings)

    def model_summary(self) -> dict:
        return {"models": self.metadata["models"], "labels": self.metadata["labels"],
                "test_metrics": self.metadata["test_metrics"], "trained_at_utc": self.metadata["trained_at_utc"]}
