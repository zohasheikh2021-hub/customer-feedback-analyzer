"""Reproducible training workflow for the Customer Feedback Analyzer.

Run once:  python train_models.py
Outputs : models/*, reports/results.json, reports/eda_summary.json,
          reports/evaluation_report.md, reports/figures/*, reports/confusion_matrices/*

The web app never imports or runs this training code; it only loads the saved artifacts.
Heavy libraries (TensorFlow, matplotlib) are imported lazily so notebooks and tests can
import the lightweight helpers quickly.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

import config as cfg
from text_utils import STOP_WORDS, clean_text, clean_texts, normalise_for_grouping

log = logging.getLogger("train")
# The majority-class reference model never predicts some classes, which makes precision undefined (= 0).
warnings.filterwarnings("ignore", message="Precision is ill-defined")

CV_SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}
ANN_SEEDS = [42, 7, 2024]           # the seed with the lowest validation loss is saved and deployed
STRUCTURED_GAIN_THRESHOLD = 0.01    # adopt structured fields only if CV macro-F1 gains >= 1 point
STRUCTURED_COLS = ["channel", "customer_type", "order_value", "days_as_customer"]


# ----------------------------------------------------------------------------------------
# 1. Data loading, validation and splitting
# ----------------------------------------------------------------------------------------
def set_seeds(seed: int = cfg.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_and_validate(path=cfg.DATA_PATH) -> Tuple[pd.DataFrame, dict]:
    """Load the CSV and return (dataframe, validation report). Raises on unusable data."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Place the CSV in the data/ folder.")
    df = pd.read_csv(path)
    missing_cols = [c for c in cfg.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    unexpected = {
        "sentiment": sorted(set(df["sentiment"].dropna().unique()) - set(cfg.SENTIMENT_ORDER)),
        "priority": sorted(set(df["priority"].dropna().unique()) - set(cfg.PRIORITY_ORDER)),
    }
    if unexpected["sentiment"] or unexpected["priority"]:
        raise ValueError(f"Unexpected target labels found: {unexpected}")

    report = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
        "unexpected_labels": unexpected,
        "empty_texts": int((df["feedback_text"].astype(str).str.strip() == "").sum()),
        "negative_order_values": int((df["order_value"] < 0).sum()),
        "non_positive_days": int((df["days_as_customer"] < 0).sum()),
    }
    if sum(report["missing_values"].values()) > 0:
        raise ValueError(f"Missing values found: {report['missing_values']}")
    df["feedback_text"] = df["feedback_text"].astype(str)
    return df, report


@dataclass
class Splits:
    train: np.ndarray     # all training rows (used by the scikit-learn models)
    test: np.ndarray      # held-out test rows (never used for fitting or model selection)
    fit: np.ndarray       # ANN fit subset of train
    val: np.ndarray       # ANN validation subset of train (early stopping)
    groups: np.ndarray    # identical feedback texts share a group id


def make_splits(df: pd.DataFrame, seed: int = cfg.RANDOM_SEED) -> Splits:
    """Leak-free split.

    The dataset has only ~2,100 distinct feedback texts repeated across 30,000 rows. A plain random
    split would put the same text in train and test (>99 % of test texts would also be in train),
    so the test score would measure memorisation. We therefore group rows by identical text and
    use StratifiedGroupKFold: no text appears on both sides of any split.
    """
    groups = df["feedback_text"].map(normalise_for_grouping).factorize()[0]
    # sentiment and complaint type are constant within a text, so stratify on their combination
    strat = (df["sentiment"] + "|" + df["complaint_type"]).values
    outer = StratifiedGroupKFold(n_splits=cfg.TEST_FRACTION_FOLDS, shuffle=True, random_state=seed)
    train, test = next(outer.split(df, strat, groups))
    inner = StratifiedGroupKFold(n_splits=cfg.VAL_FRACTION_FOLDS, shuffle=True, random_state=seed)
    fit_rel, val_rel = next(inner.split(train, strat[train], groups[train]))
    sp = Splits(train=train, test=test, fit=train[fit_rel], val=train[val_rel], groups=groups)
    assert not set(groups[sp.train]) & set(groups[sp.test]), "text leakage between train and test"
    assert not set(groups[sp.fit]) & set(groups[sp.val]), "text leakage between ANN fit and validation"
    return sp


# ----------------------------------------------------------------------------------------
# 2. Exploratory data analysis
# ----------------------------------------------------------------------------------------
def _top_words(texts: pd.Series, n: int = 15) -> List[List]:
    cv = CountVectorizer(preprocessor=lambda s: s, tokenizer=str.split, token_pattern=None, lowercase=False)
    mat = cv.fit_transform([clean_text(t) for t in texts.unique()])
    # weight by how many rows use each distinct text, so counts reflect the full dataset
    weights = texts.value_counts().reindex(texts.unique()).values
    counts = np.asarray(mat.T @ weights).ravel()
    order = np.argsort(-counts)[:n]
    names = cv.get_feature_names_out()
    return [[str(names[i]), int(counts[i])] for i in order]


def eda_summary(df: pd.DataFrame, validation: dict, sp: Splits) -> dict:
    n = len(df)
    dist = {c: df[c].value_counts().to_dict() for c in ["sentiment", "complaint_type", "priority", "channel", "customer_type"]}
    dist = {k: {str(a): int(b) for a, b in v.items()} for k, v in dist.items()}

    def imbalance(counts: Dict[str, int]) -> float:
        return round(max(counts.values()) / min(counts.values()), 2)

    text_nunique = df.groupby("feedback_text")[cfg.TARGETS].nunique()
    conflicts = {t: int((text_nunique[t] > 1).sum()) for t in cfg.TARGETS}
    rows_in_conflicting_texts = {
        t: int(df["feedback_text"].isin(text_nunique.index[text_nunique[t] > 1]).sum()) for t in cfg.TARGETS
    }
    lengths = df["feedback_text"].str.len()
    words = df["feedback_text"].str.split().str.len()
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    perm = rng.permutation(n)
    rnd_train, rnd_test = perm[: int(n * 0.8)], perm[int(n * 0.8):]
    seen = float(pd.Series(sp.groups[rnd_test]).isin(set(sp.groups[rnd_train])).mean())

    def xtab(a: str, b: str) -> dict:
        return {str(i): {str(j): int(v) for j, v in row.items()} for i, row in pd.crosstab(df[a], df[b]).iterrows()}

    top_by_sentiment = {s: _top_words(df.loc[df.sentiment == s, "feedback_text"], 10) for s in cfg.SENTIMENT_ORDER}
    summary = {
        "validation": validation,
        "n_rows": n,
        "distributions": dist,
        "imbalance_ratio_max_over_min": {k: imbalance(dist[k]) for k in ["sentiment", "complaint_type", "priority"]},
        "duplicates": {
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_feedback_ids": int(df["feedback_id"].duplicated().sum()),
            "rows_with_repeated_text": int(df["feedback_text"].duplicated(keep=False).sum()),
            "distinct_texts": int(df["feedback_text"].nunique()),
            "duplicate_text_and_labels": int(df.duplicated(["feedback_text"] + cfg.TARGETS).sum()),
        },
        "label_conflicts_per_text": conflicts,
        "rows_in_texts_with_conflicting_labels": rows_in_conflicting_texts,
        "text_length_chars": {k: float(v) for k, v in lengths.describe().round(2).items()},
        "text_length_words": {k: float(v) for k, v in words.describe().round(2).items()},
        "numeric_summary": {c: {k: float(v) for k, v in df[c].describe().round(2).items()} for c in ["order_value", "days_as_customer"]},
        "top_words": _top_words(df["feedback_text"], 20),
        "top_words_by_sentiment": top_by_sentiment,
        "crosstab_sentiment_priority": xtab("sentiment", "priority"),
        "crosstab_complaint_priority": xtab("complaint_type", "priority"),
        "crosstab_complaint_sentiment": xtab("complaint_type", "sentiment"),
        "random_split_text_overlap": round(seen, 4),
        "split_sizes": {k: int(len(v)) for k, v in
                        {"train": sp.train, "test": sp.test, "ann_fit": sp.fit, "ann_validation": sp.val}.items()},
        "test_distributions": {t: {str(a): int(b) for a, b in df.iloc[sp.test][t].value_counts().items()} for t in cfg.TARGETS},
    }
    summary["observations"] = _eda_observations(summary)
    return summary


def _eda_observations(s: dict) -> List[str]:
    d, dup = s["distributions"], s["duplicates"]
    sent, cat, pri = d["sentiment"], d["complaint_type"], d["priority"]
    n = s["n_rows"]
    big = max(cat, key=cat.get)
    small = min(cat, key=cat.get)
    obs = [
        f"The dataset has {n:,} rows, no missing values, no duplicate rows and no duplicate feedback_id values.",
        f"Sentiment is moderately imbalanced: {max(sent, key=sent.get)} is the largest class "
        f"({sent[max(sent, key=sent.get)] / n:.1%}) and {min(sent, key=sent.get)} the smallest ({sent[min(sent, key=sent.get)] / n:.1%}).",
        f"There are {len(cat)} complaint categories; {big} is the largest ({cat[big] / n:.1%}) and {small} the smallest ({cat[small] / n:.1%}), "
        f"a {s['imbalance_ratio_max_over_min']['complaint_type']}:1 ratio, so minority-class recall is evaluated explicitly.",
        f"Priority is almost balanced (Low {pri['Low'] / n:.1%}, Medium {pri['Medium'] / n:.1%}, High {pri['High'] / n:.1%}).",
        f"Only {dup['distinct_texts']:,} distinct feedback texts exist; {dup['rows_with_repeated_text']:,} rows share their text with at least one other row. "
        f"A random split would leak: {s['random_split_text_overlap']:.1%} of randomly chosen test texts also appear in the training set. "
        "This project therefore splits by distinct text (no text appears on both sides).",
        f"Sentiment and complaint_type never conflict for the same text ({s['label_conflicts_per_text']['sentiment']} and "
        f"{s['label_conflicts_per_text']['complaint_type']} conflicting texts), but {s['label_conflicts_per_text']['priority']:,} distinct texts "
        f"({s['rows_in_texts_with_conflicting_labels']['priority']:,} rows) occur with more than one priority. "
        "Priority therefore cannot be predicted perfectly from the text alone; a ceiling well below 100 % is expected.",
        f"Feedback texts are short: {s['text_length_words']['mean']:.1f} words on average "
        f"(min {s['text_length_words']['min']:.0f}, max {s['text_length_words']['max']:.0f}).",
    ]
    return obs


# ----------------------------------------------------------------------------------------
# 3. Model builders
# ----------------------------------------------------------------------------------------
def text_pipeline(clf, ngram: Tuple[int, int] = (1, 2)) -> Pipeline:
    return Pipeline([
        ("clean", FunctionTransformer(clean_texts, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=ngram, min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])


def candidate_models(tuned_lr: Pipeline) -> Dict[str, dict]:
    """name -> {estimator, proba, deployable}. The dummy model is a sanity-check reference only."""
    return {
        "Logistic Regression (tuned)": {"estimator": clone(tuned_lr), "proba": True, "deployable": True},
        "Multinomial Naive Bayes": {"estimator": text_pipeline(MultinomialNB(alpha=0.5)), "proba": True, "deployable": True},
        "Linear SVM": {"estimator": text_pipeline(LinearSVC(C=1.0, random_state=cfg.RANDOM_SEED)), "proba": False, "deployable": False},
        "Majority-class baseline": {"estimator": text_pipeline(DummyClassifier(strategy="prior")), "proba": True, "deployable": False},
    }


def evaluate(y_true, y_pred, labels: List[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
        "weighted_precision": float(pw), "weighted_recall": float(rw), "weighted_f1": float(fw),
        "per_class": {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, l in enumerate(labels)},
        "labels": list(labels),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _cv_summary(cv_out: dict) -> dict:
    return {m: {"mean": float(np.mean(cv_out[f"test_{m}"])), "std": float(np.std(cv_out[f"test_{m}"]))} for m in CV_SCORING}


def run_text_task(target: str, df: pd.DataFrame, sp: Splits) -> dict:
    """Tune, cross-validate and test the scikit-learn models for one target."""
    X, y = df["feedback_text"], df[target]
    Xtr, ytr, gtr = X.iloc[sp.train], y.iloc[sp.train], sp.groups[sp.train]
    Xte, yte = X.iloc[sp.test], y.iloc[sp.test]
    labels = sorted(y.unique())
    cv = StratifiedGroupKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)

    # values are ordered simplest-first so that ties in CV score resolve to the simpler model
    grid = {"clf__C": [0.1, 1, 10], "tfidf__ngram_range": [(1, 1), (1, 2)]}
    gs = GridSearchCV(text_pipeline(LogisticRegression(max_iter=2000, random_state=cfg.RANDOM_SEED)), grid,
                      scoring="f1_macro", cv=cv, n_jobs=1, refit=True)
    gs.fit(Xtr, ytr, groups=gtr)
    best_params = {"C": gs.best_params_["clf__C"], "ngram_range": list(gs.best_params_["tfidf__ngram_range"])}
    grid_table = [
        {"C": p["clf__C"], "ngram_range": list(p["tfidf__ngram_range"]), "cv_macro_f1": float(m), "cv_std": float(s)}
        for p, m, s in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"], gs.cv_results_["std_test_score"])
    ]

    results, fitted = {}, {}
    for name, spec in candidate_models(gs.best_estimator_).items():
        est = spec["estimator"]
        cv_out = cross_validate(est, Xtr, ytr, groups=gtr, cv=cv, scoring=CV_SCORING, n_jobs=1)
        est.fit(Xtr, ytr)
        pred = est.predict(Xte)
        results[name] = {
            "supports_probabilities": spec["proba"], "deployable": spec["deployable"],
            "cv": _cv_summary(cv_out), "test": evaluate(yte, pred, labels),
        }
        fitted[name] = est
        log.info("  %-28s cv F1=%.4f  test F1=%.4f  acc=%.4f", name, results[name]["cv"]["f1_macro"]["mean"],
                 results[name]["test"]["macro_f1"], results[name]["test"]["accuracy"])

    eligible = [n for n, r in results.items() if r["deployable"]]
    selected = max(eligible, key=lambda n: results[n]["cv"]["f1_macro"]["mean"])   # ties -> first (LR)
    return {
        "target": target, "labels": labels, "tuning": {"method": "GridSearchCV (grouped, stratified 5-fold, macro-F1)",
                                                      "grid": {"C": grid["clf__C"], "ngram_range": [list(g) for g in grid["tfidf__ngram_range"]]},
                                                      "best_params": best_params, "results": grid_table},
        "candidates": results, "selected": selected, "_fitted": fitted,
    }


def structured_feature_experiment(df: pd.DataFrame, sp: Splits, task_results: Dict[str, dict]) -> dict:
    """Does adding channel / customer_type / order_value / days_as_customer help? (CV on training data only)"""
    cv = StratifiedGroupKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_SEED)
    out = {}
    for target, res in task_results.items():
        bp = res["tuning"]["best_params"]
        feats = ColumnTransformer([
            ("text", Pipeline([("clean", FunctionTransformer(clean_texts, validate=False)),
                               ("tfidf", TfidfVectorizer(ngram_range=tuple(bp["ngram_range"]), min_df=2, sublinear_tf=True))]), "feedback_text"),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["channel", "customer_type"]),
            ("num", Pipeline([("log", FunctionTransformer(np.log1p)), ("scale", StandardScaler())]), ["order_value", "days_as_customer"]),
        ])
        with_struct = Pipeline([("features", feats), ("clf", LogisticRegression(C=bp["C"], max_iter=2000, random_state=cfg.RANDOM_SEED))])
        Xtr = df.iloc[sp.train][["feedback_text"] + STRUCTURED_COLS]
        ytr, gtr = df[target].iloc[sp.train], sp.groups[sp.train]
        s = cross_validate(with_struct, Xtr, ytr, groups=gtr, cv=cv, scoring=CV_SCORING, n_jobs=1)
        text_only = res["candidates"]["Logistic Regression (tuned)"]["cv"]
        with_cv = _cv_summary(s)
        gain = with_cv["f1_macro"]["mean"] - text_only["f1_macro"]["mean"]
        out[target] = {
            "text_only_cv_macro_f1": text_only["f1_macro"]["mean"], "with_structured_cv_macro_f1": with_cv["f1_macro"]["mean"],
            "text_only_cv_accuracy": text_only["accuracy"]["mean"], "with_structured_cv_accuracy": with_cv["accuracy"]["mean"],
            "macro_f1_gain": gain, "adopt_structured_features": bool(gain >= STRUCTURED_GAIN_THRESHOLD),
        }
        log.info("  structured features for %-14s gain in CV macro-F1: %+.4f -> adopt=%s", target, gain, out[target]["adopt_structured_features"])
    return {"threshold": STRUCTURED_GAIN_THRESHOLD, "features": STRUCTURED_COLS, "per_target": out,
            "decision": "text-only" if not any(v["adopt_structured_features"] for v in out.values()) else "REVIEW: structured features help"}


# ----------------------------------------------------------------------------------------
# 4. Keras ANN for priority
# ----------------------------------------------------------------------------------------
def build_priority_preprocessor() -> Pipeline:
    """clean -> TF-IDF (bounded vocabulary) -> TruncatedSVD. Output: dense 256-d float vector.

    SVD keeps the network input small (24k x 256 floats ~ 22 MB instead of a large sparse matrix
    densified) so the model trains comfortably on a laptop.
    """
    return Pipeline([
        ("clean", FunctionTransformer(clean_texts, validate=False)),
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=2000, sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=256, random_state=cfg.RANDOM_SEED)),
    ])


def encode_priority(series: pd.Series) -> np.ndarray:
    return series.map({l: i for i, l in enumerate(cfg.PRIORITY_ORDER)}).values.astype("int32")


def build_ann(input_dim: int, n_classes: int):
    import tensorflow as tf
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name="text_features"),
        tf.keras.layers.Dense(128, activation="relu", name="hidden_1"),
        tf.keras.layers.Dropout(0.3, name="dropout_1"),
        tf.keras.layers.Dense(64, activation="relu", name="hidden_2"),
        tf.keras.layers.Dropout(0.2, name="dropout_2"),
        tf.keras.layers.Dense(n_classes, activation="softmax", name="priority_probabilities"),
    ], name="priority_ann")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_ann(Xfit, yfit, Xval, yval, seed: int):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - older TF builds
        pass
    model = build_ann(Xfit.shape[1], len(cfg.PRIORITY_ORDER))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    hist = model.fit(Xfit, yfit, validation_data=(Xval, yval), epochs=100, batch_size=128, verbose=0, callbacks=callbacks)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    h["best_epoch"] = int(np.argmin(h["val_loss"]) + 1)
    h["epochs_run"] = len(h["loss"])
    return model, h


def grouped_bootstrap(y_true: np.ndarray, preds: Dict[str, np.ndarray], groups: np.ndarray, n_boot: int = 2000,
                      seed: int = cfg.RANDOM_SEED) -> dict:
    """95 % bootstrap CI for accuracy (and for paired differences), resampling whole texts, not rows."""
    uniq, inv = np.unique(groups, return_inverse=True)
    G = len(uniq)
    size = np.bincount(inv, minlength=G).astype(float)
    corr = {k: np.bincount(inv, weights=(p == y_true).astype(float), minlength=G) for k, p in preds.items()}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, size=(n_boot, G))
    tot = size[idx].sum(axis=1)
    acc = {k: corr[k][idx].sum(axis=1) / tot for k in preds}
    out = {k: {"accuracy": float(corr[k].sum() / size.sum()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]} for k, v in acc.items()}
    names = list(preds)
    if len(names) == 2:
        diff = acc[names[0]] - acc[names[1]]
        out["difference"] = {"a": names[0], "b": names[1], "mean": float(diff.mean()),
                             "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}
    out["n_distinct_test_texts"] = int(G)
    return out


# ----------------------------------------------------------------------------------------
# 5. Figures (offline report only)
# ----------------------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 110, "savefig.bbox": "tight"})
    return plt, sns


def make_eda_figures(df: pd.DataFrame, eda: dict) -> None:
    plt, sns = _mpl()
    fig_dir = cfg.FIGURES_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)

    def bar(counts: dict, title: str, fname: str, colors, order=None, horizontal=False):
        order = order or sorted(counts, key=counts.get, reverse=True)
        vals = [counts[k] for k in order]
        fig, ax = plt.subplots(figsize=(7.5, 4.2) if not horizontal else (7.5, 4.6))
        pal = [colors[k] for k in order] if isinstance(colors, dict) else colors[: len(order)]
        if horizontal:
            ax.barh(order[::-1], vals[::-1], color=pal[::-1])
            ax.set_xlabel("Records")
        else:
            ax.bar(order, vals, color=pal)
            ax.set_ylabel("Records")
        for i, v in enumerate(vals[::-1] if horizontal else vals):
            if horizontal:
                ax.text(v, i, f" {v:,} ({v / len(df):.1%})", va="center", fontsize=9)
            else:
                ax.text(i, v, f"{v:,}\n({v / len(df):.1%})", ha="center", va="bottom", fontsize=9)
        ax.set_title(title)
        ax.margins(y=0.15, x=0.15 if horizontal else 0.05)
        fig.savefig(fig_dir / fname)
        plt.close(fig)

    d = eda["distributions"]
    bar(d["sentiment"], "Sentiment distribution (30,000 records)", "eda_sentiment_distribution.png", cfg.SENTIMENT_COLORS, cfg.SENTIMENT_ORDER)
    bar(d["complaint_type"], "Complaint type distribution", "eda_complaint_distribution.png", cfg.CATEGORY_PALETTE, horizontal=True)
    bar(d["priority"], "Priority distribution", "eda_priority_distribution.png", cfg.PRIORITY_COLORS, cfg.PRIORITY_ORDER)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    sns.histplot(df["feedback_text"].str.len(), bins=30, ax=axes[0], color=cfg.COLORS["accent"])
    axes[0].set(title="Feedback length (characters)", xlabel="Characters", ylabel="Records")
    sns.histplot(df["feedback_text"].str.split().str.len(), bins=18, ax=axes[1], color=cfg.COLORS["accent_2"])
    axes[1].set(title="Feedback length (words)", xlabel="Words", ylabel="Records")
    fig.savefig(fig_dir / "eda_text_length.png")
    plt.close(fig)

    words = eda["top_words"]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh([w for w, _ in words][::-1], [c for _, c in words][::-1], color=cfg.COLORS["accent"])
    ax.set(title="Most frequent words after preprocessing", xlabel="Occurrences across all records")
    fig.savefig(fig_dir / "eda_top_words.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ct = pd.crosstab(df["sentiment"], df["priority"]).reindex(index=cfg.SENTIMENT_ORDER, columns=cfg.PRIORITY_ORDER)
    sns.heatmap(ct.div(ct.sum(axis=1), axis=0), annot=ct.values, fmt=",d", cmap="Purples", cbar_kws={"label": "Row share"}, ax=axes[0])
    axes[0].set_title("Priority by sentiment (counts; colour = row share)")
    ct2 = pd.crosstab(df["complaint_type"], df["priority"]).reindex(columns=cfg.PRIORITY_ORDER)
    sns.heatmap(ct2.div(ct2.sum(axis=1), axis=0), annot=ct2.values, fmt=",d", cmap="Purples", cbar_kws={"label": "Row share"}, ax=axes[1])
    axes[1].set_title("Priority by complaint type")
    fig.savefig(fig_dir / "eda_priority_relationships.png")
    plt.close(fig)


def plot_confusion(cm: List[List[int]], labels: List[str], title: str, path) -> None:
    plt, sns = _mpl()
    cm = np.array(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    size = max(4.8, 0.62 * len(labels) + 2.5)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(norm, annot=np.array([[f"{v:,}\n{p:.0%}" for v, p in zip(r, pr)] for r, pr in zip(cm, norm)]), fmt="",
                cmap="Purples", vmin=0, vmax=1, xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Share of true class"}, ax=ax)
    ax.set(title=title, xlabel="Predicted", ylabel="True")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.savefig(path)
    plt.close(fig)


def make_result_figures(results: dict) -> None:
    plt, sns = _mpl()
    rows = []
    for task, key in [("Sentiment", "sentiment"), ("Complaint type", "complaint_type"), ("Priority", "priority")]:
        for name, r in results["tasks"][key]["candidates"].items():
            rows.append({"Task": task, "Model": name, "Test macro-F1": r["test"]["macro_f1"], "Test accuracy": r["test"]["accuracy"]})
    ann = results["tasks"]["priority"]["ann"]
    rows.append({"Task": "Priority", "Model": "Keras ANN", "Test macro-F1": ann["test"]["macro_f1"], "Test accuracy": ann["test"]["accuracy"]})
    comp = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=comp, x="Task", y="Test macro-F1", hue="Model", ax=ax, palette=["#6366f1", "#8b5cf6", "#06b6d4", "#94a3b8", "#f59e0b"])
    ax.set_title("Model comparison on the held-out test set (macro-F1)")
    ax.set_ylim(0, 1.08)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    fig.savefig(cfg.FIGURES_DIR / "model_comparison.png")
    plt.close(fig)

    h = ann["history_deployed"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = range(1, len(h["loss"]) + 1)
    axes[0].plot(ep, h["loss"], label="Training loss", color=cfg.COLORS["accent"])
    axes[0].plot(ep, h["val_loss"], label="Validation loss", color=cfg.COLORS["high"])
    axes[0].axvline(h["best_epoch"], ls="--", color="#64748b", label=f"Best epoch ({h['best_epoch']})")
    axes[0].set(title="ANN loss", xlabel="Epoch", ylabel="Sparse categorical cross-entropy")
    axes[1].plot(ep, h["accuracy"], label="Training accuracy", color=cfg.COLORS["accent"])
    axes[1].plot(ep, h["val_accuracy"], label="Validation accuracy", color=cfg.COLORS["high"])
    axes[1].axvline(h["best_epoch"], ls="--", color="#64748b")
    axes[1].set(title="ANN accuracy", xlabel="Epoch", ylabel="Accuracy")
    for a in axes:
        a.legend(fontsize=8)
    fig.savefig(cfg.FIGURES_DIR / "ann_training_curves.png")
    plt.close(fig)

    cfg.CM_DIR.mkdir(parents=True, exist_ok=True)
    for key, title in [("sentiment", "Sentiment"), ("complaint_type", "Complaint type")]:
        r = results["tasks"][key]
        sel = r["candidates"][r["selected"]]["test"]
        plot_confusion(sel["confusion_matrix"], sel["labels"], f"{title}: {r['selected']}", cfg.CM_DIR / f"cm_{key}.png")
    p = results["tasks"]["priority"]
    base = p["candidates"][p["selected"]]["test"]
    plot_confusion(base["confusion_matrix"], base["labels"], f"Priority baseline: {p['selected']}", cfg.CM_DIR / "cm_priority_baseline.png")
    plot_confusion(ann["test"]["confusion_matrix"], ann["test"]["labels"], "Priority: Keras ANN", cfg.CM_DIR / "cm_priority_ann.png")


# ----------------------------------------------------------------------------------------
# 6. Report writer
# ----------------------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{x:.4f}"


def write_report(results: dict, eda: dict) -> str:
    L: List[str] = []
    L += ["# Evaluation Report", "",
          f"Generated by `train_models.py` on {results['meta']['trained_at_utc']}. Every number below is read from "
          "`reports/results.json`, which is written by the training run; nothing is typed in by hand.", ""]
    sp = eda["split_sizes"]
    L += ["## 1. Setup", "",
          f"- Dataset: {eda['n_rows']:,} rows, {eda['duplicates']['distinct_texts']:,} distinct feedback texts.",
          f"- Split (grouped by identical text, stratified, seed {cfg.RANDOM_SEED}): train {sp['train']:,} rows / test {sp['test']:,} rows. "
          f"ANN fit {sp['ann_fit']:,} / ANN validation {sp['ann_validation']:,} (both inside the training part).",
          f"- Cross-validation: {cfg.CV_FOLDS}-fold StratifiedGroupKFold on the training part only; the test set is used once, for final reporting.",
          "- Model selection uses cross-validated macro-F1, never test scores.", ""]
    L += ["### Data observations", ""] + [f"- {o}" for o in eda["observations"]] + [""]

    for title, key in [("Sentiment", "sentiment"), ("Complaint type", "complaint_type"), ("Priority (traditional baselines)", "priority")]:
        r = results["tasks"][key]
        L += [f"## {title}", "",
              f"Tuning: {r['tuning']['method']}; best parameters {r['tuning']['best_params']}. Deployed traditional model: **{r['selected']}**.", "",
              "| Model | CV macro-F1 (mean ± std) | Test accuracy | Test macro-precision | Test macro-recall | Test macro-F1 | Probabilities |",
              "|---|---|---|---|---|---|---|"]
        for n, c in r["candidates"].items():
            t = c["test"]
            L.append(f"| {n} | {c['cv']['f1_macro']['mean']:.4f} ± {c['cv']['f1_macro']['std']:.4f} | {_fmt(t['accuracy'])} | {_fmt(t['macro_precision'])} | "
                     f"{_fmt(t['macro_recall'])} | {_fmt(t['macro_f1'])} | {'yes' if c['supports_probabilities'] else 'no'} |")
        sel = r["candidates"][r["selected"]]["test"]
        L += ["", f"Per-class test results for **{r['selected']}**:", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for lab, m in sel["per_class"].items():
            L.append(f"| {lab} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {m['support']:,} |")
        L += ["", f"Confusion matrix: `reports/confusion_matrices/cm_{key if key != 'priority' else 'priority_baseline'}.png`", ""]

    p = results["tasks"]["priority"]
    ann = p["ann"]
    L += ["## Priority: Keras ANN vs traditional baseline", "",
          f"ANN input: {ann['input']}. Architecture: {ann['architecture']}. Loss `{ann['loss']}`, optimizer `{ann['optimizer']}`, "
          f"early stopping on validation loss (patience 8, best weights restored).", "",
          "| Model | Test accuracy | Macro-precision | Macro-recall | Macro-F1 |", "|---|---|---|---|---|"]
    b = p["candidates"][p["selected"]]["test"]
    L.append(f"| {p['selected']} (baseline) | {_fmt(b['accuracy'])} | {_fmt(b['macro_precision'])} | {_fmt(b['macro_recall'])} | {_fmt(b['macro_f1'])} |")
    a = ann["test"]
    L.append(f"| Keras ANN (deployed, seed {ann['deployed_seed']}) | {_fmt(a['accuracy'])} | {_fmt(a['macro_precision'])} | {_fmt(a['macro_recall'])} | {_fmt(a['macro_f1'])} |")
    L += ["", f"Robustness over {len(ANN_SEEDS)} random seeds {ANN_SEEDS}: ANN test accuracy {ann['seed_summary']['accuracy_mean']:.4f} ± "
          f"{ann['seed_summary']['accuracy_std']:.4f}, macro-F1 {ann['seed_summary']['macro_f1_mean']:.4f} ± {ann['seed_summary']['macro_f1_std']:.4f}.", ""]
    bs = p["comparison"]["bootstrap"]
    d = bs["difference"]
    lo, hi = d["ci95"]
    verdict = ("The interval includes zero, so the data do not show a reliable accuracy difference between the two models."
               if lo <= 0 <= hi else
               (f"The interval excludes zero: {d['a']} is more accurate." if d["mean"] > 0 else f"The interval excludes zero: {d['b']} is more accurate."))
    L += [f"Paired bootstrap over {bs['n_distinct_test_texts']} distinct test texts (95 % interval): accuracy difference "
          f"({d['a']} − {d['b']}) = {d['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]. {verdict}", "",
          f"Majority-class reference accuracy: {_fmt(p['candidates']['Majority-class baseline']['test']['accuracy'])}.", "",
          "Per-class test results for the ANN:", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for lab, m in a["per_class"].items():
        L.append(f"| {lab} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | {m['support']:,} |")
    L += ["", "Figures: `reports/figures/ann_training_curves.png`, `reports/confusion_matrices/cm_priority_ann.png`, `reports/figures/model_comparison.png`.", ""]

    fe = results["structured_feature_experiment"]
    L += ["## Do structured fields help?", "",
          f"Logistic Regression with TF-IDF alone vs TF-IDF plus {', '.join(fe['features'])} (cross-validated on training data, gain threshold {fe['threshold']:.2f} macro-F1).", "",
          "| Target | Text-only CV macro-F1 | + structured CV macro-F1 | Gain | Adopted |", "|---|---|---|---|---|"]
    for t, v in fe["per_target"].items():
        L.append(f"| {t} | {_fmt(v['text_only_cv_macro_f1'])} | {_fmt(v['with_structured_cv_macro_f1'])} | {v['macro_f1_gain']:+.4f} | {'yes' if v['adopt_structured_features'] else 'no'} |")
    L += ["", f"Decision: **{fe['decision']}**. The app asks only for feedback text, so its input schema matches the trained pipelines exactly.", ""]

    L += ["## Caveats", "",
          "- The data are highly templated (few distinct texts). Near-perfect sentiment and complaint scores show that these labels follow the wording patterns in this dataset; "
          "they do not prove the same accuracy on free-form real-world feedback.",
          "- Priority labels are noisy relative to the text (identical texts carry different priorities), so its scores are far below the other two tasks by construction.",
          "- Test rows repeat texts, so row-level confidence intervals would be too narrow; the bootstrap above resamples whole texts.", ""]
    text = "\n".join(L)
    cfg.EVAL_REPORT_MD_PATH.write_text(text, encoding="utf-8")
    return text


# ----------------------------------------------------------------------------------------
# 7. Save + verify
# ----------------------------------------------------------------------------------------
def verify_reload(df: pd.DataFrame, sp: Splits, expected: dict) -> dict:
    """Reload every saved artifact and confirm it reproduces the in-memory predictions."""
    import tensorflow as tf
    sample = df.iloc[sp.test].sample(300, random_state=cfg.RANDOM_SEED)
    texts = sample["feedback_text"]
    checks = {}
    for key, path in [("sentiment", cfg.SENTIMENT_MODEL_PATH), ("complaint_type", cfg.COMPLAINT_MODEL_PATH)]:
        pipe = joblib.load(path)
        checks[key] = bool((pipe.predict(texts) == expected[key]).all())
    pre = joblib.load(cfg.PRIORITY_PREPROCESSOR_PATH)
    ann = tf.keras.models.load_model(cfg.PRIORITY_ANN_PATH)
    proba = ann.predict(pre.transform(texts).astype("float32"), verbose=0)
    checks["priority_ann"] = bool(np.allclose(proba, expected["priority_proba"], atol=1e-5))
    checks["priority_ann_probabilities_sum_to_one"] = bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4))
    checks["priority_baseline"] = bool((joblib.load(cfg.PRIORITY_BASELINE_PATH).predict(texts) == expected["priority_baseline"]).all())
    checks["all_ok"] = all(checks.values())
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train all Customer Feedback Analyzer models.")
    ap.add_argument("--skip-figures", action="store_true", help="skip PNG generation")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()
    set_seeds()
    for d in (cfg.MODELS_DIR, cfg.REPORTS_DIR, cfg.FIGURES_DIR, cfg.CM_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log.info("1/8 Loading and validating dataset")
    df, validation = load_and_validate()
    sp = make_splits(df)
    log.info("   rows=%d  train=%d  test=%d  ann_fit=%d  ann_val=%d", len(df), len(sp.train), len(sp.test), len(sp.fit), len(sp.val))

    log.info("2/8 Exploratory data analysis")
    eda = eda_summary(df, validation, sp)
    cfg.EDA_JSON_PATH.write_text(json.dumps(eda, indent=2), encoding="utf-8")
    if not args.skip_figures:
        make_eda_figures(df, eda)

    tasks, fitted = {}, {}
    for i, target in enumerate(["sentiment", "complaint_type", "priority"], start=3):
        log.info("%d/8 Training scikit-learn models for %s", i if i < 5 else 5, target)
        res = run_text_task(target, df, sp)
        fitted[target] = res.pop("_fitted")
        tasks[target] = res

    log.info("6/8 Evaluating structured features (channel, customer_type, order_value, days_as_customer)")
    fe = structured_feature_experiment(df, sp, tasks)

    log.info("7/8 Training Keras ANN for priority (%d seeds)", len(ANN_SEEDS))
    pre = build_priority_preprocessor().fit(df["feedback_text"].iloc[sp.fit])      # fitted on ANN-training rows only
    Xfit, Xval, Xte = (pre.transform(df["feedback_text"].iloc[i]).astype("float32") for i in (sp.fit, sp.val, sp.test))
    yfit, yval = encode_priority(df["priority"].iloc[sp.fit]), encode_priority(df["priority"].iloc[sp.val])
    yte_lab = df["priority"].iloc[sp.test].values
    seed_runs, ann_models, ann_preds = [], {}, {}
    for seed in ANN_SEEDS:
        model, hist = train_ann(Xfit, yfit, Xval, yval, seed)
        proba = model.predict(Xte, verbose=0)
        pred = np.array(cfg.PRIORITY_ORDER)[proba.argmax(axis=1)]
        ev = evaluate(yte_lab, pred, cfg.PRIORITY_ORDER)
        seed_runs.append({"seed": seed, "best_val_loss": min(hist["val_loss"]), "accuracy": ev["accuracy"], "macro_f1": ev["macro_f1"],
                          "best_epoch": hist["best_epoch"], "epochs_run": hist["epochs_run"]})
        ann_preds[seed] = (pred, ev, hist, proba)
        ann_models[seed] = model
        log.info("   seed %-5d best_epoch=%-3d val_loss=%.4f test acc=%.4f macro-F1=%.4f", seed, hist["best_epoch"], min(hist["val_loss"]), ev["accuracy"], ev["macro_f1"])
    # The deployed network is the seed with the lowest VALIDATION loss (never chosen on test data).
    deployed_seed = min(seed_runs, key=lambda r: r["best_val_loss"])["seed"]
    deployed_model = ann_models[deployed_seed]
    pred0, ev0, hist0, proba0 = ann_preds[deployed_seed]
    log.info("   deployed seed (lowest validation loss): %d", deployed_seed)

    p = tasks["priority"]
    base_name = p["selected"]
    base_pred = fitted["priority"][base_name].predict(df["feedback_text"].iloc[sp.test])
    bs = grouped_bootstrap(yte_lab, {"Keras ANN": pred0, base_name: base_pred}, sp.groups[sp.test])
    accs = [r["accuracy"] for r in seed_runs]
    f1s = [r["macro_f1"] for r in seed_runs]
    p["ann"] = {
        "input": "TF-IDF (2,000-term bounded vocabulary, 1-2 grams) reduced to 256 dimensions with TruncatedSVD",
        "architecture": "256 -> Dense(128, ReLU) -> Dropout(0.3) -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(3, softmax)",
        "loss": "sparse_categorical_crossentropy", "optimizer": "Adam(lr=1e-3) + ReduceLROnPlateau",
        "parameters": int(deployed_model.count_params()), "deployed_seed": deployed_seed,
        "test": ev0, "seed_runs": seed_runs,
        "seed_summary": {"accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
                         "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s))},
        "history_deployed": hist0,
    }
    p["comparison"] = {"baseline": base_name, "bootstrap": bs}

    log.info("8/8 Saving artifacts and verifying reload")
    joblib.dump(fitted["sentiment"][tasks["sentiment"]["selected"]], cfg.SENTIMENT_MODEL_PATH, compress=3)
    joblib.dump(fitted["complaint_type"][tasks["complaint_type"]["selected"]], cfg.COMPLAINT_MODEL_PATH, compress=3)
    joblib.dump(fitted["priority"][base_name], cfg.PRIORITY_BASELINE_PATH, compress=3)
    joblib.dump(pre, cfg.PRIORITY_PREPROCESSOR_PATH, compress=3)
    deployed_model.save(cfg.PRIORITY_ANN_PATH)

    results = {
        "meta": {"trained_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "seed": cfg.RANDOM_SEED,
                 "split": eda["split_sizes"], "cv_folds": cfg.CV_FOLDS},
        "tasks": tasks, "structured_feature_experiment": fe,
    }
    probe = df.iloc[sp.test].sample(300, random_state=cfg.RANDOM_SEED)
    pr_pre = pre.transform(probe["feedback_text"]).astype("float32")
    expected = {
        "sentiment": fitted["sentiment"][tasks["sentiment"]["selected"]].predict(probe["feedback_text"]),
        "complaint_type": fitted["complaint_type"][tasks["complaint_type"]["selected"]].predict(probe["feedback_text"]),
        "priority_proba": deployed_model.predict(pr_pre, verbose=0),
        "priority_baseline": fitted["priority"][base_name].predict(probe["feedback_text"]),
    }
    reload_checks = verify_reload(df, sp, expected)
    log.info("   reload checks: %s", reload_checks)
    results["reload_verification"] = reload_checks

    import tensorflow as tf
    metadata = {
        "project": "Customer Feedback Analyzer",
        "trained_at_utc": results["meta"]["trained_at_utc"],
        "input_schema": {"feedback_text": "str (text only; structured fields were evaluated and not adopted)"},
        "max_input_chars": cfg.MAX_INPUT_CHARS,
        "labels": {"sentiment": tasks["sentiment"]["labels"], "complaint_type": tasks["complaint_type"]["labels"],
                   "priority": cfg.PRIORITY_ORDER},
        "models": {
            "sentiment": {"file": cfg.SENTIMENT_MODEL_PATH.name, "algorithm": tasks["sentiment"]["selected"], "supports_probabilities": True,
                          "best_params": tasks["sentiment"]["tuning"]["best_params"]},
            "complaint_type": {"file": cfg.COMPLAINT_MODEL_PATH.name, "algorithm": tasks["complaint_type"]["selected"], "supports_probabilities": True,
                               "best_params": tasks["complaint_type"]["tuning"]["best_params"]},
            "priority": {"file": cfg.PRIORITY_ANN_PATH.name, "preprocessor": cfg.PRIORITY_PREPROCESSOR_PATH.name,
                         "algorithm": "Keras feed-forward ANN", "architecture": p["ann"]["architecture"], "supports_probabilities": True,
                         "class_order": cfg.PRIORITY_ORDER, "baseline_file": cfg.PRIORITY_BASELINE_PATH.name, "baseline_algorithm": base_name},
        },
        "test_metrics": {
            "sentiment": {k: tasks["sentiment"]["candidates"][tasks["sentiment"]["selected"]]["test"][k] for k in ("accuracy", "macro_f1")},
            "complaint_type": {k: tasks["complaint_type"]["candidates"][tasks["complaint_type"]["selected"]]["test"][k] for k in ("accuracy", "macro_f1")},
            "priority_ann": {k: ev0[k] for k in ("accuracy", "macro_f1")},
            "priority_baseline": {k: p["candidates"][base_name]["test"][k] for k in ("accuracy", "macro_f1")},
        },
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__,
                        "pandas": pd.__version__, "tensorflow": tf.__version__, "joblib": joblib.__version__},
    }
    cfg.METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    cfg.RESULTS_JSON_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not args.skip_figures:
        make_result_figures(results)
    write_report(results, eda)
    log.info("Done in %.0f s. Reload verification all_ok=%s", time.time() - t0, reload_checks["all_ok"])
    return 0 if reload_checks["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
