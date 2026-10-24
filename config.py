"""Central configuration: relative paths, seeds and label orders.

Every path is derived from this file's location, so the project works from any
folder on any computer (no hard-coded absolute paths).
"""
from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "overall_capstone_customer_feedback_dataset_30000.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
CM_DIR = REPORTS_DIR / "confusion_matrices"
ASSETS_DIR = BASE_DIR / "assets"
CSS_PATH = ASSETS_DIR / "custom_styles.css"
LOG_DIR = BASE_DIR / "logs"

SENTIMENT_MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
COMPLAINT_MODEL_PATH = MODELS_DIR / "complaint_pipeline.joblib"
PRIORITY_ANN_PATH = MODELS_DIR / "priority_ann.keras"
PRIORITY_PREPROCESSOR_PATH = MODELS_DIR / "priority_preprocessor.joblib"
PRIORITY_BASELINE_PATH = MODELS_DIR / "priority_baseline_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

RESULTS_JSON_PATH = REPORTS_DIR / "results.json"
EDA_JSON_PATH = REPORTS_DIR / "eda_summary.json"
EVAL_REPORT_MD_PATH = REPORTS_DIR / "evaluation_report.md"

REQUIRED_COLUMNS = [
    "feedback_id", "feedback_text", "sentiment", "complaint_type", "priority",
    "channel", "customer_type", "order_value", "days_as_customer",
]
TARGETS = ["sentiment", "complaint_type", "priority"]

# Only the labels the specification fixes are hard-coded. The complaint
# categories are read from the dataset at training time and stored in
# models/model_metadata.json, never assumed.
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
PRIORITY_ORDER = ["Low", "Medium", "High"]

RANDOM_SEED = 42
TEST_FRACTION_FOLDS = 5          # 1 of 5 folds -> ~20 % held-out test set
VAL_FRACTION_FOLDS = 8           # 1 of 8 folds of the training data -> ANN validation set
CV_FOLDS = 5

MAX_INPUT_CHARS = 1000

# One colour language shared by the offline report figures and the web app.
COLORS = {
    "bg": "#0b1020",
    "accent": "#6366f1",
    "accent_2": "#8b5cf6",
    "positive": "#22c55e",
    "neutral": "#94a3b8",
    "negative": "#ef4444",
    "low": "#38bdf8",
    "medium": "#f59e0b",
    "high": "#ef4444",
}
SENTIMENT_COLORS = {"Positive": COLORS["positive"], "Neutral": COLORS["neutral"], "Negative": COLORS["negative"]}
PRIORITY_COLORS = {"Low": COLORS["low"], "Medium": COLORS["medium"], "High": COLORS["high"]}
CATEGORY_PALETTE = ["#6366f1", "#8b5cf6", "#3b82f6", "#06b6d4", "#14b8a6", "#a855f7", "#0ea5e9", "#818cf8"]
