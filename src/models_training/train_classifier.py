"""
train_classifier.py — Train the intent classifier
==================================================

Trains a scikit-learn ``Pipeline(TfidfVectorizer + RandomForestClassifier)``
on Arabic question-type labels.  The pipeline and its defaults reproduce the
classifier saved at ``models/RF_intent_classifier.joblib``.

A confidence-threshold rule (default 0.60) maps low-confidence predictions to
the ``"other"`` label at evaluation time.  The trained pipeline — without the
threshold applied — is serialised with ``joblib`` and can be loaded directly
by the inference layer.

Pipeline
--------
1. Load ``classification_data.csv`` (columns: ``type``, ``question``).
2. Strip Arabic diacritics (tashkeel) from every question.
3. Stratified 80 / 20 train / test split, seeded for reproducibility.
4. Fit a ``Pipeline([TfidfVectorizer(char-ngrams 3–5), RandomForestClassifier])``.
5. Evaluate with confidence-threshold prediction and print a classification report.
6. Save the pipeline to ``models/RF_intent_classifier.joblib``.

Defaults
--------
- Data            : ``data/questions_data/classification_data/classification_data.csv``
- Output          : ``models/RF_intent_classifier.joblib``
- n_estimators    : 200
- ngram_range     : (3, 5)  (character-level)
- max_features    : 30 000
- test_size       : 0.20
- threshold       : 0.60  (below → ``"other"``)
- random_state    : 42

Usage::

    # from the repo root
    python src/models_training/train_classifier.py

    # custom data / output
    python src/models_training/train_classifier.py \\
        --data      data/questions_data/classification_data/classification_data.csv \\
        --output    models/RF_intent_classifier.joblib \\
        --n-estimators 200 --threshold 0.6
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ── Tashkeel import ───────────────────────────────────────────────── #
# Try the shared data_utils module first (broader diacritic coverage).
# Fall back to the minimal pattern used in the original classifier notebook.

try:
    _src_dir = Path(__file__).parent.parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    from data_processing.data_utils import strip_tashkeel as _strip_tashkeel
except ImportError:
    import re as _re
    _TASHKEEL_RE = _re.compile(r"[\u064B-\u0652]")
    def _strip_tashkeel(text: str) -> str:  # type: ignore[misc]
        return _TASHKEEL_RE.sub("", text)


logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Data helpers ──────────────────────────────────────────────────── #

def _load_data(csv_path: str) -> pd.DataFrame:
    """Load classification CSV and strip tashkeel from the question column."""
    df = pd.read_csv(csv_path)
    required = {"type", "question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[["type", "question"]].dropna()
    df["question"] = df["question"].apply(
        lambda t: _strip_tashkeel(str(t)) if pd.notna(t) else t
    )
    logger.info("Loaded %d samples  |  classes: %s", len(df),
                sorted(df["type"].unique().tolist()))
    return df


# ── Threshold predictor ───────────────────────────────────────────── #

def _predict_with_threshold(
    pipeline: Pipeline,
    X: pd.Series,
    threshold: float,
    fallback: str = "other",
) -> list[str]:
    """
    Return predicted labels, substituting *fallback* wherever the maximum
    class probability falls below *threshold*.
    """
    probs  = pipeline.predict_proba(X)
    labels = pipeline.classes_[np.argmax(probs, axis=1)]
    confs  = np.max(probs, axis=1)
    return [lbl if c >= threshold else fallback for lbl, c in zip(labels, confs)]


# ── Main ──────────────────────────────────────────────────────────── #

def train(args: argparse.Namespace) -> None:
    """Orchestrate the full training pipeline from data loading to model serialisation."""
    # 1. Load data
    df = _load_data(args.data)

    # 2. Stratified train / test split — stratify ensures class proportions are
    #    preserved in both splits, which matters for imbalanced question-type data.
    X_train, X_test, y_train, y_test = train_test_split(
        df["question"],
        df["type"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["type"],
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # 3. Build pipeline — TF-IDF on character n-grams works well for Arabic because
    #    it is robust to morphological variation and does not require tokenisation.
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=tuple(args.ngram_range),
                max_features=args.max_features,
            ),
        ),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=args.n_estimators,
                random_state=args.random_state,
            ),
        ),
    ])

    # 4. Train
    logger.info(
        "Training Pipeline(TfidfVectorizer[char, ngram=%s, max_features=%d] + "
        "RandomForest[n_estimators=%d]) …",
        args.ngram_range, args.max_features, args.n_estimators,
    )
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    # 5. Evaluate with confidence threshold
    logger.info(
        "Evaluating on test set with confidence threshold %.2f …", args.threshold
    )
    final_preds = _predict_with_threshold(
        pipeline, X_test, threshold=args.threshold
    )

    # Collect all possible labels (including "other" if it appears)
    all_labels = sorted(set(list(y_test.unique()) + (
        ["other"] if args.threshold < 1.0 else []
    )))
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT  (threshold={:.2f})".format(args.threshold))
    print("=" * 60)
    print(classification_report(y_test, final_preds, labels=all_labels,
                                 zero_division=0))

    # 6. Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, str(output_path))
    logger.info("Pipeline saved → %s", output_path)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Arabic question-type intent classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default="data/questions_data/classification_data/classification_data.csv",
        help="CSV with columns: type, question.",
    )
    p.add_argument(
        "--output",
        default="models/RF_intent_classifier.joblib",
        help="Path to save the trained joblib pipeline.",
    )
    p.add_argument("--n-estimators", type=int,   default=200,   dest="n_estimators",
                   help="Number of trees in the Random Forest.")
    p.add_argument("--ngram-range",  type=int,   default=[3, 5], nargs=2,
                   dest="ngram_range", metavar=("MIN", "MAX"),
                   help="Character n-gram range for TF-IDF.")
    p.add_argument("--max-features", type=int,   default=30_000, dest="max_features",
                   help="Maximum vocabulary size for TF-IDF.")
    p.add_argument("--test-size",    type=float, default=0.2,   dest="test_size",
                   help="Fraction of data held out for evaluation.")
    p.add_argument("--threshold",    type=float, default=0.6,
                   help="Min confidence to assign a predicted label; "
                        "below this falls back to 'other'.")
    p.add_argument("--random-state", type=int,   default=42,    dest="random_state",
                   help="Random seed for the train/test split and the Random Forest.")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
