"""
CEContrastiveDataGenerator
==========================

Generates contrastive (positive + negative) training pairs for cross-encoder
fine-tuning from ``QD_full_data.csv``.

Pipeline
--------
1. Load ``QD_full_data.csv``.
2. Sample *n_source* rows proportionally by question type
   (stratified, ~equal per type) → the "sampled queries" subset.
3. For every row in the sample:
   - **Positive** pair  (label=1.0): query ↔ its actual document text.
   - **Negative** pair  (label=0.0): query ↔ a randomly chosen document that
     is **not** a known positive for that query (safe-negative strategy from
     the cross-encoder fine-tuning notebook).
4. Save to ``QD_contrastive_data.csv`` with columns:
   ``question, text, label``

The output CSV can be loaded directly by the ``CrossEncoderTrainer`` in
``ce_finetuning.ipynb`` via its ``load_data()`` method.

Defaults
--------
- Source file : ``data/questions_data/QD_data/QD_full_data.csv``
- Output file : ``data/questions_data/QD_data/QD_contrastive_data.csv``
- n_source    : 5000  (→ ~5 000 positives + ~5 000 negatives ≈ 10 000 rows)
- num_negatives: 1
- random_state : 42

Usage::

    python src/data_processing/build_ce_training_data.py

    # explicit paths / sizes:
    python src/data_processing/build_ce_training_data.py \\
        --qd-path   data/questions_data/QD_data/QD_full_data.csv \\
        --output    data/questions_data/QD_data/QD_contrastive_data.csv \\
        --n-source  5000 \\
        --num-negatives 1
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd


# ── Stratified sampler ────────────────────────────────────────────── #

def _stratified_sample(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Return *n* rows sampled proportionally across the ``type`` column."""
    frac = n / len(df)
    groups = [
        grp.sample(frac=frac, random_state=random_state)
        for _, grp in df.groupby("type", sort=False)
    ]
    sampled = pd.concat(groups, ignore_index=True)
    return sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ── Contrastive pair builder ──────────────────────────────────────── #

def build_contrastive_pairs(
    df: pd.DataFrame,
    question_col: str = "question",
    text_col: str = "text",
    query_id_col: str = "query_id",
    doc_id_col: str = "doc_id",
    num_negatives: int = 1,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build positive and safe-negative pairs from a QD DataFrame.

    Each row produces one positive (label=1.0) and *num_negatives* negatives
    (label=0.0).  Negatives are drawn from documents that are **not** already
    a known positive for that query, ensuring no false negatives.

    Returns a DataFrame with columns ``question, text, label``.
    """
    random.seed(random_state)

    # Normalise id columns to str for consistent lookup
    df = df.copy()
    df[query_id_col] = df[query_id_col].astype(str).str.strip()
    df[doc_id_col]   = df[doc_id_col].astype(str).str.strip()
    df = df.dropna(subset=[question_col, text_col, query_id_col, doc_id_col])

    # Build query → positive-doc and doc → text mappings
    query_to_pos_docs: dict[str, set[str]] = defaultdict(set)
    doc_to_texts: dict[str, list[str]]     = defaultdict(list)

    for _, row in df.iterrows():
        qid = row[query_id_col]
        did = row[doc_id_col]
        query_to_pos_docs[qid].add(did)
        t = str(row[text_col])
        if t not in doc_to_texts[did]:
            doc_to_texts[did].append(t)

    all_docs = list(doc_to_texts.keys())

    rows: list[dict] = []

    for _, row in df.iterrows():
        qid   = row[query_id_col]
        did   = row[doc_id_col]
        qtext = str(row[question_col])
        dtext = str(row[text_col])

        # Positive
        rows.append({"question": qtext, "text": dtext, "label": 1.0})

        # Safe negatives
        pos_docs = query_to_pos_docs[qid]
        neg_candidates = [d for d in all_docs if d not in pos_docs]

        if not neg_candidates:
            continue

        chosen = random.sample(neg_candidates, min(num_negatives, len(neg_candidates)))
        for neg_did in chosen:
            texts = doc_to_texts[neg_did]
            if not texts:
                continue
            rows.append({"question": qtext, "text": random.choice(texts), "label": 0.0})

    result = pd.DataFrame(rows, columns=["question", "text", "label"])
    n_pos = (result["label"] == 1.0).sum()
    n_neg = (result["label"] == 0.0).sum()
    print(f"[contrastive] {len(result):,} pairs  ({n_pos:,} positive, {n_neg:,} negative)")
    return result


# ── Main class ────────────────────────────────────────────────────── #

class CEContrastiveDataGenerator:
    """
    Sample QD pairs and build contrastive training data for cross-encoder
    fine-tuning.
    """

    def __init__(
        self,
        qd_path: str | Path,
        output_path: str | Path = "data/questions_data/QD_data/QD_contrastive_data.csv",
        n_source: int = 5000,
        num_negatives: int = 1,
        random_state: int = 42,
    ) -> None:
        self.qd_path       = Path(qd_path)
        self.output_path   = Path(output_path)
        self.n_source      = n_source
        self.num_negatives = num_negatives
        self.random_state  = random_state
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """Execute the full pipeline and save ``QD_contrastive_data.csv``."""
        df = pd.read_csv(self.qd_path, encoding="utf-8-sig")
        print(f"[load]   {len(df):,} rows from {self.qd_path}")

        # Step 1: stratified sample → source QD subset
        sampled = _stratified_sample(df, n=self.n_source, random_state=self.random_state)
        print(f"[sample] {len(sampled):,} rows sampled (target {self.n_source})")
        print(f"         type distribution: {sampled['type'].value_counts().to_dict()}")

        # Step 2: build contrastive pairs
        contrastive = build_contrastive_pairs(
            sampled,
            num_negatives=self.num_negatives,
            random_state=self.random_state,
        )

        # Step 3: save
        contrastive.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        print(f"[save]   {len(contrastive):,} rows → {self.output_path}")
        return contrastive


# ── CLI ───────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build contrastive CE training data from QD_full_data.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--qd-path",
        default="data/questions_data/QD_data/QD_full_data.csv",
        help="Path to QD_full_data.csv.",
    )
    p.add_argument(
        "--output",
        default="data/questions_data/QD_data/QD_contrastive_data.csv",
        help="Output path for QD_contrastive_data.csv.",
    )
    p.add_argument(
        "--n-source",
        type=int,
        default=5000,
        help="Number of QD rows to sample before building pairs (5000 → ~10k output rows).",
    )
    p.add_argument(
        "--num-negatives",
        type=int,
        default=1,
        help="Number of negative documents per query-doc pair.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    gen = CEContrastiveDataGenerator(
        qd_path=args.qd_path,
        output_path=args.output,
        n_source=args.n_source,
        num_negatives=args.num_negatives,
        random_state=args.random_state,
    )
    gen.run()
