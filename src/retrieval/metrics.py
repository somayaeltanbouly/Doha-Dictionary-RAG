"""
metrics.py — Retrieval evaluation metrics
==========================================

Shared metric functions used by ``evaluate_retrieval.py``.

All functions take a list of result dicts, each with keys:

* ``docs``         — list of retrieved document IDs (ranked, highest first).
* ``ground_truth`` — list of relevant document IDs for this query.
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd


def _to_list(value) -> list:
    """Normalise any value (list, serialised string, scalar, NaN) into a list."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        # Strip numpy scalar wrappers that can appear after round-tripping through CSV
        cleaned = re.sub(r"np\.\w+\((\d+)\)", r"\1", value)
        try:
            parsed = ast.literal_eval(cleaned)
            return parsed if isinstance(parsed, list) else [parsed]
        except (ValueError, SyntaxError):
            numbers = re.findall(r"\d+", cleaned)
            return [int(n) for n in numbers] if numbers else []
    return []


def parse_list_column(value) -> list:
    """Parse a CSV column value that was serialised as a Python list string."""
    return _to_list(value)


def recall_at_k(results: list[dict], k: int = 10) -> float:
    """Recall@K averaged across all queries that have at least one relevant doc.

    For each query the recall is ``hits / |relevant|`` where ``hits`` is the
    number of relevant documents found in the top-*k* retrieved results.
    Queries with no ground-truth documents are excluded from the average.
    """
    recalls = []
    for r in results:
        truth = _to_list(r["ground_truth"])
        if not truth:
            continue
        hits = sum(1 for doc in r["docs"][:k] if doc in truth)
        recalls.append(hits / len(truth))
    return float(np.mean(recalls)) if recalls else 0.0


def compute_mrr(results: list[dict]) -> float:
    """Mean Reciprocal Rank: average of ``1/rank`` of the first relevant doc.

    If no relevant document appears in the retrieved list for a query, its
    reciprocal rank contribution is 0.
    """
    rr_scores = []
    for r in results:
        truth = _to_list(r["ground_truth"])
        rr = 0.0
        for rank, doc in enumerate(r["docs"], start=1):
            if doc in truth:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)
    return float(np.mean(rr_scores)) if rr_scores else 0.0


def compute_map(results: list[dict]) -> float:
    """Mean Average Precision (MAP) across all queries.

    AP for each query is computed as::

        AP = sum(P@k for each relevant doc found) / |relevant|

    Queries with no ground-truth documents are excluded from the mean.
    """
    ap_scores = []
    for r in results:
        truth = _to_list(r["ground_truth"])
        if not truth:
            continue
        precision_sum, relevant = 0.0, 0
        for rank, doc in enumerate(r["docs"], start=1):
            if doc in truth:
                relevant += 1
                precision_sum += relevant / rank
        ap_scores.append(precision_sum / len(truth))
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def print_metrics(results: list[dict], k: int = 10, label: str = "") -> None:
    """Print Recall@K, MRR, and MAP to stdout."""
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Recall@{k}: {recall_at_k(results, k):.4f}")
    print(f"{prefix}MRR:       {compute_mrr(results):.4f}")
    print(f"{prefix}MAP:       {compute_map(results):.4f}")
