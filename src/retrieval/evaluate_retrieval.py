"""
evaluate_retrieval.py — Evaluate BM25 / dense / hybrid retrieval
=================================================================

Runs a retrieval method against a set of labelled evaluation queries and
reports Recall@K, MRR, and MAP against ground-truth document IDs.

The evaluation data file (``--eval-data``) must be a CSV with columns::

    type, question, answer, doc_id, query_id, text

Each unique ``query_id`` is one evaluation query.  Multiple rows with the
same ``query_id`` but different ``doc_id`` values mean that query has more
than one relevant document — all are used as ground truth.

Retrieval methods (``--method``)::

    bm25    — BM25 keyword retrieval
    dense   — Dense vector retrieval (FAISS)
    hybrid  — BM25 + dense with RRF fusion (default)

Pass ``--cross-encoder none`` to evaluate without reranking.

Output
------
A CSV at ``--output`` with one row per query (``qid``, ``docs``, ``ground_truth``),
plus a summary of Recall@K / MRR / MAP printed to stdout.

Usage::

    # default: hybrid retrieval with reranking
    python src/retrieval/evaluate_retrieval.py

    # BM25 only, no reranking
    python src/retrieval/evaluate_retrieval.py --method bm25 --cross-encoder none

    # dense retrieval, top-5
    python src/retrieval/evaluate_retrieval.py --method dense --top-k 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Allow running standalone: python src/retrieval/evaluate_retrieval.py
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.retrieval.retriever import HybridRetriever
from src.retrieval.metrics import recall_at_k, compute_mrr, compute_map


def main(args: argparse.Namespace) -> None:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cross_encoder_path = (
        None if args.cross_encoder.lower() == "none" else args.cross_encoder
    )

    retriever = HybridRetriever(
        corpus_path          = args.corpus,
        text_data_path       = args.text_data,
        embeddings_path      = args.embeddings,
        index_path           = args.index,
        classifier_path      = args.classifier,
        cross_encoder_path   = cross_encoder_path,
        embedding_model_path = args.embedding_model,
        top_k    = args.top_k,
        k_bm25   = args.k_bm25,
        k_rrf    = args.k_rrf,
        k_rerank = args.k_rerank,
    )

    # ── Load evaluation data ─────────────────────────────────────── #
    df_eval = pd.read_csv(args.eval_data)
    _required = {"query_id", "question", "doc_id"}
    if not _required.issubset(df_eval.columns):
        raise ValueError(
            f"--eval-data must contain columns {_required}. "
            f"Found: {list(df_eval.columns)}"
        )

    # Ground truth: map each query_id → sorted list of relevant doc_ids
    truth_map: dict[int, list] = (
        df_eval.groupby("query_id")["doc_id"].apply(list).to_dict()
    )

    # One row per unique query (drop duplicate query_ids keeping first)
    df_queries = (
        df_eval[["query_id", "question"]]
        .drop_duplicates(subset="query_id")
        .reset_index(drop=True)
    )

    print(
        f"\nEvaluating {args.method.upper()} on {len(df_queries)} queries "
        f"(top_k={args.top_k}) …\n"
    )

    # ── Write CSV header ─────────────────────────────────────────── #
    pd.DataFrame(columns=["qid", "docs", "ground_truth"]).to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )

    results: list[dict] = []
    for _, row in tqdm(df_queries.iterrows(), total=len(df_queries),
                       desc=f"{args.method.upper()} eval"):
        query_info = retriever.analyze_query(row["question"])
        _, final_indices = retriever.retrieve(query_info["q1"], method=args.method)

        qid = int(row["query_id"])
        retrieved_doc_ids = [
            int(retriever.corpus_df.loc[idx, "ID"])
            if idx in retriever.corpus_df.index and "ID" in retriever.corpus_df.columns
            else None
            for idx in final_indices
        ]
        entry = {
            "qid":          qid,
            "docs":         retrieved_doc_ids,
            "ground_truth": truth_map.get(qid, []),
        }
        results.append(entry)
        # Append row immediately so partial results survive interruption
        pd.DataFrame([entry]).to_csv(
            out_path, mode="a", header=False, index=False, encoding="utf-8-sig"
        )

    print(f"\nResults saved to {out_path}  ({len(results)} queries)")
    print(f"Recall@{args.top_k}: {recall_at_k(results, args.top_k):.4f}")
    print(f"MRR:                {compute_mrr(results):.4f}")
    print(f"MAP:                {compute_map(results):.4f}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate retrieval (BM25 / dense / hybrid) on labelled QD pairs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid"],
        default="hybrid",
        help="Retrieval method to evaluate.",
    )
    p.add_argument(
        "--eval-data",
        default="data/questions_data/QD_data/QD_evaluation_data.csv",
        dest="eval_data",
        help="CSV with query-document pairs (columns: query_id, question, doc_id, …). "
             "Produced by build_qd_pairs.py.",
    )
    p.add_argument(
        "--corpus",
        default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv",
        help="Structured corpus CSV used to resolve doc IDs (must have an 'ID' column).",
    )
    p.add_argument(
        "--text-data",
        default="data/retrieval_corpus/processed/DHDA_text_to_embed.csv",
        dest="text_data",
        help="Plain-text CSV (column 'text') used for BM25 and cross-encoder scoring.",
    )
    p.add_argument(
        "--embeddings",
        default="data/retrieval_corpus/vector_database/embeddings_nomic-embed.npy",
        help="Pre-computed corpus embeddings (.npy) produced by build_index.py.",
    )
    p.add_argument(
        "--index",
        default="data/retrieval_corpus/vector_database/faiss_nomic-embed.index",
        help="Serialised FAISS index produced by build_index.py.",
    )
    p.add_argument(
        "--embedding-model",
        default="models/nomic",
        dest="embedding_model",
        help="SentenceTransformer model path / ID used to encode queries.",
    )
    p.add_argument(
        "--classifier",
        default="models/RF_intent_classifier.joblib",
        help="Trained intent classifier (.joblib). Pass 'none' to skip intent analysis.",
    )
    p.add_argument(
        "--cross-encoder",
        default="models/finetuned_CE_bge",
        dest="cross_encoder",
        help="Path to fine-tuned cross-encoder for reranking. "
             "Pass 'none' to disable reranking.",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to "
             "data/retrieval_corpus/evaluation_results/<method>_results.csv.",
    )
    p.add_argument("--top-k",    type=int, default=10, dest="top_k",
                   help="Number of documents to retrieve per query.")
    p.add_argument("--k-bm25",   type=int, default=50,  dest="k_bm25",
                   help="Candidate pool size for BM25/dense single-method strategies.")
    p.add_argument("--k-rrf",    type=int, default=300, dest="k_rrf",
                   help="Candidate pool per method before RRF fusion (hybrid only).")
    p.add_argument("--k-rerank", type=int, default=50,  dest="k_rerank",
                   help="Top-RRF candidates forwarded to the cross-encoder.")

    args = p.parse_args(argv)
    if args.output is None:
        args.output = (
            f"data/retrieval_corpus/evaluation_results/{args.method}_results.csv"
        )
    return args


if __name__ == "__main__":
    main(parse_args())
