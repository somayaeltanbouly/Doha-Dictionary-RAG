"""
retrieval_pipeline.py — Full retrieval pipeline: index → evaluate
=============================================================

Runs both retrieval stages in order from a single command:

  1. **Index**    — Embed the corpus and build a FAISS index.
                    Skipped if output files already exist (pass ``--force``
                    to rebuild).

  2. **Evaluate** — Run retrieval over the labelled evaluation queries and
                    report Recall@K, MRR, and MAP.

Select which stages to run with ``--stages`` (default: both).
Select which retrieval methods to evaluate with ``--methods``.

Usage::

    # Full pipeline with defaults: build index if needed, then evaluate hybrid
    python src/retrieval/retrieval_pipeline.py

    # Skip indexing, evaluate all three methods
    python src/retrieval/retrieval_pipeline.py --stages eval --methods bm25 dense hybrid

    # Force re-embed, then evaluate BM25 without reranking
    python src/retrieval/retrieval_pipeline.py \\
        --stages index eval --force --methods bm25 --cross-encoder none

    # Use a different embedding model
    python src/retrieval/retrieval_pipeline.py --embedding-model models/bge-m3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running standalone: python src/retrieval/retrieval_pipeline.py
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.retrieval.build_index import build as _build_index, parse_args as _bi_parse
from src.retrieval.evaluate_retrieval import main as _evaluate, parse_args as _ev_parse


# ── Helpers ───────────────────────────────────────────────────────── #

def _model_tag(model_path: str) -> str:
    """Derive the two-part tag embedded in output file names.

    Examples::
        "nomic-ai/nomic-embed-text-v2-moe"  →  "nomic-embed"
        "models/nomic"                       →  "nomic"
    """
    basename = Path(model_path).name
    parts = basename.split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else basename


# ── Stage functions ───────────────────────────────────────────────── #

def run_index(args: argparse.Namespace, tag: str) -> None:
    """Embed the corpus and build the FAISS index."""
    emb_out = str(Path(args.emb_out_dir) / f"embeddings_{tag}.npy")
    idx_out = str(Path(args.emb_out_dir) / f"faiss_{tag}.index")

    print(f"\n{'='*60}")
    print("  STAGE: build-index")
    print(f"  model  : {args.embedding_model}")
    print(f"  corpus : {args.corpus_text}")
    print(f"  emb    : {emb_out}")
    print(f"  index  : {idx_out}")
    print(f"{'='*60}\n")

    index_args = _bi_parse([
        "--corpus",     args.corpus_text,
        "--model",      args.embedding_model,
        "--batch-size", str(args.batch_size),
        "--index-type", args.index_type,
        "--emb-out",    emb_out,
        "--idx-out",    idx_out,
    ] + (["--force"] if args.force else [])
      + (["--device", args.device] if args.device else []))

    _build_index(index_args)


def run_eval(args: argparse.Namespace, method: str, tag: str) -> None:
    """Run retrieval evaluation for one method."""
    emb_path = str(Path(args.emb_out_dir) / f"embeddings_{tag}.npy")
    idx_path = str(Path(args.emb_out_dir) / f"faiss_{tag}.index")
    out_path = str(Path(args.eval_out_dir) / f"{method}_results.csv")

    print(f"\n{'='*60}")
    print(f"  STAGE: evaluate-{method}")
    print(f"  eval data  : {args.eval_data}")
    print(f"  output     : {out_path}")
    print(f"{'='*60}\n")

    eval_args = _ev_parse([
        "--method",          method,
        "--eval-data",       args.eval_data,
        "--corpus",          args.corpus,
        "--text-data",       args.corpus_text,
        "--embeddings",      emb_path,
        "--index",           idx_path,
        "--embedding-model", args.embedding_model,
        "--classifier",      args.classifier,
        "--cross-encoder",   args.cross_encoder,
        "--top-k",           str(args.top_k),
        "--k-bm25",          str(args.k_bm25),
        "--k-rrf",           str(args.k_rrf),
        "--k-rerank",        str(args.k_rerank),
        "--output",          out_path,
    ])

    _evaluate(eval_args)


# ── Main ──────────────────────────────────────────────────────────── #

def main(args: argparse.Namespace) -> None:
    tag     = _model_tag(args.embedding_model)
    stages  = set(args.stages)
    methods = args.methods

    if "index" in stages:
        run_index(args, tag)

    if "eval" in stages:
        for method in methods:
            run_eval(args, method, tag)

    print("\nPipeline complete.")


# ── CLI ───────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the full retrieval pipeline: "
            "embed corpus → build FAISS index → evaluate retrieval."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Stage selection ──────────────────────────────────────────── #
    p.add_argument(
        "--stages",
        nargs="+",
        choices=["index", "eval"],
        default=["index", "eval"],
        help="Pipeline stages to run. Use 'index' to embed/build and "
             "'eval' to run evaluation. Both run by default.",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["bm25", "dense", "hybrid"],
        default=["hybrid"],
        help="Retrieval methods to evaluate (used only when 'eval' stage is active).",
    )

    # ── Data paths ───────────────────────────────────────────────── #
    p.add_argument(
        "--corpus",
        default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv",
        help="Structured corpus CSV (must have an 'ID' column).",
    )
    p.add_argument(
        "--corpus-text",
        default="data/retrieval_corpus/processed/DHDA_text_to_embed.csv",
        dest="corpus_text",
        help="Plain-text corpus CSV (column 'text') for BM25 and embedding.",
    )
    p.add_argument(
        "--eval-data",
        default="data/questions_data/QD_data/QD_evaluation_data.csv",
        dest="eval_data",
        help="Evaluation CSV with query-document relevance pairs "
             "(columns: query_id, question, doc_id, …).",
    )

    # ── Output directories ───────────────────────────────────────── #
    p.add_argument(
        "--emb-out-dir",
        default="data/retrieval_corpus/vector_database",
        dest="emb_out_dir",
        help="Directory for embeddings .npy and .index files.",
    )
    p.add_argument(
        "--eval-out-dir",
        default="data/retrieval_corpus/evaluation_results",
        dest="eval_out_dir",
        help="Directory for per-method evaluation result CSVs.",
    )

    # ── Model paths ──────────────────────────────────────────────── #
    p.add_argument(
        "--embedding-model",
        default="models/nomic",
        dest="embedding_model",
        help="SentenceTransformer model path/ID for corpus embedding and query encoding.",
    )
    p.add_argument(
        "--classifier",
        default="models/RF_intent_classifier.joblib",
        help="Intent classifier (.joblib). Pass 'none' to skip intent analysis.",
    )
    p.add_argument(
        "--cross-encoder",
        default="models/finetuned_CE_bge",
        dest="cross_encoder",
        help="Cross-encoder model for reranking. Pass 'none' to disable.",
    )

    # ── Indexing options ─────────────────────────────────────────── #
    p.add_argument(
        "--index-type",
        default="Flat",
        dest="index_type",
        choices=["Flat", "IVFFlat", "HNSW"],
        help="FAISS index type (passed to build_index.py).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        dest="batch_size",
        help="Encoding batch size (passed to build_index.py).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Compute device for embedding ('cuda' or 'cpu'). Auto-detected if omitted.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-embedding and index rebuild even if outputs already exist.",
    )

    # ── Retrieval / evaluation options ───────────────────────────── #
    p.add_argument("--top-k",    type=int, default=10,  dest="top_k")
    p.add_argument("--k-bm25",   type=int, default=50,  dest="k_bm25")
    p.add_argument("--k-rrf",    type=int, default=300, dest="k_rrf")
    p.add_argument("--k-rerank", type=int, default=50,  dest="k_rerank")

    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
