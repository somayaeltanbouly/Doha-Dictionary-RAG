"""
Doha Dictionary RAG — Single entry point for all pipelines.

Choose a pipeline with the first positional argument; use ``--help`` after the
pipeline name for its specific options.

Available pipelines
-------------------
  Data preparation
    build-corpus          Build BM25 + embedding corpora from raw dictionary files
    build-qa              Generate QA pairs from the corpus
    build-qd              Generate Query-Document pairs for retrieval training
    build-ce-data         Build contrastive training data for the cross-encoder
    build-clf-data        Build labelled data for the intent classifier

  Model training
    train-classifier      Train the intent classifier (Random Forest + TF-IDF)
    finetune-reranker     Fine-tune the cross-encoder reranker (BGE)

  Retrieval
    build-index           Embed the corpus and build a FAISS index
    eval-retrieval        Evaluate retrieval quality (Recall@K, MRR, MAP)

  RAG generation
    generate              Run the full RAG generation pipeline
                          (Fanar / Gemini / ALLaM  ×  few-shot / zero-shot / baseline)

  Evaluation
    judge                 Score model answers with Gemini-as-judge
    summarize-scores      Print statistics for one judging file, or compare two

Examples
--------
    # Data
    python run.py build-corpus
    python run.py build-qa
    python run.py build-qd
    python run.py build-ce-data
    python run.py build-clf-data

    # Training
    python run.py train-classifier
    python run.py finetune-reranker

    # Retrieval
    python run.py build-index
    python run.py eval-retrieval --method hybrid

    # Generation
    python run.py generate --model gemini --mode fs
    python run.py generate --model fanar  --mode zs  --method bm25
    python run.py generate --model hf     --mode baseline

    # Evaluation
    python run.py judge \\
        --input  output/gemini_fs_results.csv \\
        --output judging/gemini_fs_judged.csv

    python run.py summarize-scores --input judging/gemini_fs_judged.csv
    python run.py summarize-scores \\
        --input  judging/fanar_zs_judged.csv \\
        --input2 judging/gemini_fs_judged.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _set_env(key: str, value: str | None) -> None:
    if value:
        os.environ[key] = value


# ──────────────────────────────────────────────────────────────────────────── #
# Data preparation                                                               #
# ──────────────────────────────────────────────────────────────────────────── #

def _sub_build_corpus(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "build-corpus",
        help="Build retrieval corpus (filtered columns CSV + text-to-embed CSV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir",    default="data/retrieval_corpus/raw",
                   help="Directory containing raw dictionary source files.")
    p.add_argument("--output-dir", default="data/retrieval_corpus/processed",
                   help="Output directory for processed corpus files.")
    p.set_defaults(func=_run_build_corpus)


def _run_build_corpus(args: argparse.Namespace) -> None:
    from src.data_processing.build_retrieval_corpus import main
    main(args)


# ─────────────────────────────────────────────────────────────────────────── #

def _sub_build_qa(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "build-qa",
        help="Generate QA pairs from the processed corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus",     default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv",
                   help="Processed corpus CSV.")
    p.add_argument("--output-dir", default="data/questions_data/QA_data",
                   help="Directory to write per-type QA CSVs.")
    p.set_defaults(func=_run_build_qa)


def _run_build_qa(args: argparse.Namespace) -> None:
    from src.data_processing.build_qa_pairs import main
    main(args)


# ─────────────────────────────────────────────────────────────────────────── #

def _sub_build_qd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "build-qd",
        help="Generate Query-Document pairs for retrieval training/evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus",      default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv")
    p.add_argument("--output",      default="data/questions_data/QD_data/QD_full_data.csv")
    p.add_argument("--eval-output", default=None,
                   help="Path for the evaluation split (default: auto-derived).")
    p.add_argument("--n-sample",    type=int, default=1002,
                   help="Number of QD pairs to generate.")
    p.set_defaults(func=_run_build_qd)


def _run_build_qd(args: argparse.Namespace) -> None:
    from src.data_processing.build_qd_pairs import main
    main(args)


# ─────────────────────────────────────────────────────────────────────────── #

def _sub_build_ce_data(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "build-ce-data",
        help="Build contrastive training data for cross-encoder fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--qd-path",       default="data/questions_data/QD_data/QD_full_data.csv")
    p.add_argument("--output",        default="data/questions_data/QD_data/QD_contrastive_data.csv")
    p.add_argument("--n-source",      type=int, default=5000)
    p.add_argument("--num-negatives", type=int, default=1)
    p.add_argument("--random-state",  type=int, default=42)
    p.set_defaults(func=_run_build_ce_data)


def _run_build_ce_data(args: argparse.Namespace) -> None:
    from src.data_processing.build_ce_training_data import main
    main(args)


# ─────────────────────────────────────────────────────────────────────────── #

def _sub_build_clf_data(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "build-clf-data",
        help="Build labelled data for intent classifier training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus",       default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv")
    p.add_argument("--inscriptions", default="data/retrieval_corpus/processed/DHDA_inscriptions_data.csv")
    p.add_argument("--output",       default="data/questions_data/classification_data/classification_data.csv")
    p.add_argument("--n-per-type",   type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.set_defaults(func=_run_build_clf_data)


def _run_build_clf_data(args: argparse.Namespace) -> None:
    from src.data_processing.build_classification_data import main
    main(args)


# ──────────────────────────────────────────────────────────────────────────── #
# Model training                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def _sub_train_classifier(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "train-classifier",
        help="Train the intent classifier (Random Forest + TF-IDF).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",         default="data/questions_data/classification_data/classification_data.csv")
    p.add_argument("--output",       default="models/RF_intent_classifier.joblib")
    p.add_argument("--n-estimators", type=int,   default=200,    dest="n_estimators")
    p.add_argument("--ngram-range",  type=int,   default=[3, 5], nargs=2, dest="ngram_range")
    p.add_argument("--max-features", type=int,   default=30_000, dest="max_features")
    p.add_argument("--test-size",    type=float, default=0.2,    dest="test_size")
    p.add_argument("--threshold",    type=float, default=0.6)
    p.add_argument("--random-state", type=int,   default=42,     dest="random_state")
    p.set_defaults(func=_run_train_classifier)


def _run_train_classifier(args: argparse.Namespace) -> None:
    from src.models_training.train_classifier import main
    main(args)


# ─────────────────────────────────────────────────────────────────────────── #

def _sub_finetune_reranker(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "finetune-reranker",
        help="Fine-tune the cross-encoder reranker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",         default="data/questions_data/QD_data/QD_contrastive_data.csv")
    p.add_argument("--output",       default="models/finetuned_CE_bge")
    p.add_argument("--model-name",   default="BAAI/bge-reranker-v2-m3",  dest="model_name")
    p.add_argument("--max-length",   type=int,   default=512,   dest="max_length")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch-size",   type=int,   default=8,     dest="batch_size")
    p.add_argument("--lr",           type=float, default=5e-5)
    p.add_argument("--warmup-steps", type=int,   default=0,     dest="warmup_steps")
    p.add_argument("--eval-steps",   type=int,   default=500,   dest="eval_steps")
    p.add_argument("--amp",      action=argparse.BooleanOptionalAction, default=True,
                   dest="use_amp",
                   help="Use mixed-precision (fp16) training. Pass --no-amp to disable.")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--val-ratio",    type=float, default=0.1,   dest="val_ratio")
    p.add_argument("--test-ratio",   type=float, default=0.1,   dest="test_ratio")
    p.set_defaults(func=_run_finetune_reranker)


def _run_finetune_reranker(args: argparse.Namespace) -> None:
    from src.models_training.finetune_cross_encoder import main
    main(args)


# ──────────────────────────────────────────────────────────────────────────── #
# Retrieval                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

def _sub_build_index(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "build-index",
        help="Embed the corpus and build a FAISS index for dense retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--corpus",     default="data/retrieval_corpus/processed/DHDA_text_to_embed.csv",
                   help="Text corpus CSV (single 'text' column).")
    p.add_argument("--text-col",   default="text",        dest="text_col")
    p.add_argument("--model",      default="nomic-ai/nomic-embed-text-v2-moe",
                   help="SentenceTransformer model (local path or HuggingFace ID).")
    # Note: build_index.py inserts the model tag into the output file stems, so
    # the defaults below become embeddings_nomic-embed.npy / faiss_nomic-embed.index
    # — matching the defaults used by eval-retrieval and generate.
    p.add_argument("--emb-out",    default="data/retrieval_corpus/vector_database/embeddings.npy",
                   dest="emb_out",
                   help="Base output path for embeddings (.npy). The model tag is inserted "
                        "into the stem automatically (e.g. embeddings_nomic-embed.npy).")
    p.add_argument("--idx-out",    default="data/retrieval_corpus/vector_database/faiss.index",
                   dest="idx_out",
                   help="Base output path for FAISS index. The model tag is inserted "
                        "into the stem automatically (e.g. faiss_nomic-embed.index).")
    p.add_argument("--index-type", default="Flat", choices=["Flat", "IVFFlat", "HNSW"],
                   dest="index_type")
    p.add_argument("--batch-size", type=int, default=256, dest="batch_size")
    p.add_argument("--device",     default=None)
    p.add_argument("--force",      action="store_true",
                   help="Overwrite existing outputs.")
    p.set_defaults(func=_run_build_index)


def _run_build_index(args: argparse.Namespace) -> None:
    from src.retrieval.build_index import build
    build(args)


# ─────────────────────────────────────────────────────────────────────────── #

def _sub_eval_retrieval(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "eval-retrieval",
        help="Evaluate retrieval quality (Recall@K, MRR, MAP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--method",          default="hybrid", choices=["bm25", "dense", "hybrid"])
    p.add_argument("--eval-data",       default="data/questions_data/QD_data/QD_evaluation_data.csv",
                   dest="eval_data")
    p.add_argument("--corpus",          default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv")
    p.add_argument("--text-data",       default="data/retrieval_corpus/processed/DHDA_text_to_embed.csv",
                   dest="text_data")
    p.add_argument("--embeddings",      default="data/retrieval_corpus/vector_database/embeddings_nomic-embed.npy")
    p.add_argument("--index",           default="data/retrieval_corpus/vector_database/faiss_nomic-embed.index")
    p.add_argument("--embedding-model", default="models/nomic", dest="embedding_model")
    p.add_argument("--classifier",      default="models/RF_intent_classifier.joblib")
    p.add_argument("--cross-encoder",   default="models/finetuned_CE_bge", dest="cross_encoder",
                   help="Pass 'none' to disable reranking.")
    p.add_argument("--output",          default=None,
                   help="Output CSV. Defaults to retrieval_output/<method>_results.csv.")
    p.add_argument("--top-k",    type=int, default=10,  dest="top_k")
    p.add_argument("--k-bm25",   type=int, default=50,  dest="k_bm25")
    p.add_argument("--k-rrf",    type=int, default=300, dest="k_rrf")
    p.add_argument("--k-rerank", type=int, default=50,  dest="k_rerank")
    p.set_defaults(func=_run_eval_retrieval)


def _run_eval_retrieval(args: argparse.Namespace) -> None:
    if args.output is None:
        args.output = f"retrieval_output/{args.method}_results.csv"
    from src.retrieval.evaluate_retrieval import main
    main(args)


# ──────────────────────────────────────────────────────────────────────────── #
# RAG generation                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def _sub_generate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "generate",
        help="Run the full RAG generation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    required=True, choices=["fanar", "gemini", "hf"],
                   help="Generation model backend.")
    p.add_argument("--mode",     required=True, choices=["fs", "zs", "baseline"],
                   help="Prompting mode: few-shot, zero-shot, or baseline (no retrieval).")
    p.add_argument("--method",   default="hybrid", choices=["bm25", "dense", "hybrid"],
                   help="Retrieval method (ignored for baseline mode).")
    p.add_argument("--input",    default="data/questions_data/QA_data/evaluation_data/islamic_full_testing.csv",
                   help="CSV with 'question' and 'answer' columns.")
    p.add_argument("--output",   default=None,
                   help="Output CSV path. Defaults to output/<model>_<mode>_<method>_results.csv.")
    p.add_argument("--start-from", type=int, default=0, dest="start_from",
                   help="Row index to resume from after interruption.")
    p.add_argument("--model-id", default=None, dest="model_id",
                   help="Override default model ID / local path.")
    p.add_argument("--load-in-4bit", action="store_true", dest="load_in_4bit",
                   help="Enable 4-bit quantization for the hf backend.")
    p.add_argument("--corpus",          default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv")
    p.add_argument("--text-data",       default="data/retrieval_corpus/processed/DHDA_text_to_embed.csv",
                   dest="text_data")
    p.add_argument("--embeddings",      default="data/retrieval_corpus/vector_database/embeddings_nomic-embed.npy")
    p.add_argument("--index",           default="data/retrieval_corpus/vector_database/faiss_nomic-embed.index")
    p.add_argument("--embedding-model", default="models/nomic", dest="embedding_model")
    p.add_argument("--classifier",      default="models/RF_intent_classifier.joblib")
    p.add_argument("--cross-encoder",   default="models/finetuned_CE_bge", dest="cross_encoder",
                   help="Pass 'none' to disable reranking.")
    p.add_argument("--fanar-api-key",   default=None, dest="fanar_api_key",
                   help="Fanar API key (or set FANAR_API_KEY env var).")
    p.add_argument("--gemini-api-key",  default=None, dest="gemini_api_key",
                   help="Gemini API key (or set GEMINI_API_KEY env var).")
    p.set_defaults(func=_run_generate)


def _run_generate(args: argparse.Namespace) -> None:
    _set_env("FANAR_API_KEY",  args.fanar_api_key)
    _set_env("GEMINI_API_KEY", args.gemini_api_key)

    if args.output is None:
        tag = "baseline" if args.mode == "baseline" else f"{args.mode}_{args.method}"
        args.output = f"output/{args.model}_{tag}_results.csv"

    from src.retrieval.retriever import HybridRetriever
    from src.rag.model_loader import ModelLoader
    from src.rag.rag_pipeline import RAGPipeline

    backend_kwargs: dict = {}
    if args.model == "hf" and args.load_in_4bit:
        backend_kwargs["load_in_4bit"] = True
    backend = ModelLoader.load(args.model, model_id=args.model_id, **backend_kwargs)

    retriever = None
    if args.mode != "baseline":
        cross_encoder = None if args.cross_encoder.lower() == "none" else args.cross_encoder
        retriever = HybridRetriever(
            corpus_path=args.corpus,
            text_data_path=args.text_data,
            embeddings_path=args.embeddings,
            index_path=args.index,
            classifier_path=args.classifier,
            cross_encoder_path=cross_encoder,
            embedding_model_path=args.embedding_model,
        )

    pipeline = RAGPipeline(
        backend=backend,
        mode=args.mode,
        retriever=retriever,
        retrieval_method=args.method,
    )
    pipeline.run(
        input_file=args.input,
        output_file=args.output,
        start_from=args.start_from,
    )


# ──────────────────────────────────────────────────────────────────────────── #
# Evaluation                                                                     #
# ──────────────────────────────────────────────────────────────────────────── #

def _sub_judge(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "judge",
        help="Score model answers with Gemini-as-judge. Requires GEMINI_API_KEY.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",      required=True,
                   help="CSV with columns: question, answer, correct_answer.")
    p.add_argument("--output",     default="judging/judging_results.csv",
                   help="Output CSV for judging results.")
    p.add_argument("--api-delay",  type=float, default=1.0, dest="api_delay",
                   help="Seconds to sleep between Gemini API calls.")
    p.add_argument("--gemini-api-key", default=None, dest="gemini_api_key",
                   help="Gemini API key (or set GEMINI_API_KEY env var).")
    p.set_defaults(func=_run_judge)


def _run_judge(args: argparse.Namespace) -> None:
    _set_env("GEMINI_API_KEY", args.gemini_api_key)
    from src.evaluation.judge import evaluate_dataset
    evaluate_dataset(
        input_file=args.input,
        output_file=args.output,
        api_delay=args.api_delay,
    )


# ─────────────────────────────────────────────────────────────────────────── #

def _sub_summarize_scores(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "summarize-scores",
        help="Print score statistics for one judging CSV, or compare two.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  required=True,
                   help="Judging output CSV.")
    p.add_argument("--input2", default=None,
                   help="Second judging CSV for comparison.")
    p.set_defaults(func=_run_summarize_scores)


def _run_summarize_scores(args: argparse.Namespace) -> None:
    from src.evaluation.summarize_scores import main
    main(["--input", args.input] + (["--input2", args.input2] if args.input2 else []))


# ──────────────────────────────────────────────────────────────────────────── #
# Entry point                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="pipeline", metavar="<pipeline>")
    sub.required = True

    _sub_build_corpus(sub)
    _sub_build_qa(sub)
    _sub_build_qd(sub)
    _sub_build_ce_data(sub)
    _sub_build_clf_data(sub)
    _sub_train_classifier(sub)
    _sub_finetune_reranker(sub)
    _sub_build_index(sub)
    _sub_eval_retrieval(sub)
    _sub_generate(sub)
    _sub_judge(sub)
    _sub_summarize_scores(sub)

    return parser


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
