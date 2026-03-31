"""
rag_pipeline.py — End-to-end RAG pipeline for the Doha Dictionary QA system.

``RAGPipeline`` wires together the retriever, prompt builder, and generation
backend into a single configurable object.

Supported modes
---------------
- ``"fs"``       : few-shot RAG — retrieval + intent-aware prompting with examples.
- ``"zs"``       : zero-shot RAG — retrieval + intent-aware prompting, no examples.
- ``"baseline"`` : no retrieval — query sent directly to the model.

Usage::

    from src.retrieval.retriever  import HybridRetriever
    from src.rag.model_loader     import ModelLoader
    from src.rag.rag_pipeline     import RAGPipeline

    retriever = HybridRetriever()
    backend   = ModelLoader.load("gemini")
    pipeline  = RAGPipeline(backend, mode="fs", retriever=retriever)

    # Single query
    result = pipeline.process_query("ما معنى كلمة الجفاء؟")

    # Batch CSV
    pipeline.run(
        input_file="data/questions_data/QA_data/evaluation_data/islamic_full_testing.csv",
        output_file="output/gemini_fs_results.csv",
    )
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pandas as pd

# Allow running directly: python src/rag/rag_pipeline.py
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.rag.prompt_builder import PromptBuilder

if TYPE_CHECKING:
    from src.rag.model_loader import GenerationBackend
    from src.retrieval.retriever import HybridRetriever


def _format_eta(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


# ──────────────────────────────────────────────────────────────────────────── #
# Pipeline                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #

class RAGPipeline:
    """End-to-end RAG pipeline for the Doha Dictionary QA system.

    Args:
        backend:          Generation backend produced by :class:`~model_loader.ModelLoader`.
        mode:             ``"fs"``, ``"zs"``, or ``"baseline"``.
        retriever:        :class:`~retriever.HybridRetriever` instance.
                          Required for ``"fs"`` and ``"zs"`` modes.
        retrieval_method: ``"bm25"``, ``"dense"``, or ``"hybrid"``.
    """

    VALID_MODES = ("fs", "zs", "baseline")

    def __init__(
        self,
        backend: "GenerationBackend",
        mode: str = "fs",
        retriever: Optional["HybridRetriever"] = None,
        retrieval_method: str = "hybrid",
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode {mode!r}. Choose from {self.VALID_MODES}."
            )
        if mode != "baseline" and retriever is None:
            raise ValueError(
                "A HybridRetriever is required for 'fs' and 'zs' modes."
            )

        self.backend = backend
        self.mode = mode
        self.retriever = retriever
        self.retrieval_method = retrieval_method

    # ------------------------------------------------------------------ #
    # Single-query processing                                              #
    # ------------------------------------------------------------------ #

    def process_query(self, query: str) -> dict[str, str]:
        """Run the full pipeline on a single query.

        Returns:
            Dict with keys ``"answer"`` and ``"intent"``.
        """
        if self.mode == "baseline":
            return {
                "answer": self.backend.generate(query),
                "intent": "none",
            }

        assert self.retriever is not None
        query_info = self.retriever.analyze_query(query)
        intent = query_info["intent"]

        docs_df, _ = self.retriever.retrieve(
            query_info["q1"], method=self.retrieval_method
        )
        docs_str = self.retriever.format_documents(docs_df, intent)
        prompt = PromptBuilder.build(intent, self.mode, query, docs_str)
        answer = self.backend.generate(prompt)

        return {"answer": answer, "intent": intent}

    # ------------------------------------------------------------------ #
    # Batch processing                                                     #
    # ------------------------------------------------------------------ #

    def run(
        self,
        input_file: str,
        output_file: str,
        start_from: int = 0,
    ) -> None:
        """Process all questions in *input_file* and write results to *output_file*.

        The output CSV is written row-by-row so progress is preserved on interruption.

        Args:
            input_file:  Path to a CSV with columns ``question`` and ``answer``
                         (``answer`` holds the gold reference).
            output_file: Destination CSV path. Parent directory is created automatically.
            start_from:  0-based row index to resume from (useful for restarts).
        """
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        questions = pd.read_csv(input_file)
        if start_from > 0:
            questions = questions.iloc[start_from:]

        total = len(questions)
        print(
            f"Processing {total} questions "
            f"| model={self.backend} "
            f"| mode={self.mode!r} "
            f"| retrieval={self.retrieval_method!r}"
        )

        fieldnames = ["question", "answer", "correct_answer"]

        with open(output_file, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()

            start_time = time.time()
            for pos, (_, row) in enumerate(questions.iterrows(), start=1):
                q_start = time.time()
                result = self.process_query(row["question"])

                writer.writerow({
                    "question":       row["question"],
                    "answer":         result["answer"],
                    "correct_answer": row["answer"],
                })
                fh.flush()  # preserve progress on interruption

                elapsed = time.time() - start_time
                avg_time = elapsed / pos
                eta = avg_time * (total - pos)
                q_time = time.time() - q_start
                print(
                    f"[{pos}/{total}] intent={result['intent']!r} "
                    f"| {q_time:.1f}s "
                    f"| ETA: {_format_eta(eta)}"
                )

        total_time = time.time() - start_time
        print(f"\nDone — {total} questions in {_format_eta(total_time)}.")
        print(f"Results saved to: {output_file}")


# ──────────────────────────────────────────────────────────────────────────── #
# CLI                                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Doha Dictionary RAG pipeline on a CSV of questions."
    )
    parser.add_argument(
        "--model", required=True,
        choices=["fanar", "gemini", "hf"],
        help="Generation model backend.",
    )
    parser.add_argument(
        "--model-id", default=None,
        help="Override the backend's default model ID / local path.",
    )
    parser.add_argument(
        "--mode", default="fs",
        choices=["fs", "zs", "baseline"],
        help="Prompting mode (default: fs).",
    )
    parser.add_argument(
        "--method", default="hybrid",
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval method (default: hybrid).",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input CSV with 'question' and 'answer' columns.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the output CSV.",
    )
    parser.add_argument(
        "--start-from", type=int, default=0,
        help="0-based row index to resume from (default: 0).",
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Enable 4-bit quantization for the HuggingFace backend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.rag.model_loader import ModelLoader
    from src.retrieval.retriever import HybridRetriever

    # Build optional kwargs for the backend
    backend_kwargs: dict = {}
    if args.model == "hf" and args.load_in_4bit:
        backend_kwargs["load_in_4bit"] = True

    backend = ModelLoader.load(args.model, model_id=args.model_id, **backend_kwargs)

    retriever = None
    if args.mode != "baseline":
        retriever = HybridRetriever()

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


if __name__ == "__main__":
    main()
