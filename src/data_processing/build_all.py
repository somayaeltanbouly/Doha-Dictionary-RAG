"""
build_all.py — DHDA Data Processing Pipeline Orchestrator
==========================================================

Runs the full end-to-end data preparation pipeline for the Doha Historical
Dictionary of Arabic (DHDA) RAG system. Each processing step can be run
individually or selectively via ``--steps``.

Execution order and dependencies
---------------------------------
The five steps must run in order because later steps consume outputs of
earlier ones:

    Step 1  build_retrieval_corpus
            Combines raw scraped CSVs → processed corpus files
            (must run first; all other steps depend on its outputs)

    Step 2  build_qa_pairs
            Generates structured QA pairs for model evaluation
            (depends on: DHDA_filtered_AR.csv)

    Step 3  build_qd_pairs
            Generates query-document pairs for bi-encoder training
            (depends on: DHDA_filtered_AR.csv)

    Step 4  build_ce_training_data
            Generates contrastive pairs for cross-encoder fine-tuning
            (depends on: QD_full_data.csv, output of step 3)

    Step 5  build_classification_data
            Generates labelled questions for question-type classification
            (depends on: DHDA_filtered_AR.csv, DHDA_inscriptions_data.csv)

Usage
-----
Run all steps (default)::

    python src/data_processing/build_all.py

Run a specific subset of steps::

    python src/data_processing/build_all.py --steps retrieval_corpus qd_pairs ce_training

Available step names:

    retrieval_corpus      build_retrieval_corpus
    qa_pairs              build_qa_pairs
    qd_pairs              build_qd_pairs
    ce_training           build_ce_training_data
    classification        build_classification_data

All path defaults match the repository layout under ``data/``. Override
any path or tuning argument via the corresponding flag documented below.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add the package root to sys.path so that sibling imports work whether the
# script is invoked from the repo root or from within src/data_processing/.
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_processing.build_retrieval_corpus import RetrievalDataGenerator
from data_processing.build_qa_pairs import QADataGenerator
from data_processing.build_qd_pairs import QDDataGenerator
from data_processing.build_ce_training_data import CEContrastiveDataGenerator
from data_processing.build_classification_data import ClassificationDataGenerator

# ── Step registry ─────────────────────────────────────────────────── #

#: Canonical names accepted by --steps, in execution order.
ALL_STEPS: list[str] = [
    "retrieval_corpus",
    "qa_pairs",
    "qd_pairs",
    "ce_training",
    "classification",
]


# ── Individual step runners ───────────────────────────────────────── #

def run_retrieval_corpus(args: argparse.Namespace) -> None:
    """Step 1 — Combine raw CSVs and build processed corpus files."""
    gen = RetrievalDataGenerator(
        raw_dir=args.raw_dir,
        output_dir=args.corpus_output_dir,
    )
    gen.run()


def run_qa_pairs(args: argparse.Namespace) -> None:
    """Step 2 — Generate structured QA pairs for model evaluation."""
    gen = QADataGenerator(
        corpus_csv=args.corpus,
        output_dir=args.qa_output_dir,
    )
    gen.run()


def run_qd_pairs(args: argparse.Namespace) -> None:
    """Step 3 — Generate query-document pairs for bi-encoder training."""
    gen = QDDataGenerator(
        corpus_csv=args.corpus,
        output_path=args.qd_output,
    )
    qd = gen.run()
    # Override evaluation-sample path or size only if the user explicitly
    # passed non-default values.
    if args.qd_eval_output != QDDataGenerator._EVAL_PATH or args.qd_n_sample != 1002:
        gen.sample_evaluation(
            qd,
            n_sample=args.qd_n_sample,
            eval_path=args.qd_eval_output,
        )


def run_ce_training(args: argparse.Namespace) -> None:
    """Step 4 — Generate contrastive pairs for cross-encoder fine-tuning."""
    gen = CEContrastiveDataGenerator(
        qd_path=args.qd_output,
        output_path=args.ce_output,
        n_source=args.ce_n_source,
        num_negatives=args.ce_num_negatives,
        random_state=args.random_state,
    )
    gen.run()


def run_classification(args: argparse.Namespace) -> None:
    """Step 5 — Generate labelled questions for question-type classification."""
    gen = ClassificationDataGenerator(
        corpus_csv=args.corpus,
        inscriptions_csv=args.inscriptions,
        output_path=args.clf_output,
        n_per_type=args.clf_n_per_type,
        random_state=args.random_state,
    )
    gen.run()


# ── Dispatch map ──────────────────────────────────────────────────── #

_STEP_FN = {
    "retrieval_corpus": run_retrieval_corpus,
    "qa_pairs":         run_qa_pairs,
    "qd_pairs":         run_qd_pairs,
    "ce_training":      run_ce_training,
    "classification":   run_classification,
}


# ── CLI ───────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end DHDA data processing pipeline. "
            "Runs all five steps by default; use --steps to run a subset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Step selection ──────────────────────────────────────────── #
    p.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        metavar="STEP",
        help=(
            "Space-separated list of steps to execute. "
            f"Choices: {', '.join(ALL_STEPS)}. "
            "Steps always run in dependency order regardless of the order "
            "they are listed here."
        ),
    )

    # ── Shared paths ────────────────────────────────────────────── #
    shared = p.add_argument_group("shared paths")
    shared.add_argument(
        "--corpus",
        default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv",
        help=(
            "Path to DHDA_filtered_AR.csv. "
            "Consumed by steps: qa_pairs, qd_pairs, classification."
        ),
    )
    shared.add_argument(
        "--inscriptions",
        default="data/retrieval_corpus/processed/DHDA_inscriptions_data.csv",
        help="Path to DHDA_inscriptions_data.csv. Consumed by: classification.",
    )
    shared.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Global random seed forwarded to all steps that accept one.",
    )

    # ── Step 1: build_retrieval_corpus ──────────────────────────── #
    grp1 = p.add_argument_group("step 1 — retrieval_corpus")
    grp1.add_argument(
        "--raw-dir",
        default="data/retrieval_corpus/raw",
        help=(
            "Root folder containing lexical_details/, etymological/, "
            "and inscriptions/ sub-folders with raw scraped CSVs."
        ),
    )
    grp1.add_argument(
        "--corpus-output-dir",
        default="data/retrieval_corpus/processed",
        help="Destination folder for all processed corpus files.",
    )

    # ── Step 2: build_qa_pairs ──────────────────────────────────── #
    grp2 = p.add_argument_group("step 2 — qa_pairs")
    grp2.add_argument(
        "--qa-output-dir",
        default="data/questions_data/QA_data",
        help=(
            "Root output folder for QA files "
            "(QA_roots/, QA_combined/, evaluation_data/ are created inside)."
        ),
    )

    # ── Step 3: build_qd_pairs ──────────────────────────────────── #
    grp3 = p.add_argument_group("step 3 — qd_pairs")
    grp3.add_argument(
        "--qd-output",
        default="data/questions_data/QD_data/QD_full_data.csv",
        help="Output path for QD_full_data.csv.",
    )
    grp3.add_argument(
        "--qd-eval-output",
        default=QDDataGenerator._EVAL_PATH,
        help="Output path for QD_evaluation_data.csv.",
    )
    grp3.add_argument(
        "--qd-n-sample",
        type=int,
        default=1002,
        help="Number of rows in the QD evaluation sample.",
    )

    # ── Step 4: build_ce_training_data ──────────────────────────── #
    grp4 = p.add_argument_group("step 4 — ce_training")
    grp4.add_argument(
        "--ce-output",
        default="data/questions_data/QD_data/QD_contrastive_data.csv",
        help="Output path for QD_contrastive_data.csv.",
    )
    grp4.add_argument(
        "--ce-n-source",
        type=int,
        default=5000,
        help=(
            "Number of QD rows to sample before building contrastive pairs. "
            "Approximately twice this many rows will appear in the output."
        ),
    )
    grp4.add_argument(
        "--ce-num-negatives",
        type=int,
        default=1,
        help="Number of negative document samples per query-document pair.",
    )

    # ── Step 5: build_classification_data ───────────────────────── #
    grp5 = p.add_argument_group("step 5 — classification")
    grp5.add_argument(
        "--clf-output",
        default="data/questions_data/classification_data/classification_data.csv",
        help="Output path for classification_data.csv.",
    )
    grp5.add_argument(
        "--clf-n-per-type",
        type=int,
        default=1000,
        help="Number of labelled questions to sample per question type.",
    )

    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────── #

def main(argv=None) -> None:
    args = parse_args(argv)

    # Enforce dependency order: sort requested steps by their position in
    # ALL_STEPS so that upstream outputs are always ready before downstream
    # steps start, regardless of the order the user listed them.
    ordered = [s for s in ALL_STEPS if s in args.steps]

    print("=" * 60)
    print("DHDA Data Processing Pipeline")
    print(f"Steps to run: {', '.join(ordered)}")
    print("=" * 60)

    pipeline_start = time.time()

    for step in ordered:
        print(f"\n{'─' * 60}")
        print(f"[pipeline] Starting: {step}")
        print(f"{'─' * 60}")
        step_start = time.time()

        try:
            _STEP_FN[step](args)
        except Exception as exc:
            print(f"\n[pipeline] ERROR in step '{step}': {exc}", file=sys.stderr)
            raise

        elapsed = time.time() - step_start
        print(f"[pipeline] Completed: {step}  ({elapsed:.1f}s)")

    total = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"[pipeline] All {len(ordered)} step(s) completed in {total:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
