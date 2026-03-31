"""
QDDataGenerator
===============

Generates query-document pairs for retrieval training from the processed
lexical corpus (``DHDA_filtered_AR.csv``).

Output columns
--------------
``type, question, answer, doc_id, query_id, text``

- ``doc_id``   — the corpus row ``ID`` (unique per entry/row)
- ``query_id`` — sequential integer per unique question text (1-based)
- ``text``     — tashkeel-stripped concatenation of
                 الكلمة + الجذر + المعنى + الشاهد

n-to-n relationship
--------------------
For ``basic_meaning`` and ``part_of_speech``, multiple corpus rows that share
the same ``meaning_head`` (or ``الكلمة``) produce the same question → same
``query_id`` but different ``doc_id`` and ``answer``.

All other question types include the unique citation text in the question,
so each maps 1-to-1 to a single document.

Usage::

    python src/data_processing/build_qd_pairs.py

    # or with explicit paths:
    python src/data_processing/build_qd_pairs.py \\
        --corpus  data/retrieval_corpus/processed/DHDA_filtered_AR.csv \\
        --output  data/questions_data/QD_data/QD_full_data.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running directly: python src/data_processing/build_qd_pairs.py
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from data_processing.data_utils import (
    nonempty,
    normalize_author,
    parse_meaning,
    strip_tashkeel,
    word_type,
)

# Aliases so all existing call-sites work without change.
_nonempty         = nonempty
_word_type        = word_type
_parse_meaning    = parse_meaning
_normalize_author = normalize_author

_TEXT_COLS = ["الكلمة", "الجذر", "المعنى", "الشاهد"]


def _build_text(row: pd.Series) -> str:
    """Tashkeel-stripped concatenation of the four text columns."""
    parts = []
    for col in _TEXT_COLS:
        val = row.get(col)
        if val is not None and not pd.isna(val):
            s = str(val).strip().replace("***", " ")
            if s:
                parts.append(s)
    return strip_tashkeel(" ".join(parts))


# ── Per-row QD generation ─────────────────────────────────────────── #

def _generate_qd_for_row(row: pd.Series) -> list[dict]:
    """
    Return QD pair dicts for one corpus row.

    Each dict contains: ``type, question, answer, doc_id, text``.
    ``query_id`` is assigned later based on unique question text.
    """
    pairs: list[dict] = []

    doc_id    = row["ID"]
    lemma_val = str(row["الكلمة"]).strip()
    meaning   = row["المعنى"]
    citation  = row["الشاهد"]
    head_cit  = row["مقدمة الشاهد"]
    author    = row["القائل"]
    source    = row["المصدر"]
    date      = row["تاريخ استعمال الشاهد"]
    pos       = row["الاشتقاق الصرفي للكلمة"]
    text      = _build_text(row)

    if not _nonempty(meaning):
        return pairs

    meaning_str             = str(meaning).strip()
    meaning_head, meaning_2 = _parse_meaning(meaning_str)
    wt                      = _word_type(meaning_head)

    head_cit_str = str(head_cit).strip() if _nonempty(head_cit) else "السياق"

    if _nonempty(author):
        author = _normalize_author(str(author).strip())

    base = {"doc_id": doc_id, "text": text}

    # ── basic_meaning ────────────────────────────────────────────── #
    # Same meaning_head across multiple rows → same question, multiple doc_ids
    if meaning_head:
        pairs.append({
            **base,
            "type":     "basic_meaning",
            "question": f'ما معنى {wt} "{meaning_head}"؟',
            "answer":   meaning_str,
        })

    # ── contextual_meaning ───────────────────────────────────────── #
    # Citation text is unique per row → 1-to-1 mapping
    if _nonempty(citation):
        if head_cit_str != "السياق":
            pairs.append({
                **base,
                "type":     "contextual_meaning",
                "question": f' ما هو معنى {wt} "{lemma_val}"،؟ في الشاهد التالي: "{citation}"',
                "answer":   f' {wt} "{lemma_val}" تعني: {meaning_2}، وقد وردت في هذا الشاهد حيث "{head_cit_str}"',
            })
        else:
            pairs.append({
                **base,
                "type":     "contextual_meaning",
                "question": f'في الشاهد التالي: "{citation}"، ما هو معنى {wt} "{lemma_val}"؟',
                "answer":   f' {wt} "{lemma_val}" تعني: {meaning_2}',
            })

    # ── part_of_speech ───────────────────────────────────────────── #
    # Same lemma_val across multiple rows → same question, multiple doc_ids
    if _nonempty(pos):
        pairs.append({
            **base,
            "type":     "part_of_speech",
            "question": f'ما الاشتقاق الصرفي لكلمة "{lemma_val}"؟',
            "answer":   f'كلمة "{lemma_val}" هي {str(pos).strip()}.',
        })

    # ── author_of_citation ───────────────────────────────────────── #
    if _nonempty(author) and _nonempty(citation):
        pairs.append({
            **base,
            "type":     "author_of_citation",
            "question": f'من القائل الذي استخدم {wt} "{lemma_val}" في الشاهد: "{citation}"؟',
            "answer":   f"القائل: {author}.",
        })

    # ── historical_date ──────────────────────────────────────────── #
    if _nonempty(date) and _nonempty(citation) and meaning_head:
        pairs.append({
            **base,
            "type":     "historical_date",
            "question": f'ما هو تاريخ الشاهد الذي استعمل فيه {wt} "{meaning_head}" بمعنى "{meaning_2}"',
            "answer":   f'تم توثيق هذا الاستخدام {wt} "{lemma_val}" حوالي عام {str(date).strip()}.',
        })

    # ── source_of_citation ───────────────────────────────────────── #
    if _nonempty(source) and _nonempty(citation) and meaning_head:
        pairs.append({
            **base,
            "type":     "source_of_citation",
            "question": f'ما هو المصدر الذي ورد فيه شاهد استخدام {wt} "{lemma_val}"؟ بمعنى "{meaning_2}"',
            "answer":   f'ورد هذا الشاهد في المصدر التالي: {str(source).strip()}',
        })

    return pairs


# ── Main class ────────────────────────────────────────────────────── #

class QDDataGenerator:
    """Generate query-document pairs from the DHDA lexical corpus."""

    _EVAL_PATH = "data/questions_data/QD_data/QD_evaluation_data.csv"

    def __init__(
        self,
        corpus_csv: str | Path,
        output_path: str | Path = "data/questions_data/QD_data/QD_full_data.csv",
    ) -> None:
        self.corpus_csv  = Path(corpus_csv)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """Generate all QD pairs and save to ``output_path``."""
        df = pd.read_csv(self.corpus_csv, encoding="utf-8-sig", low_memory=False)
        print(f"[load]     {len(df):,} rows from {self.corpus_csv}")

        # Generate raw (type, question, answer, doc_id, text) records
        records: list[dict] = []
        for _, row in df.iterrows():
            records.extend(_generate_qd_for_row(row))

        print(f"[generate] {len(records):,} QD pairs generated")

        # Assign query_ids: one sequential integer per unique question text
        question_to_qid: dict[str, int] = {}
        next_qid = 1
        for rec in records:
            q = rec["question"]
            if q not in question_to_qid:
                question_to_qid[q] = next_qid
                next_qid += 1
            rec["query_id"] = question_to_qid[q]

        result = pd.DataFrame(
            records,
            columns=["type", "question", "answer", "doc_id", "query_id", "text"],
        )
        result.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        print(f"[save]     {len(result):,} rows → {self.output_path}")
        print(f"           unique query_ids  : {result['query_id'].nunique():,}")
        print(f"           type distribution :")
        for t, n in result["type"].value_counts().items():
            print(f"             {t}: {n:,}")

        self.sample_evaluation(result)
        return result

    def sample_evaluation(
        self,
        qd_data: pd.DataFrame,
        n_sample: int = 1002,
        eval_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """Proportional stratified sample of *n_sample* rows across all types.

        Each type contributes ``frac = n_sample / len(qd_data)`` of its rows,
        preserving the natural type distribution of the full QD dataset.
        """
        out = Path(eval_path) if eval_path is not None else Path(self._EVAL_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)

        frac = n_sample / len(qd_data)
        sampled = (
            qd_data
            .groupby("type", group_keys=False)
            .apply(lambda x: x.sample(frac=frac, random_state=42))
            .reset_index(drop=True)
        )

        sampled.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[eval]     {len(sampled):,} rows → {out}")
        print(f"           type distribution :")
        for t, n in sampled["type"].value_counts().items():
            print(f"             {t}: {n:,}")
        return sampled


# ── CLI ───────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate query-document pairs from the DHDA filtered corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--corpus",
        default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv",
        help="Path to DHDA_filtered_AR.csv (the processed Arabic-column corpus).",
    )
    p.add_argument(
        "--output",
        default="data/questions_data/QD_data/QD_full_data.csv",
        help="Output path for QD_full_data.csv.",
    )
    p.add_argument(
        "--eval-output",
        default=QDDataGenerator._EVAL_PATH,
        help="Output path for QD_evaluation_data.csv.",
    )
    p.add_argument(
        "--n-sample",
        type=int,
        default=1002,
        help="Number of rows in the evaluation sample.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    gen = QDDataGenerator(corpus_csv=args.corpus, output_path=args.output)
    qd = gen.run()
    # run() already calls sample_evaluation with defaults; override if CLI args differ
    if args.eval_output != QDDataGenerator._EVAL_PATH or args.n_sample != 1002:
        gen.sample_evaluation(qd, n_sample=args.n_sample, eval_path=args.eval_output)
