"""
QADataGenerator
===============

Generates structured question-answer pairs from the processed lexical corpus
(``DHDA_filtered_AR.csv``) and organises them into the files needed for
model evaluation and fine-tuning.

Pipeline
--------

1. **build_all_qa** — Generate QA pairs for every lemma, grouped by root.
   - Saves one JSON per root  →  ``QA_roots/all_data/qa_root_{rootId}.json``
   - Combines all roots      →  ``QA_combined/full_QA.json``

2. **build_islamic_qa** — Filter to Islamic entries (rows that have a value in
   either ``السورة`` or ``رقم الحديث``), then repeat step 1 for that subset.
   - Saves one JSON per root  →  ``QA_roots/islamic_data/qa_root_{rootId}.json``
   - Combines all roots       →  ``QA_combined/islamic_QA_comprehensive.json``
   - Filters to meaning types →  ``QA_combined/islamic_QA_meaning.json``

3. **sample_islamic_full** — Stratified sample of 2 000 questions from
   ``islamic_QA_comprehensive.json`` (all 6 question types).
   Output → ``evaluation_data/islamic_full_testing.csv``

4. **sample_islamic_meanings** — 500 ``contextual_meaning`` + 500
   ``basic_meaning`` rows sampled from ``islamic_QA_meaning.json``.
   Output → ``evaluation_data/islamic_meaning_testing.csv``

Question types generated per lemma
-----------------------------------
- ``author_of_citation``  — requires القائل and الشاهد
- ``contextual_meaning``  — requires المعنى and الشاهد
- ``source_of_citation``  — requires المصدر and الشاهد
- ``historical_date``     — requires تاريخ استعمال الشاهد and الشاهد
- ``basic_meaning``       — requires المعنى
- ``part_of_speech``      — requires الاشتقاق الصرفي للكلمة

Output CSV columns (evaluation files)
--------------------------------------
``lemma, type, question, answer``

For ``basic_meaning`` and ``part_of_speech``, duplicate ``(lemma, question)``
pairs are collapsed into one row with answers joined by `` | ``.

Usage::

    python src/data_processing/build_qa_pairs.py

    # or with explicit paths:
    python src/data_processing/build_qa_pairs.py \\
        --corpus  data/retrieval_corpus/processed/DHDA_filtered_AR.csv \\
        --output-dir data/questions_data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow running directly: python src/data_processing/build_qa_pairs.py
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from data_processing.data_utils import nonempty, normalize_author, parse_meaning, word_type

# Alias to private-looking names so all existing call-sites work without change.
_nonempty        = nonempty
_word_type       = word_type
_parse_meaning   = parse_meaning
_normalize_author = normalize_author


# ── QA pair generation ────────────────────────────────────────────── #

def _generate_qa_for_row(row: pd.Series) -> list[dict]:
    """Return all applicable QA pairs for a single corpus row."""
    pairs: list[dict] = []

    lemma_id  = row["lemmaId"]
    lemma_val = str(row["الكلمة"]).strip()
    meaning   = row["المعنى"]
    citation  = row["الشاهد"]
    head_cit  = row["مقدمة الشاهد"]
    author    = row["القائل"]
    source    = row["المصدر"]
    date      = row["تاريخ استعمال الشاهد"]
    pos       = row["الاشتقاق الصرفي للكلمة"]

    # Skip entries with no meaning
    if not _nonempty(meaning):
        return pairs

    meaning_str             = str(meaning).strip()
    meaning_head, meaning_2 = _parse_meaning(meaning_str)
    # One word_type for all templates — based on meaning_head, matching original code
    wt = _word_type(meaning_head)

    # Normalise head_citation: empty → sentinel "السياق"
    head_cit_str = str(head_cit).strip() if _nonempty(head_cit) else "السياق"

    if _nonempty(author):
        author = _normalize_author(str(author).strip())

    base = {"Source": True, "lemma": lemma_id}

    # ── basic_meaning ─────────────────────────────────────────
    # Question uses meaning_head; answer is the full meaning string
    if meaning_head:
        pairs.append({
            **base,
            "type": "basic_meaning",
            "question": f'ما معنى {wt} "{meaning_head}"؟',
            "answer":   meaning_str,
        })

    # ── contextual_meaning ────────────────────────────────────
    # Two variants: when head_citation is present the question order is inverted
    if _nonempty(citation):
        if head_cit_str != "السياق":
            pairs.append({
                **base,
                "type": "contextual_meaning",
                "question": f' ما هو معنى {wt} "{lemma_val}"،؟ في الشاهد التالي: "{citation}"',
                "answer":   f' {wt} "{lemma_val}" تعني: {meaning_2}، وقد وردت في هذا الشاهد حيث "{head_cit_str}"',
            })
        else:
            pairs.append({
                **base,
                "type": "contextual_meaning",
                "question": f'في الشاهد التالي: "{citation}"، ما هو معنى {wt} "{lemma_val}"؟',
                "answer":   f' {wt} "{lemma_val}" تعني: {meaning_2}',
            })

    # ── part_of_speech ────────────────────────────────────────
    # Always uses "كلمة" regardless of word_type (matches original)
    if _nonempty(pos):
        pairs.append({
            **base,
            "type": "part_of_speech",
            "question": f'ما الاشتقاق الصرفي لكلمة "{lemma_val}"؟',
            "answer":   f'كلمة "{lemma_val}" هي {str(pos).strip()}.',
        })

    # ── author_of_citation ────────────────────────────────────
    if _nonempty(author) and _nonempty(citation):
        pairs.append({
            **base,
            "type": "author_of_citation",
            "question": f'من القائل الذي استخدم {wt} "{lemma_val}" في الشاهد: "{citation}"؟',
            "answer":   f"القائل: {author}.",
        })

    # ── historical_date ───────────────────────────────────────
    # Question uses meaning_head and meaning_2; answer uses lemma_val
    if _nonempty(date) and _nonempty(citation) and meaning_head:
        pairs.append({
            **base,
            "type": "historical_date",
            "question": f'ما هو تاريخ الشاهد الذي استعمل فيه {wt} "{meaning_head}" بمعنى "{meaning_2}"',
            "answer":   f'تم توثيق هذا الاستخدام {wt} "{lemma_val}" حوالي عام {str(date).strip()}.',
        })

    # ── source_of_citation ────────────────────────────────────
    # Question uses lemma_val and meaning_2
    if _nonempty(source) and _nonempty(citation) and meaning_head:
        pairs.append({
            **base,
            "type": "source_of_citation",
            "question": f'ما هو المصدر الذي ورد فيه شاهد استخدام {wt} "{lemma_val}"؟ بمعنى "{meaning_2}"',
            "answer":   f'ورد هذا الشاهد في المصدر التالي: {str(source).strip()}',
        })

    return pairs


# ── Stratified sampling ───────────────────────────────────────────── #

def _stratified_sample(df: pd.DataFrame, total: int, random_state: int = 42) -> pd.DataFrame:
    """
    Sample *total* rows from *df* with equal representation per ``type``.

    If a type has fewer rows than the per-type quota, all its rows are kept
    and the shortfall is redistributed to types with surplus rows.
    """
    types = df["type"].unique().tolist()
    remaining = total
    available = {t: df[df["type"] == t].copy() for t in types}
    sampled: list[pd.DataFrame] = []
    pending = list(types)

    while pending and remaining > 0:
        quota = remaining // len(pending)
        next_pending: list[str] = []
        for t in pending:
            group = available[t]
            take = min(len(group), quota)
            if take > 0:
                sampled.append(group.sample(n=take, random_state=random_state))
                remaining -= take
            if len(group) > quota:
                available[t] = group  # still has surplus
                next_pending.append(t)
        # Stop if no progress
        if len(next_pending) == len(pending):
            break
        pending = next_pending

    result = pd.concat(sampled, ignore_index=True)
    return result.sample(frac=1, random_state=random_state).reset_index(drop=True)


# ── Multi-answer deduplication ───────────────────────────────────── #

_MULTI_ANS_TYPES = frozenset({"basic_meaning", "part_of_speech"})


def _dedup_qa_pairs(pairs: list[dict]) -> pd.DataFrame:
    """
    Convert QA pairs to a DataFrame, collapsing duplicate questions.

    For ``basic_meaning`` and ``part_of_speech``, rows sharing the same
    ``(lemma, type, question)`` are merged into one row whose ``answer``
    contains unique answers joined by `` | `` (insertion-order preserved).
    """
    df = pd.DataFrame(pairs)
    multi  = df[df["type"].isin(_MULTI_ANS_TYPES)]
    single = df[~df["type"].isin(_MULTI_ANS_TYPES)]

    merged = (
        multi
        .groupby(["lemma", "type", "question"], sort=False, as_index=False)
        .agg({"answer": lambda x: " | ".join(dict.fromkeys(x))})
    )

    return pd.concat([single, merged], ignore_index=True)


# ── Main class ────────────────────────────────────────────────────── #

class QADataGenerator:
    """Generate, filter and sample QA pairs from the DHDA lexical corpus."""

    _MEANING_TYPES = ("contextual_meaning", "basic_meaning")

    def __init__(self, corpus_csv: str | Path, output_dir: str | Path) -> None:
        self.corpus_csv  = Path(corpus_csv)
        self.output_dir  = Path(output_dir)

        self.qa_roots_all     = self.output_dir / "QA_roots" / "all_data"
        self.qa_roots_islamic = self.output_dir / "QA_roots" / "islamic_data"
        self.combined_dir     = self.output_dir / "QA_combined"
        self.evaluation_dir   = self.output_dir / "evaluation_data"

        for d in (self.qa_roots_all, self.qa_roots_islamic, self.combined_dir, self.evaluation_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────── #

    def _load_corpus(self) -> pd.DataFrame:
        df = pd.read_csv(self.corpus_csv, encoding="utf-8-sig", low_memory=False)
        print(f"[load] {len(df):,} rows from {self.corpus_csv}")
        return df

    @staticmethod
    def _is_islamic(row: pd.Series) -> bool:
        """True if the entry has a Quran surah or a hadith number."""
        return _nonempty(row.get("السورة")) or _nonempty(row.get("رقم الحديث"))

    # ── Step 1: All QA ────────────────────────────────────────────── #

    def build_all_qa(self, df: pd.DataFrame | None = None) -> list[dict]:
        """Generate QA pairs for all roots and combine into full_QA.json."""
        if df is None:
            df = self._load_corpus()

        all_pairs: list[dict] = []
        roots_written = 0

        for root_id, group in df.groupby("rootId"):
            root_pairs: list[dict] = []
            for _, row in group.iterrows():
                root_pairs.extend(_generate_qa_for_row(row))

            if root_pairs:
                out = self.qa_roots_all / f"qa_root_{root_id}.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(root_pairs, f, ensure_ascii=False, indent=2)
                all_pairs.extend(root_pairs)
                roots_written += 1

        combined_path = self.combined_dir / "full_QA.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)

        print(f"[all_qa]     {roots_written} roots, {len(all_pairs):,} pairs → {combined_path}")
        return all_pairs

    # ── Step 2: Islamic QA ────────────────────────────────────────── #

    def build_islamic_qa(self, df: pd.DataFrame | None = None) -> list[dict]:
        """Filter to Islamic entries, build per-root JSONs and combined files."""
        if df is None:
            df = self._load_corpus()

        mask = df.apply(self._is_islamic, axis=1)
        df_islamic = df[mask].copy()
        print(f"[islamic_qa] {len(df_islamic):,} Islamic rows "
              f"({mask.sum() / len(df) * 100:.1f}% of corpus)")

        all_pairs: list[dict] = []
        roots_written = 0

        for root_id, group in df_islamic.groupby("rootId"):
            root_pairs: list[dict] = []
            for _, row in group.iterrows():
                root_pairs.extend(_generate_qa_for_row(row))

            if root_pairs:
                out = self.qa_roots_islamic / f"qa_root_{root_id}.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(root_pairs, f, ensure_ascii=False, indent=2)
                all_pairs.extend(root_pairs)
                roots_written += 1

        # islamic_QA_comprehensive.json
        comp_path = self.combined_dir / "islamic_QA_comprehensive.json"
        with open(comp_path, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, ensure_ascii=False, indent=2)
        print(f"[islamic_qa] {roots_written} roots, {len(all_pairs):,} pairs → {comp_path}")

        # islamic_QA_meaning.json — meaning types only
        meaning_pairs = [p for p in all_pairs if p["type"] in self._MEANING_TYPES]
        meaning_path = self.combined_dir / "islamic_QA_meaning.json"
        with open(meaning_path, "w", encoding="utf-8") as f:
            json.dump(meaning_pairs, f, ensure_ascii=False, indent=2)
        print(f"[islamic_qa] {len(meaning_pairs):,} meaning pairs → {meaning_path}")

        return all_pairs

    # ── Step 3: Stratified 2 000-row sample ───────────────────────── #

    def sample_islamic_full(
        self,
        islamic_data: list[dict] | None = None,
        total: int = 2000,
    ) -> pd.DataFrame:
        """Stratified sample of *total* rows across all question types."""
        if islamic_data is None:
            comp_path = self.combined_dir / "islamic_QA_comprehensive.json"
            with open(comp_path, "r", encoding="utf-8") as f:
                islamic_data = json.load(f)

        df = _dedup_qa_pairs(islamic_data)
        result = _stratified_sample(df, total=total)
        out = self.evaluation_dir / "islamic_full_testing.csv"
        result[["lemma", "type", "question", "answer"]].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[sample_full]     {len(result):,} rows → {out}")
        print(f"  type distribution: {result['type'].value_counts().to_dict()}")
        return result

    # ── Step 4: Meanings-only 1 000-row sample ────────────────────── #

    def sample_islamic_meanings(
        self,
        islamic_data: list[dict] | None = None,
        n_per_type: int = 500,
    ) -> pd.DataFrame:
        """500 contextual_meaning + 500 basic_meaning rows."""
        if islamic_data is None:
            meaning_path = self.combined_dir / "islamic_QA_meaning.json"
            with open(meaning_path, "r", encoding="utf-8") as f:
                islamic_data = json.load(f)

        df = _dedup_qa_pairs(islamic_data)
        parts: list[pd.DataFrame] = []
        for t in self._MEANING_TYPES:
            group = df[df["type"] == t]
            take = min(len(group), n_per_type)
            parts.append(group.sample(n=take, random_state=42))
            print(f"[sample_meanings] {t}: {take} rows sampled "
                  f"(available: {len(group):,})")

        result = pd.concat(parts, ignore_index=True)
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        out = self.evaluation_dir / "islamic_meaning_testing.csv"
        result[["lemma", "type", "question", "answer"]].to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[sample_meanings] {len(result):,} rows → {out}")
        return result

    # ── Run all steps ─────────────────────────────────────────────── #

    def run(self) -> None:
        """Execute all four steps in order."""
        print("=== QADataGenerator starting ===")
        df = self._load_corpus()
        self.build_all_qa(df=df)
        islamic_data = self.build_islamic_qa(df=df)
        self.sample_islamic_full(islamic_data=islamic_data)
        self.sample_islamic_meanings()   # reads islamic_QA_meaning.json
        print("=== Done ===")


# ── CLI ───────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate QA pairs from the DHDA filtered corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--corpus",
        default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv",
        help="Path to DHDA_filtered_AR.csv (Arabic-named columns file).",
    )
    p.add_argument(
        "--output-dir",
        default="data/questions_data/QA_data",
        help="Root folder for all QA output files (expects QA_roots/, QA_combined/, evaluation_data/ inside).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    gen = QADataGenerator(corpus_csv=args.corpus, output_dir=args.output_dir)
    gen.run()
