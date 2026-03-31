"""
ClassificationDataGenerator
============================

Generates labelled question samples for question-type classification from the
processed DHDA corpus.

Output columns
--------------
``type, question``

12 question types
-----------------
Entry-level  (one candidate question per corpus row):
  basic_meaning, contextual_meaning, part_of_speech,
  author_of_citation, historical_date, source_of_citation

Root-level  (one candidate question per unique Arabic root):
  etymology           — roots that have any etymological cross-language entry
                        (filter: السورة not needed, use all roots or etymol. data)
  first_quranic_usage — roots whose entries reference the Quran  (السورة)
  first_usage         — all roots (every root has at least one dated entry)
  list_derivations    — all roots
  terminological_usage — roots with at least one semantic-field entry
                         (الحقل الاصطلاحي)

Inscription-level  (one candidate per root present in inscriptions data):
  inscription         — unique Arabic roots from DHDA_inscriptions_data.csv,
                        joined with main corpus for the Arabic root string

Sampling
--------
*n_per_type* rows are sampled per type (default=1000) using a fixed
random seed, then shuffled and saved as ``classification_data.csv``.

Questions are deduplicated per type before sampling so that the same
question cannot appear twice under the same label.

Defaults
--------
- Corpus          : data/retrieval_corpus/processed/DHDA_filtered_AR.csv
- Inscriptions    : data/retrieval_corpus/processed/DHDA_inscriptions_data.csv
- Output          : data/questions_data/classification_data/classification_data.csv
- n_per_type      : 1000
- random_state    : 42

Usage::

    python src/data_processing/build_classification_data.py

    # explicit paths:
    python src/data_processing/build_classification_data.py \\
        --corpus       data/retrieval_corpus/processed/DHDA_filtered_AR.csv \\
        --inscriptions data/retrieval_corpus/processed/DHDA_inscriptions_data.csv \\
        --output       data/questions_data/classification_data/classification_data.csv \\
        --n-per-type   1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running directly: python src/data_processing/build_classification_data.py
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from data_processing.data_utils import nonempty, parse_meaning, word_type

# Aliases so all existing call-sites work without change.
_nonempty      = nonempty
_word_type     = word_type
_parse_meaning = parse_meaning


# ── Entry-level question generation ───────────────────────────────── #

def _entry_questions_for_row(row: pd.Series) -> list[dict]:
    """Return (type, question) dicts for one corpus row — no answers needed.

    All surface variants observed in the reference combined_sampled_data.csv
    are reproduced here so the classifier is exposed to every format.
    """
    pairs: list[dict] = []

    lemma_val = str(row["الكلمة"]).strip()
    meaning   = row["المعنى"]
    citation  = row["الشاهد"]
    head_cit  = row["مقدمة الشاهد"]
    author    = row["القائل"]
    source    = row["المصدر"]
    date      = row["تاريخ استعمال الشاهد"]
    pos       = row["الاشتقاق الصرفي للكلمة"]

    if not _nonempty(meaning):
        return pairs

    meaning_str             = str(meaning).strip()
    meaning_head, meaning_2 = _parse_meaning(meaning_str)
    wt                      = _word_type(meaning_head)

    # ── basic_meaning ────────────────────────────────────────────────
    # 6 variants: 3 question prefixes × 2 word-types (كلمة / عبارة)
    # Reference counts: ما معنى (968x), ما هو معنى (27x), معنى (5x)
    if meaning_head:
        for prefix in ("ما معنى", "ما هو معنى", "معنى"):
            pairs.append({
                "type":     "basic_meaning",
                "question": f'{prefix} {wt} "{meaning_head}"؟',
            })

    # ── contextual_meaning ───────────────────────────────────────────
    # 3 surface patterns observed in the reference data:
    #   1. Citation-first  (604x): في الشاهد التالي: "X"، ما معنى {wt} "Y"؟
    #   2. Word-first A   (386x):  ما هو معنى {wt} "Y"،؟ في الشاهد التالي: "X"
    #   3. Word-first B    (10x):  ما معنى {wt} "Y"،؟ في الشاهد التالي: "X"
    # Generated for every row that has a citation so that all formats are covered.
    if _nonempty(citation):
        cit_str = str(citation).strip()
        pairs.append({
            "type":     "contextual_meaning",
            "question": f'في الشاهد التالي: "{cit_str}"، ما معنى {wt} "{lemma_val}"؟',
        })
        pairs.append({
            "type":     "contextual_meaning",
            "question": f' ما هو معنى {wt} "{lemma_val}"،؟ في الشاهد التالي: "{cit_str}"',
        })
        pairs.append({
            "type":     "contextual_meaning",
            "question": f' ما معنى {wt} "{lemma_val}"،؟ في الشاهد التالي: "{cit_str}"',
        })

    # ── part_of_speech ───────────────────────────────────────────────
    # Single template (1000x in reference)
    if _nonempty(pos):
        pairs.append({
            "type":     "part_of_speech",
            "question": f'ما الاشتقاق الصرفي لكلمة "{lemma_val}"؟',
        })

    # ── author_of_citation ───────────────────────────────────────────
    # Multiple variants observed in reference:
    #   من القائل الذي استخدم {wt} "X" في الشاهد: "Y"؟  (975x — استخدم)
    #   من القائل الذي استعمل {wt} "X" في الشاهد: "Y"؟  (2x  — استعمل)
    #   من قائل: "citation"؟                              (7x)
    #   من قائل الشاهد: "citation"؟                       (4x)
    #   من القائل: "citation"؟                             (4x)
    #   من القائل للشاهد: "citation"؟                      (2x)
    if _nonempty(author) and _nonempty(citation):
        cit_str = str(citation).strip()
        pairs.append({
            "type":     "author_of_citation",
            "question": f'من القائل الذي استخدم {wt} "{lemma_val}" في الشاهد: "{cit_str}"؟',
        })
        pairs.append({
            "type":     "author_of_citation",
            "question": f'من القائل الذي استعمل {wt} "{lemma_val}" في الشاهد: "{cit_str}"؟',
        })
        pairs.append({
            "type":     "author_of_citation",
            "question": f'من قائل: "{cit_str}"؟',
        })
        pairs.append({
            "type":     "author_of_citation",
            "question": f'من قائل الشاهد: "{cit_str}"؟',
        })
        pairs.append({
            "type":     "author_of_citation",
            "question": f'من القائل: "{cit_str}"؟',
        })
        pairs.append({
            "type":     "author_of_citation",
            "question": f'من القائل للشاهد: "{cit_str}"؟',
        })

    # ── historical_date ──────────────────────────────────────────────
    # 2 variants: كلمة (414x) / عبارة (586x) — determined by wt from meaning_head
    if _nonempty(date) and _nonempty(citation) and meaning_head:
        pairs.append({
            "type":     "historical_date",
            "question": f'ما هو تاريخ الشاهد الذي استعمل فيه {wt} "{meaning_head}" بمعنى "{meaning_2}"',
        })

    # ── source_of_citation ───────────────────────────────────────────
    # 2 variants: كلمة (414x) / عبارة (586x) — determined by wt from meaning_head
    if _nonempty(source) and _nonempty(citation) and meaning_head:
        pairs.append({
            "type":     "source_of_citation",
            "question": f'ما هو المصدر الذي ورد فيه شاهد استخدام {wt} "{lemma_val}"؟ بمعنى "{meaning_2}"',
        })

    return pairs


# ── Root-level question generation ────────────────────────────────── #

def _root_questions(df: pd.DataFrame) -> list[dict]:
    """
    Generate one question per unique root for each root-level type.

    Filters applied:
    - first_quranic_usage : roots where at least one entry has a Quran surah
                            (السورة)
    - terminological_usage: roots where at least one entry has a semantic field
                            (الحقل الاصطلاحي)
    - etymology, first_usage, list_derivations: all unique roots
    """
    pairs: list[dict] = []

    # All unique roots (rootId → Arabic root string الجذر)
    all_roots = (
        df[["rootId", "الجذر"]]
        .dropna(subset=["الجذر"])
        .drop_duplicates("rootId")
        .set_index("rootId")["الجذر"]
    )

    # Roots with Quran entries
    quran_roots = set(
        df[df["السورة"].notna() & (df["السورة"].astype(str).str.strip() != "")]
        ["rootId"]
    )

    # Roots with semantic-field entries
    term_roots = set(
        df[df["الحقل الاصطلاحي"].notna() & (df["الحقل الاصطلاحي"].astype(str).str.strip() != "")]
        ["rootId"]
    )

    for root_id, root in all_roots.items():
        root = str(root).strip()
        if not root or root == "nan":
            continue

        # list_derivations — all roots
        pairs.append({
            "type":     "list_derivations",
            "question": f'ما هي الاشتقاقات الموثقة للجذر "{root}"؟',
        })

        # first_usage — all roots
        pairs.append({
            "type":     "first_usage",
            "question": f'متى تم توثيق أول استخدام للجذر "{root}" في المصادر، وبأي صيغة؟',
        })

        # etymology — all roots (cross-language cognates can be asked for any root)
        pairs.append({
            "type":     "etymology",
            "question": f'هل للجذر العربي "{root}" أصول أو نظائر في لغات أخرى؟',
        })

        # terminological_usage — roots with semantic field data
        if root_id in term_roots:
            pairs.append({
                "type":     "terminological_usage",
                "question": f'هل للجذر "{root}" استخدام اصطلاحي متخصص؟',
            })

        # first_quranic_usage — roots that appear in Quran
        if root_id in quran_roots:
            pairs.append({
                "type":     "first_quranic_usage",
                "question": f'ما هو أول اشتقاق من الجذر "{root}" ظهر في القرآن الكريم حسب التسلسل الزمني للنزول؟',
            })

    return pairs


# ── Inscription question generation ───────────────────────────────── #

def _inscription_questions(
    inscriptions_csv: Path,
    df_main: pd.DataFrame,
) -> list[dict]:
    """
    One question per unique root present in the inscriptions corpus.

    Arabic root string is obtained by joining on rootId with the main corpus.
    Note: no space between لجذر and the opening quote — matches reference data.
    """
    try:
        df_ins = pd.read_csv(inscriptions_csv, encoding="utf-8-sig", low_memory=False)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print(f"[inscription] Warning: could not load {inscriptions_csv} — skipping inscription type")
        return []

    # Determine rootId column name (may differ between raw and processed)
    root_id_col = "rootId" if "rootId" in df_ins.columns else None
    if root_id_col is None:
        print("[inscription] Warning: rootId column not found — skipping inscription type")
        return []

    inscription_root_ids = df_ins[root_id_col].dropna().unique()

    # Map rootId → Arabic root via main corpus
    root_map = (
        df_main[["rootId", "الجذر"]]
        .dropna(subset=["الجذر"])
        .drop_duplicates("rootId")
        .set_index("rootId")["الجذر"]
        .to_dict()
    )

    pairs: list[dict] = []
    for rid in inscription_root_ids:
        root = root_map.get(rid)
        if not root or str(root).strip() in ("", "nan"):
            continue
        pairs.append({
            "type":     "inscription",
            "question": f'هل لجذر"{str(root).strip()}" شواهد من النقوش القديمة؟',
        })
    return pairs


# ── Main class ────────────────────────────────────────────────────── #

class ClassificationDataGenerator:
    """Generate type-labelled question samples for question-intent classification."""

    _ENTRY_TYPES = frozenset({
        "basic_meaning", "contextual_meaning", "part_of_speech",
        "author_of_citation", "historical_date", "source_of_citation",
    })
    _ROOT_TYPES = frozenset({
        "etymology", "first_quranic_usage", "first_usage",
        "list_derivations", "terminological_usage",
    })

    def __init__(
        self,
        corpus_csv: str | Path,
        inscriptions_csv: str | Path,
        output_path: str | Path = "data/questions_data/classification_data/classification_data.csv",
        n_per_type: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.corpus_csv       = Path(corpus_csv)
        self.inscriptions_csv = Path(inscriptions_csv)
        self.output_path      = Path(output_path)
        self.n_per_type       = n_per_type
        self.random_state     = random_state
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        """Generate all questions, sample, and save."""
        df = pd.read_csv(self.corpus_csv, encoding="utf-8-sig", low_memory=False)
        print(f"[load]       {len(df):,} rows from {self.corpus_csv}")

        # ── Entry-level questions ──────────────────────────────────────── #
        entry_records: list[dict] = []
        for _, row in df.iterrows():
            entry_records.extend(_entry_questions_for_row(row))
        print(f"[entry]      {len(entry_records):,} raw entry-level questions generated")

        # ── Root-level questions ──────────────────────────────────────────── #
        root_records = _root_questions(df)
        print(f"[root]       {len(root_records):,} raw root-level questions generated")

        # ── Inscription questions ─────────────────────────────────────────── #
        inscription_records = _inscription_questions(self.inscriptions_csv, df)
        print(f"[inscription]{len(inscription_records):,} raw inscription questions generated")

        # ── Combine and deduplicate per type ──────────────────────────────── #
        all_records = entry_records + root_records + inscription_records
        full_df = pd.DataFrame(all_records, columns=["type", "question"])
        full_df = full_df.drop_duplicates(subset=["type", "question"])
        print(f"[dedup]      {len(full_df):,} unique (type, question) pairs")
        print(f"             available per type:")
        for t, n in full_df["type"].value_counts().items():
            print(f"               {t}: {n:,}")

        # ── Sample n_per_type per type ────────────────────────────────────── #
        parts: list[pd.DataFrame] = []
        for t, group in full_df.groupby("type"):
            take = min(len(group), self.n_per_type)
            parts.append(group.sample(n=take, random_state=self.random_state))
            if take < self.n_per_type:
                print(f"[sample]     Warning: {t} has only {take} unique questions "
                      f"(requested {self.n_per_type})")

        result = (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1, random_state=self.random_state)
            .reset_index(drop=True)
        )

        result.to_csv(self.output_path, index=False, encoding="utf-8-sig")
        print(f"[save]       {len(result):,} rows → {self.output_path}")
        print(f"             final type distribution:")
        for t, n in result["type"].value_counts().items():
            print(f"               {t}: {n:,}")
        return result


# ── CLI ───────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate type-labelled question data for classification.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--corpus",
        default="data/retrieval_corpus/processed/DHDA_filtered_AR.csv",
        help="Path to DHDA_filtered_AR.csv (Arabic-column corpus).",
    )
    p.add_argument(
        "--inscriptions",
        default="data/retrieval_corpus/processed/DHDA_inscriptions_data.csv",
        help="Path to DHDA_inscriptions_data.csv.",
    )
    p.add_argument(
        "--output",
        default="data/questions_data/classification_data/classification_data.csv",
        help="Output path for classification_data.csv.",
    )
    p.add_argument(
        "--n-per-type",
        type=int,
        default=1000,
        help="Number of questions to sample per type.",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    gen = ClassificationDataGenerator(
        corpus_csv=args.corpus,
        inscriptions_csv=args.inscriptions,
        output_path=args.output,
        n_per_type=args.n_per_type,
        random_state=args.random_state,
    )
    gen.run()
