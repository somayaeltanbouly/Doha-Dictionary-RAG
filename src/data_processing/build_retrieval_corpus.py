"""
RetrievalDataGenerator
======================

Transforms raw scraped data from ``data/retrieval_corpus/raw/`` into the processed files
needed by the retrieval pipeline.

Raw data layout expected under ``raw_dir``:
    raw_dir/
        lexical_details/*.csv
        etymological/*.csv
        inscriptions/*.csv

Outputs written to ``output_dir``:
    DHDA_filtered_EN.csv             — 20-column subset, English names
    DHDA_filtered_AR.csv             — same 20 columns, Arabic names
    DHDA_text_to_embed.csv           — single 'text' column (tashkeel stripped)
    DHDA_lexical_data.csv            — all lexical_details roots combined
    DHDA_etymological_data.csv       — all etymological roots combined
    DHDA_inscriptions_data.csv       — all inscriptions roots combined

Usage::

    python src/data_processing/build_retrieval_corpus.py

    # or with explicit paths:
    python src/data_processing/build_retrieval_corpus.py \\
        --raw-dir  data/retrieval_corpus/raw \\
        --output-dir data/retrieval_corpus/processed
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow running directly: python src/data_processing/build_retrieval_corpus.py
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from data_processing.data_utils import strip_tashkeel

# ── Column specs ──────────────────────────────────────────────────── #

# 20 columns to keep from lexical_details (English names, in order)
_ENGLISH_COLUMNS: list[str] = [
    "ID",
    "rootId",
    "lemmaId",
    "rootValue",
    "lemmaValueUV",
    "lemmaValue",
    "additionalTag",
    "meaningHead",
    "headCitation",
    "citation",
    "meaning",
    "authorName",
    "verbaldate",
    "semanticFieldValue",
    "source",
    "referenceSourcePage",
    "referenceSourceReadingQuranStr",
    "referenceSourceAyahNbr",
    "referenceSourceHaditNbr",
    "remarksargument",
]

# Arabic display names — same order as _ENGLISH_COLUMNS
_ARABIC_COLUMNS: list[str] = [
    "ID",
    "rootId",
    "lemmaId",
    "الجذر",
    "الكلمة بدون تشكيل",
    "الكلمة",
    "الاشتقاق الصرفي للكلمة",
    "العبارة أو اللفظ المركب",
    "مقدمة الشاهد",
    "الشاهد",
    "المعنى",
    "القائل",
    "تاريخ استعمال الشاهد",
    "الحقل الاصطلاحي",
    "المصدر",
    "رقم الصفحة",
    "السورة",
    "رقم الآية",
    "رقم الحديث",
    "تعليقات إضافية",
]

_EN_TO_AR: dict[str, str] = dict(zip(_ENGLISH_COLUMNS, _ARABIC_COLUMNS))

# Columns concatenated for the plain-text file
_TEXT_COLUMNS: list[str] = ["الكلمة", "الجذر", "المعنى", "الشاهد"]

# ── Helpers ───────────────────────────────────────────────────────── #

def _safe_str(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = text.replace("***", " ")
    return text


def _build_text(row: pd.Series, columns: list[str]) -> str:
    parts = [_safe_str(row[col]) for col in columns if col in row.index]
    return " ".join(p for p in parts if p)


def _fix_morphology(series: pd.Series) -> pd.Series:
    """Prepend 'فعل ' to bare 'متعد' or 'لازم' values."""
    return series.apply(
        lambda v: "فعل " + str(v) if isinstance(v, str) and str(v).strip() in {"متعد", "لازم"} else v
    )


# ── Main class ────────────────────────────────────────────────────── #

class RetrievalDataGenerator:
    """Generate retrieval-ready files from raw scraped CSVs."""

    RAW_TYPES = ("lexical_details", "etymological", "inscriptions")

    def __init__(self, raw_dir: str | Path, output_dir: str | Path) -> None:
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Functionality 1: Combine raw files ────────────────────────── #

    def combine_raw(self, data_type: str) -> pd.DataFrame:
        """Concatenate all per-root CSVs for *data_type* into one DataFrame."""
        folder = self.raw_dir / data_type
        files = sorted(folder.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {folder}")
        dfs = [pd.read_csv(f, encoding="utf-8-sig", low_memory=False) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        _name_map = {
            "lexical_details": "DHDA_lexical_data.csv",
            "etymological": "DHDA_etymological_data.csv",
            "inscriptions": "DHDA_inscriptions_data.csv",
        }
        out = self.output_dir / _name_map.get(data_type, f"DHDA_{data_type}_data.csv")
        combined.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[combine] {data_type}: {len(files)} file(s), {len(combined):,} rows → {out}")
        return combined

    def combine_all(self) -> dict[str, pd.DataFrame]:
        """Combine all three data types and return them as a dict."""
        return {t: self.combine_raw(t) for t in self.RAW_TYPES}

    # ── Functionality 2: Reduced file — English column names ─────── #

    def build_reduced_english(self, df_lexical: pd.DataFrame | None = None) -> pd.DataFrame:
        """Select the 20 spec columns (English names) from lexical_details."""
        if df_lexical is None:
            df_lexical = self.combine_raw("lexical_details")

        present = [c for c in _ENGLISH_COLUMNS if c in df_lexical.columns]
        missing = [c for c in _ENGLISH_COLUMNS if c not in df_lexical.columns]
        if missing:
            print(f"[english] Warning: columns not found and skipped: {missing}")

        df_out = df_lexical[present].copy()
        out = self.output_dir / "DHDA_filtered_EN.csv"
        df_out.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[english] {len(df_out):,} rows, {len(present)} cols → {out}")
        return df_out

    # ── Functionality 3: Reduced file — Arabic column names ──────── #

    def build_columns_arabic(self, df_english: pd.DataFrame | None = None) -> pd.DataFrame:
        """Rename English columns to Arabic display names.

        Also applies morphology fix: bare 'متعد' / 'لازم' in
        'الاشتقاق الصرفي للكلمة' are prefixed with 'فعل '.
        """
        if df_english is None:
            df_english = self.build_reduced_english()

        df_out = df_english.rename(columns=_EN_TO_AR).copy()

        morph_col = "الاشتقاق الصرفي للكلمة"
        if morph_col in df_out.columns:
            df_out[morph_col] = _fix_morphology(df_out[morph_col])

        out = self.output_dir / "DHDA_filtered_AR.csv"
        df_out.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[arabic]  {len(df_out):,} rows → {out}")
        return df_out

    # ── Functionality 4: Plain text file ─────────────────────────── #

    def build_text_data(self, df_arabic: pd.DataFrame | None = None) -> pd.DataFrame:
        """Concatenate الكلمة + الجذر + المعنى + الشاهد, strip tashkeel."""
        if df_arabic is None:
            df_arabic = self.build_columns_arabic()

        text_present = [c for c in _TEXT_COLUMNS if c in df_arabic.columns]
        text_missing = [c for c in _TEXT_COLUMNS if c not in df_arabic.columns]
        if text_missing:
            print(f"[text]    Warning: text columns not found and skipped: {text_missing}")

        texts = df_arabic.apply(
            lambda row: strip_tashkeel(_build_text(row, text_present)), axis=1
        )
        df_out = pd.DataFrame({"text": texts})
        out = self.output_dir / "DHDA_text_to_embed.csv"
        df_out.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"[text]    {len(df_out):,} rows → {out}")
        return df_out

    # ── Run all steps ─────────────────────────────────────────────── #

    def run(self) -> None:
        """Execute all four steps in order."""
        print("=== RetrievalDataGenerator starting ===")
        self.combine_all()
        df_en = self.build_reduced_english()
        df_ar = self.build_columns_arabic(df_english=df_en)
        self.build_text_data(df_arabic=df_ar)
        print("=== Done ===")


# ── CLI ───────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate retrieval-ready files from raw scraped CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--raw-dir",
        default="data/retrieval_corpus/raw",
        help="Root folder containing lexical_details/, etymological/, inscriptions/ subfolders.",
    )
    p.add_argument(
        "--output-dir",
        default="data/retrieval_corpus/processed",
        help="Folder where all processed CSV files will be written.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    gen = RetrievalDataGenerator(raw_dir=args.raw_dir, output_dir=args.output_dir)
    gen.run()
