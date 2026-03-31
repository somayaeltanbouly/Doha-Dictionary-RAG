"""
data_utils.py — Shared utilities for the DHDA data processing pipeline
=======================================================================

This module contains lightweight helper functions that are used by two or more
pipeline scripts.  Centralising them here ensures consistent behaviour across
all scripts and eliminates maintenance overhead from keeping multiple copies in
sync.

Functions
---------
strip_tashkeel(text)
    Remove Arabic diacritics (tashkeel) from a string.

nonempty(value)
    Return True when a value is a non-null, non-blank, non-'nan' string.

word_type(text)
    Return 'عبارة' (phrase) or 'كلمة' (word) depending on whether the text
    contains a space.

parse_meaning(meaning)
    Split a meaning string into (meaning_head, meaning_2) at the first ':' or
    '؛' separator.

normalize_author(author)
    Normalise generic hadith attribution labels to the Prophet's full name.
"""
from __future__ import annotations

import re

# ── Tashkeel removal ──────────────────────────────────────────────── #

_TASHKEEL_RE = re.compile(
    r"[\u064B-\u065F"   # fathatan … sukun
    r"\u0610-\u061A"    # extended Arabic marks
    r"\u06D6-\u06DC"    # Quranic annotation marks
    r"\u06DF-\u06E4"    # Quranic annotation marks (cont.)
    r"\u06E7\u06E8"     # Arabic small high yeh / noon
    r"\u06EA-\u06ED]"   # Quranic annotation marks (cont.)
)


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritical marks from *text*."""
    return _TASHKEEL_RE.sub("", text)


# ── String validation ─────────────────────────────────────────────── #

def nonempty(value) -> bool:
    """Return True when *value* is a non-null, non-blank, non-'nan' string."""
    if value is None:
        return False
    if _is_na(value):
        return False
    return str(value).strip() not in ("", "nan")


def _is_na(value) -> bool:
    """Thin wrapper around pandas isna that avoids importing pandas at module level."""
    try:
        import pandas as pd
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


# ── Arabic question-building helpers ─────────────────────────────── #

def word_type(text: str) -> str:
    """Return 'عبارة' for multi-word text, 'كلمة' for single-word text."""
    return "عبارة" if " " in str(text).strip() else "كلمة"


def parse_meaning(meaning: str) -> tuple[str, str]:
    """
    Split a meaning string into ``(meaning_head, meaning_2)``.

    The split is attempted at the first ``':'`` character, then at the first
    ``'؛'`` (Arabic semicolon).  If neither separator is found both returned
    values are equal to the full input string.
    """
    s = str(meaning).strip()
    if ":" in s:
        head, rest = s.split(":", 1)
        return head, rest
    if "؛" in s:
        head, rest = s.split("؛", 1)
        return head, rest
    return s, s


def normalize_author(author: str) -> str:
    """
    Normalise generic hadith attribution labels to the Prophet's full name.

    Replaces 'الحديث النبوي' and 'حديث نبوي' with the formal attribution
    'النبي محمد صلى الله عليه وسلم'.
    """
    if "الحديث النبوي" in author or "حديث نبوي" in author:
        return "النبي محمد صلى الله عليه وسلم"
    return author
