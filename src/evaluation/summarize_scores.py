"""
summarize_scores.py — Summarise and compare LLM-judge scoring results.

Two modes
---------
**Single file** — print score statistics for one judging output CSV::

    python src/evaluation/summarize_scores.py --input judging/gemini_fs_judged.csv

**Compare two files** — print side-by-side statistics and per-question diffs::

    python src/evaluation/summarize_scores.py \\
        --input  judging/fanar_fs_judged.csv \\
        --input2 judging/gemini_fs_judged.csv
"""

from __future__ import annotations

import argparse
import csv
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────── #
# Helpers                                                                       #
# ──────────────────────────────────────────────────────────────────────────── #

def _parse_score(score_str: str) -> Optional[float]:
    if not score_str or score_str.strip().upper() == "ERROR":
        return None
    try:
        return float(score_str.strip().rstrip("%"))
    except (ValueError, AttributeError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return (s[n // 2 - 1] + s[n // 2]) / 2 if n % 2 == 0 else s[n // 2]


def _load_file(path: str) -> dict[str, dict]:
    data: dict[str, dict] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            data[row["question"]] = {
                "score":         _parse_score(row["score"]),
                "score_raw":     row["score"],
                "justification": row.get("justification", ""),
            }
    return data


def _print_stats(label: str, scores: list[float], total: int) -> None:
    print(f"\n  {label}:")
    print(f"    Valid scores : {len(scores)}/{total}")
    if scores:
        print(f"    Average      : {_mean(scores):.2f}%")
        print(f"    Median       : {_median(scores):.2f}%")
        print(f"    Min / Max    : {min(scores):.0f}% / {max(scores):.0f}%")

        bins = {"0-24%": 0, "25-49%": 0, "50-74%": 0, "75-99%": 0, "100%": 0}
        for s in scores:
            if s < 25:
                bins["0-24%"] += 1
            elif s < 50:
                bins["25-49%"] += 1
            elif s < 75:
                bins["50-74%"] += 1
            elif s < 100:
                bins["75-99%"] += 1
            else:
                bins["100%"] += 1
        print("    Distribution :", "  ".join(f"{k}:{v}" for k, v in bins.items()))


# ──────────────────────────────────────────────────────────────────────────── #
# Single-file summary                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def summarize_single(path: str) -> None:
    data = _load_file(path)
    scores = [v["score"] for v in data.values() if v["score"] is not None]
    errors = sum(1 for v in data.values() if v["score"] is None)

    print(f"\n{'=' * 60}")
    print(f"JUDGING SUMMARY: {path}")
    print(f"{'=' * 60}")
    print(f"  Total rows : {len(data)}")
    print(f"  Errors     : {errors}")
    _print_stats("Scores", scores, len(data))


# ──────────────────────────────────────────────────────────────────────────── #
# Two-file comparison                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def compare_two(path1: str, path2: str) -> None:
    data1 = _load_file(path1)
    data2 = _load_file(path2)

    common = set(data1) & set(data2)
    print(f"\n{'=' * 60}")
    print("JUDGING SCORE COMPARISON")
    print(f"{'=' * 60}")
    print(f"  File 1 : {path1}  ({len(data1)} rows)")
    print(f"  File 2 : {path2}  ({len(data2)} rows)")
    print(f"  Common questions : {len(common)}")

    scores1 = [data1[q]["score"] for q in common if data1[q]["score"] is not None]
    scores2 = [data2[q]["score"] for q in common if data2[q]["score"] is not None]

    _print_stats("File 1", scores1, len(common))
    _print_stats("File 2", scores2, len(common))

    diffs = []
    better_in_2, better_in_1, same = [], [], []

    for q in common:
        s1, s2 = data1[q]["score"], data2[q]["score"]
        if s1 is None or s2 is None:
            continue
        d = s2 - s1
        diffs.append(d)
        if d > 0:
            better_in_2.append((q, s1, s2, d))
        elif d < 0:
            better_in_1.append((q, s1, s2, abs(d)))
        else:
            same.append(q)

    if diffs:
        print(f"\n  Score differences (File 2 − File 1):")
        print(f"    Average diff : {_mean(diffs):.2f} pts")
        print(f"    Median diff  : {_median(diffs):.2f} pts")
        print(f"    Identical    : {len(same)}")
        print(f"    File 2 better: {len(better_in_2)}")
        print(f"    File 1 better: {len(better_in_1)}")

    if better_in_2:
        print(f"\n  Top-10 improvements in File 2:")
        for i, (q, s1, s2, d) in enumerate(
            sorted(better_in_2, key=lambda x: x[3], reverse=True)[:10], 1
        ):
            q_short = q[:80] + "..." if len(q) > 80 else q
            print(f"    {i}. {q_short}")
            print(f"       {s1:.0f}% → {s2:.0f}%  (+{d:.0f} pts)")

    if better_in_1:
        print(f"\n  Top-10 improvements in File 1:")
        for i, (q, s1, s2, d) in enumerate(
            sorted(better_in_1, key=lambda x: x[3], reverse=True)[:10], 1
        ):
            q_short = q[:80] + "..." if len(q) > 80 else q
            print(f"    {i}. {q_short}")
            print(f"       {s1:.0f}% → {s2:.0f}%  (-{d:.0f} pts)")


# ──────────────────────────────────────────────────────────────────────────── #
# CLI                                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarise or compare LLM-judge scoring results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        required=True,
        help="Judging output CSV (columns: question, reference, candidate, score, justification).",
    )
    p.add_argument(
        "--input2",
        default=None,
        help="Second judging output CSV for side-by-side comparison.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.input2:
        compare_two(args.input, args.input2)
    else:
        summarize_single(args.input)


if __name__ == "__main__":
    main()
