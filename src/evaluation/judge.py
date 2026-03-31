"""
judge.py — LLM-as-judge evaluation for the Doha Dictionary RAG system.

Uses Gemini as an automatic evaluator to score model-generated answers against
gold reference answers on a 5-point scale (0 / 25 / 50 / 75 / 100 %).

The judge prompt evaluates two dimensions:
- **Factual correctness** — no wrong dates, names, meanings, or fabricated sources.
- **Completeness** — covers all key facts from the reference answer.

Usage::

    python src/evaluation/judge.py \\
        --input  output/gemini_fs_results.csv \\
        --output judging/gemini_fs_judged.csv

The output CSV is written row-by-row so progress is preserved on interruption.
Re-running with the same output path **appends** to the existing file, so
truncate or delete the output first if you want a clean run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import timedelta

import pandas as pd


# ──────────────────────────────────────────────────────────────────────────── #
# Judge prompt                                                                  #
# ──────────────────────────────────────────────────────────────────────────── #

_JUDGE_PROMPT = """\
You are a fair and consistent evaluator specialized in historical Arabic texts.

You will be given:
1. A system output (candidate).
2. A reference answer (gold standard).

Your task: Evaluate how well the candidate matches the reference.

Evaluation dimensions:
- Factual correctness (no wrong dates, names, meanings, or fabricated sources).
- Completeness (covers all key facts from the reference).

NOTES:
- Consider the question as well while evaluating the candidate answer to confirm that the answer is a valid answer for the question.
- Please consider the answer as correct if it is semantically correct. For example, if the candidate answer is "الحديث النبوي" and the reference answer is "النبي صلى الله عليه وسلم", consider it as fully correct, since both of them refer to the prophet.
- Don't use score 0 unless the answer is completely wrong.
- Missing minor details is fine in all scores, as long as the main answer is there and the answer is covering what the user is asking about.

Scoring Rules:
- 0%   = Wrong answer (any incorrect fact such as wrong dates, names, meanings, or fabricated sources).
- 25%  = Partially correct answer (some correct elements, but also incorrect details and missing important parts of the reference).
- 50%  = Factually correct but incomplete (all stated facts are correct, but key details from the reference are missing).
- 75%  = Mostly correct and mostly complete (facts are correct and most details are covered, but the answer lacks full precision).
- 100% = Correct and complete (all facts match the reference, with nothing important missing or fabricated).

The question: {question}
The candidate answer: {model_answer}
The reference answer: {correct_answer}

Return your judgment strictly in **JSON** format, no extra text:
{{
  "score": "<0% | 25% | 50% | 75% | 100%>",
  "explanation": "<short reasoning (in Arabic)>"
}}
"""

_GEMINI_MODEL = "gemini-2.5-pro"


# ──────────────────────────────────────────────────────────────────────────── #
# Gemini caller                                                                 #
# ──────────────────────────────────────────────────────────────────────────── #

def _build_gemini_model():
    """Initialise the Gemini client once and return ``(model, genai)``."""
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=_GEMINI_MODEL,
        generation_config={"response_mime_type": "application/json"},
    )
    return model, genai


def _call_gemini(model, genai_module, question: str, model_answer: str, correct_answer: str) -> tuple[str, str]:
    """Call Gemini judge and return (score, explanation).

    Returns ``("ERROR", raw_output)`` if JSON parsing fails.
    """
    prompt = _JUDGE_PROMPT.format(
        question=question,
        model_answer=model_answer,
        correct_answer=correct_answer,
    )
    response = model.generate_content(
        prompt,
        generation_config=genai_module.types.GenerationConfig(temperature=0),
    )
    raw = response.text

    try:
        parsed = json.loads(raw)
        return parsed.get("score", ""), parsed.get("explanation", "")
    except Exception:
        return "ERROR", f"Raw output: {raw}"


# ──────────────────────────────────────────────────────────────────────────── #
# Main evaluation loop                                                          #
# ──────────────────────────────────────────────────────────────────────────── #

def evaluate_dataset(input_file: str, output_file: str, api_delay: float = 1.0) -> None:
    """Score every row in *input_file* and write results to *output_file*.

    Args:
        input_file:  CSV with columns ``question``, ``answer``, ``correct_answer``.
        output_file: Destination CSV. Appends if the file already exists.
        api_delay:   Seconds to sleep between API calls (rate-limit buffer).
    """
    data = pd.read_csv(input_file)
    total = len(data)
    print(f"Judging {total} answers from: {input_file}")

    # Initialise Gemini once — avoids re-configuring the client on every row.
    gemini_model, genai_module = _build_gemini_model()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    file_exists = os.path.isfile(output_file)
    fieldnames = ["question", "reference", "candidate", "score", "justification"]

    with open(output_file, "a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        start_time = time.time()
        for idx, row in data.iterrows():
            try:
                score, explanation = _call_gemini(
                    gemini_model, genai_module,
                    question=row["question"],
                    model_answer=row["answer"],
                    correct_answer=row["correct_answer"],
                )
                writer.writerow({
                    "question":      row["question"],
                    "reference":     row["correct_answer"],
                    "candidate":     row["answer"],
                    "score":         score,
                    "justification": explanation,
                })
                fh.flush()

                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                eta = str(timedelta(seconds=int(avg_time * (total - idx - 1))))
                print(f"[{idx + 1}/{total}] score={score} | ETA: {eta}")
                time.sleep(api_delay)

            except Exception as exc:
                print(f"Error on row {idx}: {exc}")
                continue

    print(f"\nDone. Results saved to: {output_file}")


# ──────────────────────────────────────────────────────────────────────────── #
# CLI                                                                           #
# ──────────────────────────────────────────────────────────────────────────── #

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score model answers using Gemini as an LLM judge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default="output/model_answers.csv",
        help="CSV with columns: question, answer, correct_answer.",
    )
    p.add_argument(
        "--output",
        default="judging/judging_results.csv",
        help="Output CSV path for judging results.",
    )
    p.add_argument(
        "--api-delay",
        type=float,
        default=1.0,
        dest="api_delay",
        help="Seconds to sleep between Gemini API calls.",
    )
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    evaluate_dataset(
        input_file=args.input,
        output_file=args.output,
        api_delay=args.api_delay,
    )


if __name__ == "__main__":
    main()
