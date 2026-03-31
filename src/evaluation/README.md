# src/evaluation/

Scripts for evaluating and comparing generation pipeline outputs.

---

## Files

| File | Description |
|---|---|
| `judge.py` | Gemini-as-judge scorer — assigns a factual correctness score to each model answer |
| `summarize_scores.py` | Print score statistics for one judging CSV, or compare two side-by-side |

---

## Prerequisites

All scripts expect to be run from the **repo root**.  The generation pipeline
must have produced an output CSV before running evaluation; generate answers
first with:

```bash
python run.py generate --model gemini --mode fs \
    --output output/gemini_fs_results.csv
```

`judge.py` requires a Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

---

## Quick start

```bash
# Score answers with the LLM judge
python run.py judge \
    --input  output/gemini_fs_results.csv \
    --output judging/gemini_fs_judged.csv

# Print statistics for a single judging file
python run.py summarize-scores --input judging/gemini_fs_judged.csv

# Compare two judging files side-by-side
python run.py summarize-scores \
    --input  judging/fanar_zs_judged.csv \
    --input2 judging/gemini_fs_judged.csv
```

---

## 1. `judge.py` — LLM-as-judge evaluation

Uses Gemini as an automatic evaluator to score model-generated answers against
gold reference answers.

### Input CSV format

| Column | Description |
|---|---|
| `question` | Arabic question |
| `answer` | Model-generated candidate answer |
| `correct_answer` | Gold reference answer |

### Output CSV format

| Column | Description |
|---|---|
| `question` | Original question |
| `reference` | Gold reference answer |
| `candidate` | Model-generated answer |
| `score` | Judge score: `0%`, `25%`, `50%`, `75%`, or `100%` |
| `justification` | Short Arabic explanation of the score |

### Scoring rubric

| Score | Meaning |
|---|---|
| `0%` | Wrong answer (incorrect facts, dates, names, or fabricated sources) |
| `25%` | Partially correct (some correct elements, incorrect details, missing key parts) |
| `50%` | Factually correct but incomplete (all stated facts correct, key details missing) |
| `75%` | Mostly correct and mostly complete (minor precision issues only) |
| `100%` | Correct and complete (all facts match the reference, nothing missing) |

### CLI options

| Option | Default | Description |
|---|---|---|
| `--input` | *(required)* | Input CSV with `question`, `answer`, `correct_answer` columns |
| `--output` | `judging/judging_results.csv` | Output CSV for judging results |
| `--api-delay` | `1.0` | Seconds to sleep between Gemini API calls (rate-limit buffer) |
| `--gemini-api-key` | — | API key (or set `GEMINI_API_KEY` env var) |

> **Note:** The output CSV is written row-by-row. If the run is interrupted,
> re-running with the same `--output` path will **append** to the existing file.
> Delete or truncate the output file first if you want a clean run.

---

## 2. `summarize_scores.py` — Score statistics

### Single-file mode

Prints score distribution, mean, median, min/max, and error count for one
judging output file.

```bash
python run.py summarize-scores --input judging/gemini_fs_judged.csv
```

Example output:
```
============================================================
JUDGING SUMMARY: judging/gemini_fs_judged.csv
============================================================
  Total rows : 1000
  Errors     : 2

  Scores:
    Valid scores : 998/1000
    Average      : 72.14%
    Median       : 75.00%
    Min / Max    : 0% / 100%
    Distribution : 0-24%:45  25-49%:78  50-74%:210  75-99%:315  100%:350
```

### Compare mode

Prints side-by-side statistics and a list of questions with the largest score
improvements between two files.

```bash
python run.py summarize-scores \
    --input  judging/fanar_zs_judged.csv \
    --input2 judging/gemini_fs_judged.csv
```

### CLI options

| Option | Default | Description |
|---|---|---|
| `--input` | *(required)* | Primary judging CSV |
| `--input2` | — | Second judging CSV (enables compare mode) |
