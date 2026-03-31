# Data Processing Pipeline

This directory contains the scripts that transform raw scraped data from the Doha Historical Dictionary of Arabic (DHDA) into every dataset needed by the retrieval and question-answering pipeline — from the processed retrieval corpus through to labelled training and evaluation files.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Pipeline Overview](#pipeline-overview)
5. [Script Reference](#script-reference)
   - [data_utils.py](#data_utilspy--shared-utilities)
   - [build_all.py](#build_allpy--orchestrator)
   - [build_retrieval_corpus.py](#build_retrieval_corpuspy--step-1)
   - [build_qa_pairs.py](#build_qa_pairspy--step-2)
   - [build_qd_pairs.py](#build_qd_pairspy--step-3)
   - [build_ce_training_data.py](#build_ce_training_datapy--step-4)
   - [build_classification_data.py](#build_classification_datapy--step-5)
6. [Output File Reference](#output-file-reference)
7. [Shared Conventions](#shared-conventions)

---

## Repository Layout

```
Doha-Dictionary-RAG/
├── data/
│   ├── retrieval_corpus/
│   │   ├── raw/                          # Raw scraped CSVs (input — not committed)
│   │   │   ├── lexical_details/          # One CSV per root — entry-level data
│   │   │   ├── etymological/             # One CSV per root — etymological data
│   │   │   └── inscriptions/             # One CSV per root — inscription data
│   │   └── processed/                    # Output of build_retrieval_corpus.py
│   │       ├── DHDA_filtered_EN.csv
│   │       ├── DHDA_filtered_AR.csv      # Primary corpus (consumed by steps 2–5)
│   │       ├── DHDA_text_to_embed.csv
│   │       ├── DHDA_lexical_data.csv
│   │       ├── DHDA_etymological_data.csv
│   │       └── DHDA_inscriptions_data.csv
│   └── questions_data/
│       ├── QA_data/                      # Output of build_qa_pairs.py
│       │   ├── QA_roots/
│       │   │   ├── all_data/             # qa_root_{rootId}.json (all entries)
│       │   │   └── islamic_data/         # qa_root_{rootId}.json (Islamic subset)
│       │   ├── QA_combined/
│       │   │   ├── full_QA.json
│       │   │   ├── islamic_QA_comprehensive.json
│       │   │   └── islamic_QA_meaning.json
│       │   └── evaluation_data/
│       │       ├── islamic_full_testing.csv
│       │       └── islamic_meaning_testing.csv
│       ├── QD_data/                      # Output of build_qd_pairs.py / build_ce_training_data.py
│       │   ├── QD_full_data.csv
│       │   ├── QD_evaluation_data.csv
│       │   └── QD_contrastive_data.csv
│       └── classification_data/          # Output of build_classification_data.py
│           └── classification_data.csv
└── src/
    └── data_processing/
        ├── data_utils.py                 # Shared helper functions (imported by all pipeline scripts)
        ├── build_all.py                  # Orchestrator — runs the full pipeline
        ├── build_retrieval_corpus.py     # Step 1
        ├── build_qa_pairs.py             # Step 2
        ├── build_qd_pairs.py             # Step 3
        ├── build_ce_training_data.py     # Step 4
        └── build_classification_data.py  # Step 5
```

---

## Prerequisites

```bash
pip install pandas
```

All scripts require only the Python standard library and `pandas`. No GPU or network access is needed.

Scripts should be run from the **repository root** (`Doha-Dictionary-RAG/`) so that the default relative paths under `data/` resolve correctly:

```bash
cd Doha-Dictionary-RAG
python src/data_processing/build_all.py
```

---

## Quick Start

**Run the complete pipeline with default settings:**

```bash
python src/data_processing/build_all.py
```

**Run only specific steps** (any subset, any order — execution always follows dependency order):

```bash
python src/data_processing/build_all.py --steps retrieval_corpus qd_pairs ce_training
```

**Run a single step directly:**

```bash
python src/data_processing/build_qd_pairs.py
```

---

## Pipeline Overview

The five steps form a directed pipeline. Each step's output becomes the input to one or more downstream steps:

```
raw CSVs
    │
    ▼
[1] build_retrieval_corpus  ──► DHDA_filtered_AR.csv
                                DHDA_inscriptions_data.csv
                                (+ 4 other processed files)
         │                 │                    │
         ▼                 ▼                    ▼
[2] build_qa_pairs   [3] build_qd_pairs   [5] build_classification_data
                              │
                              ▼
                    [4] build_ce_training_data
```

| Step | Script | Depends on | Produces |
|------|--------|------------|----------|
| —    | `data_utils` | — | Shared utilities (no output) |
| 1    | `build_retrieval_corpus` | Raw scraped CSVs | Processed corpus files |
| 2    | `build_qa_pairs` | `DHDA_filtered_AR.csv` | QA JSON + evaluation CSVs |
| 3    | `build_qd_pairs` | `DHDA_filtered_AR.csv` | `QD_full_data.csv`, `QD_evaluation_data.csv` |
| 4    | `build_ce_training_data` | `QD_full_data.csv` | `QD_contrastive_data.csv` |
| 5    | `build_classification_data` | `DHDA_filtered_AR.csv`, `DHDA_inscriptions_data.csv` | `classification_data.csv` |

---

## Script Reference

### `data_utils.py` — Shared utilities

Contains lightweight helper functions imported by multiple pipeline scripts. It is not executable directly. Centralising these functions ensures consistent behaviour across all steps and eliminates duplicated maintenance.

| Function | Used by | Description |
|----------|---------|-------------|
| `strip_tashkeel(text)` | `build_retrieval_corpus`, `build_qd_pairs` | Removes all Arabic diacritical marks from a string using a compiled regex. |
| `nonempty(value)` | `build_qa_pairs`, `build_qd_pairs`, `build_classification_data` | Returns `True` when a value is non-null, non-blank, and not the string `'nan'`. |
| `word_type(text)` | `build_qa_pairs`, `build_qd_pairs`, `build_classification_data` | Returns `'عبارة'` for multi-word text, `'كلمة'` for single-word text. |
| `parse_meaning(meaning)` | `build_qa_pairs`, `build_qd_pairs`, `build_classification_data` | Splits a meaning string at the first `':'` or `'؛'` into `(meaning_head, meaning_2)`. |
| `normalize_author(author)` | `build_qa_pairs`, `build_qd_pairs` | Replaces generic hadith attribution labels with the Prophet's full name. |

---

### `build_all.py` — Orchestrator

Runs any combination of the five pipeline steps in enforced dependency order.

**Usage:**

```bash
# Run all steps
python src/data_processing/build_all.py

# Run a subset
python src/data_processing/build_all.py --steps retrieval_corpus qd_pairs ce_training

# Override paths and tuning arguments
python src/data_processing/build_all.py \
    --raw-dir        data/retrieval_corpus/raw \
    --corpus         data/retrieval_corpus/processed/DHDA_filtered_AR.csv \
    --ce-n-source    8000 \
    --clf-n-per-type 1500
```

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--steps` | all five | Space-separated step names to run. Choices: `retrieval_corpus`, `qa_pairs`, `qd_pairs`, `ce_training`, `classification`. |
| `--corpus` | `data/retrieval_corpus/processed/DHDA_filtered_AR.csv` | Path to the processed Arabic-column corpus. |
| `--inscriptions` | `data/retrieval_corpus/processed/DHDA_inscriptions_data.csv` | Path to the processed inscriptions data. |
| `--random-state` | `42` | Global random seed forwarded to all steps. |
| `--raw-dir` | `data/retrieval_corpus/raw` | Root folder for raw scraped CSVs (Step 1). |
| `--corpus-output-dir` | `data/retrieval_corpus/processed` | Destination for processed corpus files (Step 1). |
| `--qa-output-dir` | `data/questions_data/QA_data` | Root output folder for QA files (Step 2). |
| `--qd-output` | `data/questions_data/QD_data/QD_full_data.csv` | Output path for `QD_full_data.csv` (Step 3). |
| `--qd-eval-output` | `data/questions_data/QD_data/QD_evaluation_data.csv` | Output path for `QD_evaluation_data.csv` (Step 3). |
| `--qd-n-sample` | `1002` | Number of rows in the QD evaluation sample (Step 3). |
| `--ce-output` | `data/questions_data/QD_data/QD_contrastive_data.csv` | Output path for `QD_contrastive_data.csv` (Step 4). |
| `--ce-n-source` | `5000` | QD rows to sample before building contrastive pairs; output is approximately `2 × n-source` rows (Step 4). |
| `--ce-num-negatives` | `1` | Negative document samples per query-document pair (Step 4). |
| `--clf-output` | `data/questions_data/classification_data/classification_data.csv` | Output path for `classification_data.csv` (Step 5). |
| `--clf-n-per-type` | `1000` | Number of labelled questions to sample per question type (Step 5). |

---

### `build_retrieval_corpus.py` — Step 1

**Class:** `RetrievalDataGenerator`

Combines per-root raw CSV files from three data categories (`lexical_details`, `etymological`, `inscriptions`) and produces the processed corpus files used by all downstream steps.

**Usage:**

```bash
python src/data_processing/build_retrieval_corpus.py \
    --raw-dir    data/retrieval_corpus/raw \
    --output-dir data/retrieval_corpus/processed
```

**What it does:**

1. **Combine raw files** — concatenates all per-root CSVs in each of the three sub-folders into a single file each.
2. **Build English-column file** — selects the 20 canonical columns by their original (English) names.
3. **Build Arabic-column file** — renames the 20 columns to their Arabic display names; also normalises bare morphology values (`متعد`, `لازم`) by prepending `فعل`.
4. **Build text-to-embed file** — concatenates `الكلمة + الجذر + المعنى + الشاهد` for each row and strips all tashkeel diacritics.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--raw-dir` | `data/retrieval_corpus/raw` | Folder containing `lexical_details/`, `etymological/`, `inscriptions/` sub-folders. |
| `--output-dir` | `data/retrieval_corpus/processed` | Destination for all output files. |

---

### `build_qa_pairs.py` — Step 2

**Class:** `QADataGenerator`

Generates structured question-answer pairs for model evaluation and fine-tuning from the processed lexical corpus.

**Usage:**

```bash
python src/data_processing/build_qa_pairs.py \
    --corpus     data/retrieval_corpus/processed/DHDA_filtered_AR.csv \
    --output-dir data/questions_data/QA_data
```

**What it does:**

1. **Build all QA** — generates QA pairs for every lemma across the full corpus, saves one JSON per root under `QA_roots/all_data/` and a combined `full_QA.json`.
2. **Build Islamic QA** — filters to entries with a Quran surah (`السورة`) or hadith number (`رقم الحديث`) and repeats step 1 for that subset.
3. **Sample full** — proportional stratified sample of 2 000 rows across all six question types → `islamic_full_testing.csv`.
4. **Sample meanings** — 500 `contextual_meaning` + 500 `basic_meaning` rows → `islamic_meaning_testing.csv`.

**Question types generated:**

| Type | Required fields |
|------|----------------|
| `author_of_citation` | `القائل`, `الشاهد` |
| `contextual_meaning` | `المعنى`, `الشاهد` |
| `source_of_citation` | `المصدر`, `الشاهد` |
| `historical_date` | `تاريخ استعمال الشاهد`, `الشاهد` |
| `basic_meaning` | `المعنى` |
| `part_of_speech` | `الاشتقاق الصرفي للكلمة` |

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | `data/retrieval_corpus/processed/DHDA_filtered_AR.csv` | Path to `DHDA_filtered_AR.csv`. |
| `--output-dir` | `data/questions_data/QA_data` | Root folder for all QA output. |

---

### `build_qd_pairs.py` — Step 3

**Class:** `QDDataGenerator`

Generates query-document pairs for bi-encoder retrieval training. Each corpus row maps to one or more questions; a `query_id` is shared across all rows that produce the same question text (enabling n-to-1 retrieval supervision).

**Usage:**

```bash
python src/data_processing/build_qd_pairs.py \
    --corpus      data/retrieval_corpus/processed/DHDA_filtered_AR.csv \
    --output      data/questions_data/QD_data/QD_full_data.csv \
    --eval-output data/questions_data/QD_data/QD_evaluation_data.csv \
    --n-sample    1002
```

**What it does:**

1. Generates one record per applicable question type for each corpus row.
2. Assigns a unique sequential `query_id` to each distinct question text (so identical questions across rows share one `query_id`).
3. Saves the full dataset to `QD_full_data.csv`.
4. Produces a proportionally stratified evaluation sample in `QD_evaluation_data.csv`.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | `data/retrieval_corpus/processed/DHDA_filtered_AR.csv` | Path to `DHDA_filtered_AR.csv`. |
| `--output` | `data/questions_data/QD_data/QD_full_data.csv` | Output path for the full QD dataset. |
| `--eval-output` | `data/questions_data/QD_data/QD_evaluation_data.csv` | Output path for the evaluation sample. |
| `--n-sample` | `1002` | Number of rows in the evaluation sample. |

---

### `build_ce_training_data.py` — Step 4

**Class:** `CEContrastiveDataGenerator`

Builds positive/negative contrastive pairs for cross-encoder fine-tuning from `QD_full_data.csv`.

**Usage:**

```bash
python src/data_processing/build_ce_training_data.py \
    --qd-path       data/questions_data/QD_data/QD_full_data.csv \
    --output        data/questions_data/QD_data/QD_contrastive_data.csv \
    --n-source      5000 \
    --num-negatives 1
```

**What it does:**

1. Draws a proportionally stratified sample of `n-source` rows from `QD_full_data.csv`.
2. For each sampled row, creates:
   - One **positive** pair (query ↔ its actual document text, `label = 1.0`).
   - `num-negatives` **safe-negative** pairs (query ↔ randomly chosen documents that are not a known positive for that query, `label = 0.0`).
3. Saves to `QD_contrastive_data.csv`.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--qd-path` | `data/questions_data/QD_data/QD_full_data.csv` | Source QD dataset. |
| `--output` | `data/questions_data/QD_data/QD_contrastive_data.csv` | Output path. |
| `--n-source` | `5000` | QD rows to sample; output is approximately `n-source × (1 + num-negatives)` rows. |
| `--num-negatives` | `1` | Negative documents per query-document pair. |
| `--random-state` | `42` | Random seed. |

---

### `build_classification_data.py` — Step 5

**Class:** `ClassificationDataGenerator`

Generates labelled question samples for training and evaluating a question-type classifier. Questions are generated from three levels of the corpus (entry-level, root-level, inscription-level) with **all observed surface variants** per question type to prevent the classifier from overfitting to a single phrasing.

**Usage:**

```bash
python src/data_processing/build_classification_data.py \
    --corpus       data/retrieval_corpus/processed/DHDA_filtered_AR.csv \
    --inscriptions data/retrieval_corpus/processed/DHDA_inscriptions_data.csv \
    --output       data/questions_data/classification_data/classification_data.csv \
    --n-per-type   1000
```

**What it does:**

1. **Entry-level** — generates multiple surface-variant questions per corpus row for: `basic_meaning`, `contextual_meaning`, `part_of_speech`, `author_of_citation`, `historical_date`, `source_of_citation`.
2. **Root-level** — generates one question per unique root for: `etymology`, `first_quranic_usage`, `first_usage`, `list_derivations`, `terminological_usage`.
3. **Inscription-level** — generates one question per unique root present in `DHDA_inscriptions_data.csv` for: `inscription`.
4. Deduplicates all `(type, question)` pairs, then samples up to `n-per-type` rows per type.

**12 question types:**

| Type | Level | Phrasing variants |
|------|-------|------------------|
| `basic_meaning` | entry | 6 (3 question prefixes × كلمة/عبارة) |
| `contextual_meaning` | entry | 3 |
| `part_of_speech` | entry | 1 |
| `author_of_citation` | entry | 6 |
| `historical_date` | entry | 2 (كلمة/عبارة) |
| `source_of_citation` | entry | 2 (كلمة/عبارة) |
| `etymology` | root | 1 |
| `first_quranic_usage` | root | 1 |
| `first_usage` | root | 1 |
| `list_derivations` | root | 1 |
| `terminological_usage` | root | 1 |
| `inscription` | inscription | 1 |

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | `data/retrieval_corpus/processed/DHDA_filtered_AR.csv` | Path to `DHDA_filtered_AR.csv`. |
| `--inscriptions` | `data/retrieval_corpus/processed/DHDA_inscriptions_data.csv` | Path to `DHDA_inscriptions_data.csv`. |
| `--output` | `data/questions_data/classification_data/classification_data.csv` | Output path. |
| `--n-per-type` | `1000` | Maximum questions sampled per type. |
| `--random-state` | `42` | Random seed. |

---

## Output File Reference

### Retrieval corpus files (`data/retrieval_corpus/processed/`)

| File | Columns | Description |
|------|---------|-------------|
| `DHDA_filtered_EN.csv` | 20 (English names) | Canonical 20-column subset of lexical details, original column names. |
| `DHDA_filtered_AR.csv` | 20 (Arabic names) | Same 20 columns with Arabic display names. **Primary corpus for all downstream steps.** |
| `DHDA_text_to_embed.csv` | `text` | Tashkeel-stripped concatenation of word + root + meaning + citation. One row per corpus entry. Used to build retrieval embeddings. |
| `DHDA_lexical_data.csv` | all raw cols | All lexical-details raw CSVs combined. |
| `DHDA_etymological_data.csv` | all raw cols | All etymological raw CSVs combined. |
| `DHDA_inscriptions_data.csv` | all raw cols | All inscriptions raw CSVs combined. |

**Arabic column names in `DHDA_filtered_AR.csv`:**

| Column | Description |
|--------|-------------|
| `ID` | Unique entry identifier |
| `rootId` | Root identifier (shared across all entries of the same root) |
| `lemmaId` | Lemma identifier |
| `الجذر` | Arabic root string |
| `الكلمة بدون تشكيل` | Lemma without diacritics |
| `الكلمة` | Lemma with tashkeel |
| `الاشتقاق الصرفي للكلمة` | Morphological derivation / part of speech |
| `العبارة أو اللفظ المركب` | Multi-word expression tag |
| `مقدمة الشاهد` | Citation preamble (head citation) |
| `الشاهد` | Textual citation |
| `المعنى` | Meaning definition |
| `القائل` | Author / speaker of the citation |
| `تاريخ استعمال الشاهد` | Date of the citation's usage |
| `الحقل الاصطلاحي` | Semantic / terminological field |
| `المصدر` | Source reference |
| `رقم الصفحة` | Page number |
| `السورة` | Quran surah (if applicable) |
| `رقم الآية` | Quran verse number |
| `رقم الحديث` | Hadith number (if applicable) |
| `تعليقات إضافية` | Additional remarks |

---

### QA files (`data/questions_data/QA_data/`)

| File | Format | Columns | Description |
|------|--------|---------|-------------|
| `QA_roots/all_data/qa_root_{rootId}.json` | JSON array | `Source, lemma, type, question, answer` | All QA pairs for one root. |
| `QA_roots/islamic_data/qa_root_{rootId}.json` | JSON array | same | Islamic-filtered QA pairs for one root. |
| `QA_combined/full_QA.json` | JSON array | same | All QA pairs across the entire corpus. |
| `QA_combined/islamic_QA_comprehensive.json` | JSON array | same | All QA pairs for Islamic entries. |
| `QA_combined/islamic_QA_meaning.json` | JSON array | same | Meaning-type QA pairs for Islamic entries only. |
| `evaluation_data/islamic_full_testing.csv` | CSV | `lemma, type, question, answer` | Stratified 2 000-row sample across all 6 question types. |
| `evaluation_data/islamic_meaning_testing.csv` | CSV | `lemma, type, question, answer` | 500 `contextual_meaning` + 500 `basic_meaning` rows. |

---

### QD files (`data/questions_data/QD_data/`)

| File | Columns | Description |
|------|---------|-------------|
| `QD_full_data.csv` | `type, question, answer, doc_id, query_id, text` | Complete query-document pair dataset. `query_id` is a sequential integer shared across all rows that produce the same question text. `doc_id` is the corpus entry `ID`. `text` is the tashkeel-stripped concatenation of the four text columns. |
| `QD_evaluation_data.csv` | same | Proportionally stratified sample of ~1 002 rows for retrieval evaluation. |
| `QD_contrastive_data.csv` | `question, text, label` | Contrastive pairs for cross-encoder fine-tuning. `label = 1.0` for positives; `label = 0.0` for safe negatives. |

---

### Classification file (`data/questions_data/classification_data/`)

| File | Columns | Description |
|------|---------|-------------|
| `classification_data.csv` | `type, question` | Labelled questions for question-type classification. Up to 1 000 rows per type; 12 types total. Shuffled before saving. |

---

## Shared Conventions

- **Encoding** — all output CSV files use UTF-8 with BOM (`utf-8-sig`) for compatibility with Arabic text in downstream tools.
- **Reproducibility** — all sampling operations accept a `--random-state` argument (default `42`). Fixing this value guarantees identical splits across runs.
- **Shared utilities** — common helpers (`strip_tashkeel`, `nonempty`, `word_type`, `parse_meaning`, `normalize_author`) are defined once in `data_utils.py` and imported by all scripts that need them.
- **Tashkeel stripping** — Arabic diacritics (tashkeel) are removed via `data_utils.strip_tashkeel`, which uses a compiled regex covering Unicode ranges U+064B–U+065F, U+0610–U+061A, U+06D6–U+06E4, U+06E7–U+06E8, and U+06EA–U+06ED.
- **Word type** (`كلمة` / `عبارة`) — question templates that contain a word-type placeholder use `عبارة` for multi-word `meaning_head` values (containing a space) and `كلمة` for single-word values.
- **Meaning splitting** — `parse_meaning` splits a meaning string at the first `:` or `؛` separator to obtain `meaning_head` (the term) and `meaning_2` (the definition body). If no separator is found, both parts are the full string.
- **Path resolution** — all default paths are relative to the repository root. Scripts must be invoked from `Doha-Dictionary-RAG/` or paths must be overridden explicitly.
