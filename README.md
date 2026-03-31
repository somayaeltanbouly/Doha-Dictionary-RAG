# Doha Dictionary RAG

A modular Retrieval-Augmented Generation (RAG) pipeline for question answering over the *Doha Historical Dictionary of Arabic* (DHDA / معجم الدوحة التاريخي للغة العربية). The system supports multiple retrieval strategies, three LLM backends, and intent-aware prompting calibrated per question type.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Layout](#repository-layout)
4. [Data Description](#data-description)
5. [Environment Setup](#environment-setup)
6. [Full Pipeline Walkthrough](#full-pipeline-walkthrough)
7. [Configuration](#configuration)
8. [Pipeline Reference](#pipeline-reference)
9. [Supported Intent Types](#supported-intent-types)
10. [Supported Model Backends](#supported-model-backends)
11. [Retrieval Strategies](#retrieval-strategies)
12. [Evaluation Methodology](#evaluation-methodology)
13. [Reproducibility Notes](#reproducibility-notes)

---

## Overview

The Doha Historical Dictionary is an Arabic linguistic resource that documents the historical usage of words through citations from classical texts, including the Quran, Hadith, poetry, and prose. This project builds a question-answering system on top of that resource, enabling users to ask natural-language Arabic questions and receive accurate, document-grounded answers.

### What the system does

1. **Understands intent** — a trained classifier (Random Forest + character-level TF-IDF) identifies the question type (meaning, source, author, date, morphology, etymology, inscription, or other).
2. **Retrieves relevant documents** — a hybrid retriever combines BM25 keyword search, dense FAISS vector search, and cross-encoder reranking, with intent-aware query cleaning.
3. **Builds intent-aware prompts** — few-shot or zero-shot prompts are assembled dynamically based on intent and mode, exposing only the document columns relevant to each question type.
4. **Generates answers** — via Fanar API, Google Gemini API, or ALLaM (local HuggingFace inference).
5. **Evaluates quality** — Gemini-as-judge scores answers on factual correctness and completeness using a 5-point rubric.

---

## Architecture

```
Arabic Question
      │
      ▼
┌─────────────────────┐
│  Intent Classifier  │  Random Forest + TF-IDF (character n-grams 3–5)
│  (9 question types) │  → intent label (e.g. basic_meaning, source_of_citation)
└─────────┬───────────┘
          │  intent + cleaned query
          ▼
┌─────────────────────────────────────────────────────────┐
│                  HybridRetriever                        │
│                                                         │
│  BM25 (sparse) ──┐                                      │
│                  ├─► Reciprocal Rank Fusion (RRF)       │
│  FAISS (dense) ──┘         │                            │
│                            ▼                            │
│               Cross-Encoder Reranker                    │
│               (fine-tuned BGE)                          │
└─────────────────────────────┬───────────────────────────┘
                              │  top-K documents
                              │  (intent-filtered columns)
                              ▼
              ┌───────────────────────────┐
              │      PromptBuilder        │
              │  intent × mode (fs / zs)  │
              └──────────────┬────────────┘
                             │  structured Arabic prompt
                             ▼
              ┌──────────────────────────┐
              │     LLM Backend          │
              │  Fanar / Gemini / ALLaM  │
              └──────────────┬───────────┘
                             │
                             ▼
                     Generated Answer
                             │
                             ▼
              ┌──────────────────────────┐
              │    Gemini-as-Judge       │
              │  score: 0/25/50/75/100%  │
              └──────────────────────────┘
```

---

## Repository Layout

```
Doha-Dictionary-RAG/
├── run.py                          ← Single CLI entry point for all pipelines
├── requirements.txt                ← Python dependencies (pinned minimum versions)
├── config.yaml                     ← Central configuration — edit to change defaults
├── .gitignore
│
├── src/
│   ├── data_processing/            Build and process all training/retrieval data
│   │   ├── build_retrieval_corpus.py   Step 1: raw → processed corpus
│   │   ├── build_qa_pairs.py           Step 2: QA pairs for evaluation
│   │   ├── build_qd_pairs.py           Step 3: QD pairs for retrieval training
│   │   ├── build_ce_training_data.py   Step 4: contrastive pairs for cross-encoder
│   │   ├── build_classification_data.py Step 5: labelled data for intent classifier
│   │   ├── data_utils.py               Shared helpers (tashkeel, normalization)
│   │   └── build_all.py                Orchestrator — runs all five steps in order
│   │
│   ├── models_training/            Train intent classifier and cross-encoder
│   │   ├── train_classifier.py
│   │   └── finetune_cross_encoder.py
│   │
│   ├── retrieval/                  Build FAISS index and evaluate retrieval
│   │   ├── build_index.py
│   │   ├── retriever.py            HybridRetriever (BM25 + FAISS + reranker)
│   │   ├── evaluate_retrieval.py
│   │   ├── retrieval_pipeline.py   Standalone script: build-index + evaluate in one run
│   │   │                           (not a run.py subcommand — run directly with python)
│   │   └── metrics.py              Recall@K, MRR, MAP
│   │
│   ├── rag/                        Model backends, prompt builder, and pipeline
│   │   ├── model_loader.py         ModelLoader factory (Fanar / Gemini / HuggingFace)
│   │   ├── prompt_builder.py       Intent-aware, mode-aware prompt assembly
│   │   └── rag_pipeline.py         RAGPipeline — orchestrates retrieval → generation
│   │
│   └── evaluation/                 LLM-as-judge evaluation
│       ├── judge.py                Gemini-as-judge scoring
│       └── summarize_scores.py     Score statistics and two-file comparison
│
├── data/                           Data files (see Data Description below)
│   ├── retrieval_corpus/
│   │   ├── raw/                    Raw dictionary source files  ← NOT tracked in git
│   │   ├── processed/              Cleaned corpus CSVs          ← generated
│   │   └── vector_database/        FAISS index + embeddings     ← generated
│   └── questions_data/
│       ├── QA_data/                QA pairs per intent type     ← generated
│       ├── QD_data/                Query-Document pairs         ← generated
│       └── classification_data/    Intent classifier training   ← generated
│
├── models/                         Trained model artefacts      ← NOT tracked in git
│   ├── nomic/                      Local SentenceTransformer embedding model
│   ├── RF_intent_classifier.joblib
│   └── finetuned_CE_bge/           Fine-tuned cross-encoder reranker
│
├── output/                         Generation pipeline outputs  ← NOT tracked in git
└── judging/                        LLM-judge scoring outputs    ← NOT tracked in git
```

---

## Data Description

### What is the DHDA?

The **Doha Historical Dictionary of Arabic** (المعجم التاريخي للغة العربية) is a large-scale lexicographic project documenting the historical usage of Arabic words from pre-Islamic times to the modern era. Each entry records a word's root, morphological form, meaning, a citation (شاهد) from a classical text, the citation's author (القائل), source, and date of use.

### Abbreviations

| Abbreviation | Meaning | Description |
|---|---|---|
| **QA** | Question-Answer | Pairs used for model evaluation; each has a reference answer |
| **QD** | Query-Document | Pairs used for retrieval training; each links a question to the relevant corpus entry |
| **CE** | Cross-Encoder | The reranking model fine-tuned on contrastive QD pairs |
| **DHDA** | Doha Historical Dictionary of Arabic | The primary data source |
| **BM25** | Best Match 25 | Classic TF-IDF-variant keyword retrieval algorithm |
| **RRF** | Reciprocal Rank Fusion | Score fusion method for combining BM25 and dense rankings |

### Data folder contents

| Folder | Contents | How it is produced |
|---|---|---|
| `data/retrieval_corpus/raw/` | Per-root CSVs scraped from the dictionary (lexical, etymological, inscriptions) | External — must be sourced separately; **not tracked in git** |
| `data/retrieval_corpus/processed/` | Cleaned 20-column corpus (`DHDA_filtered_AR.csv`), text-to-embed CSV, and per-type data files | `python run.py build-corpus` |
| `data/retrieval_corpus/vector_database/` | Dense embeddings (`.npy`) and FAISS index (`.index`) | `python run.py build-index` |
| `data/questions_data/QA_data/` | QA JSON files (per-root and combined) and evaluation CSVs | `python run.py build-qa` |
| `data/questions_data/QD_data/` | Full QD pairs, evaluation split, and contrastive pairs for CE training | `python run.py build-qd` then `python run.py build-ce-data` |
| `data/questions_data/classification_data/` | Labelled question samples for classifier training | `python run.py build-clf-data` |

### What is tracked in git

Only source code and configuration are tracked. Datasets, trained models, generated outputs, and the FAISS index are excluded via `.gitignore`. To reproduce the full pipeline from scratch, you need the raw dictionary source files and the embedding model (see Environment Setup).

---

## Environment Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Not required (CPU inference possible) | CUDA GPU with ≥ 16 GB VRAM for ALLaM (HuggingFace backend) |
| RAM | 8 GB | 32 GB (for large corpus + BM25 index in memory) |
| Storage | 10 GB | 50 GB (raw data, embeddings, model artefacts) |

The **Fanar** and **Gemini** backends are API-based and require no local GPU. The **ALLaM (HuggingFace)** backend loads a 7B-parameter model locally; use `--load-in-4bit` to reduce GPU memory by approximately 50%.

### API keys

```bash
export FANAR_API_KEY="..."    # required for --model fanar
export GEMINI_API_KEY="..."   # required for --model gemini and for the judge
# no key needed for --model hf (local inference)
```

### Embedding model

Download or link the `nomic-ai/nomic-embed-text-v2-moe` SentenceTransformer model to `models/nomic/`:

```bash
python -c "from sentence_transformers import SentenceTransformer; \
           SentenceTransformer('nomic-ai/nomic-embed-text-v2-moe') \
           .save('models/nomic')"
```

---

## Full Pipeline Walkthrough

Run all stages in order to go from raw data to evaluated RAG results. Each stage can also be run independently; see [Pipeline Reference](#pipeline-reference) for options.

```bash
# ── Stage 1: Data preparation ─────────────────────────────── #

# 1a. Build retrieval corpus from raw dictionary files
python run.py build-corpus

# 1b. Generate QA pairs for model evaluation
python run.py build-qa

# 1c. Generate Query-Document pairs for retrieval training
python run.py build-qd

# 1d. Build contrastive CE training data from QD pairs
python run.py build-ce-data

# 1e. Build labelled data for intent classifier
python run.py build-clf-data

# ── Stage 2: Model training ───────────────────────────────── #

# 2a. Train the intent classifier
python run.py train-classifier

# 2b. Fine-tune the cross-encoder reranker
python run.py finetune-reranker

# ── Stage 3: Retrieval ────────────────────────────────────── #

# 3a. Embed the corpus and build the FAISS index
python run.py build-index

# 3b. Evaluate retrieval quality (Recall@K, MRR, MAP)
python run.py eval-retrieval --method hybrid

# ── Stage 4: RAG generation ───────────────────────────────── #

# 4a. Few-shot RAG with Gemini (hybrid retrieval — recommended)
python run.py generate --model gemini --mode fs

# 4b. Zero-shot RAG with Fanar
python run.py generate --model fanar --mode zs

# 4c. Baseline (no retrieval) — useful as a lower bound
python run.py generate --model gemini --mode baseline

# ── Stage 5: Evaluation ───────────────────────────────────── #

# 5a. Score answers with Gemini-as-judge
python run.py judge \
    --input  output/gemini_fs_results.csv \
    --output judging/gemini_fs_judged.csv

# 5b. Print score statistics
python run.py summarize-scores --input judging/gemini_fs_judged.csv

# 5c. Compare two systems side-by-side
python run.py summarize-scores \
    --input  judging/fanar_zs_judged.csv \
    --input2 judging/gemini_fs_judged.csv
```

You can also run all data-preparation steps in one command:

```bash
python src/data_processing/build_all.py
```

---

## Configuration

All pipeline defaults are centralised in `config.yaml` at the repository root. This file is loaded automatically by `run.py` and values are used as default arguments for all commands.

### Editing Configuration

To change parameters globally, edit `config.yaml`:

```yaml
retrieval:
  top_k: 5        # change default number of retrieved documents

generation:
  default_mode: zs  # change default prompting mode
```

Command-line arguments always override config file values. See `config.yaml` for the full list of documented parameters grouped by pipeline stage.

### Using Config in Custom Scripts

If you're writing custom scripts that need to access configuration values, use the config loader:

```python
from src.config_loader import get_config, get_config_value

# Load configuration once
config = get_config()

# Access nested values using dot notation
top_k = get_config_value(config, "retrieval.top_k", default=10)
corpus_path = get_config_value(config, "data.corpus_ar")
random_seed = get_config_value(config, "random_seed", default=42)

# Provide fallback for missing values
custom_value = get_config_value(config, "nonexistent.key", default="fallback")
```

The config is loaded once and cached, so subsequent calls to `get_config()` are fast.

---

## Pipeline Reference

### Data Preparation

```bash
# Build retrieval corpus from raw dictionary files
python run.py build-corpus

# Generate QA pairs (one per intent type)
python run.py build-qa

# Generate Query-Document pairs for retrieval training
python run.py build-qd

# Build contrastive data for cross-encoder fine-tuning
python run.py build-ce-data

# Build labelled data for intent classifier
python run.py build-clf-data
```

### Model Training

```bash
# Train the intent classifier
python run.py train-classifier

# Fine-tune the cross-encoder reranker
python run.py finetune-reranker
```

### Retrieval

```bash
# Embed the corpus and build a FAISS index
python run.py build-index

# Evaluate retrieval quality (Recall@K, MRR, MAP)
python run.py eval-retrieval --method hybrid
```

### RAG Generation

```bash
# Few-shot RAG with Gemini (hybrid retrieval)
python run.py generate --model gemini --mode fs

# Zero-shot RAG with Fanar using BM25
python run.py generate --model fanar --mode zs --method bm25

# Baseline (no retrieval) with ALLaM
python run.py generate --model hf --mode baseline

# Resume an interrupted run from row 300
python run.py generate --model gemini --mode fs --start-from 300
```

API keys can be passed as flags or environment variables:

```bash
export FANAR_API_KEY="..."
export GEMINI_API_KEY="..."
# or
python run.py generate --model fanar --mode fs --fanar-api-key "..."
```

### Evaluation

```bash
# Score answers with Gemini-as-judge
python run.py judge \
    --input  output/gemini_fs_results.csv \
    --output judging/gemini_fs_judged.csv

# Print score statistics for a single judging file
python run.py summarize-scores --input judging/gemini_fs_judged.csv

# Compare two judging files side-by-side
python run.py summarize-scores \
    --input  judging/fanar_zs_judged.csv \
    --input2 judging/gemini_fs_judged.csv
```

---

## Supported Intent Types

| Intent label | Question type | Example |
|---|---|---|
| `basic_meaning` | What does a word or phrase mean? | ما معنى كلمة الجفاء؟ |
| `contextual_meaning` | What does a word mean in a specific citation? | ما معنى الشاهد "..." في السياق؟ |
| `source_of_citation` | What is the source of a citation? | ما مصدر الشاهد "..."؟ |
| `author_of_citation` | Who is the author (القائل) of a citation? | من قائل الشاهد "..."؟ |
| `historical_date` | What is the documented date of usage? | ما تاريخ استعمال الكلمة؟ |
| `part_of_speech` | What is the morphological derivation (الاشتقاق الصرفي)? | ما الاشتقاق الصرفي لكلمة ...؟ |
| `etymology` | What are the historical-language roots of a word? | ما أصل الكلمة في اللغات القديمة؟ |
| `inscription` | What are the inscription/carving details for a root? | ما النقوش المتعلقة بجذر ...؟ |
| `other` | General questions across all document fields | — |

---

## Supported Model Backends

| Flag | Backend | Default model | Required env var |
|---|---|---|---|
| `--model fanar` | Fanar API | `Fanar-C-1-8.7B` | `FANAR_API_KEY` |
| `--model gemini` | Google Gemini | `gemini-2.5-pro` | `GEMINI_API_KEY` |
| `--model hf` | HuggingFace (ALLaM) | `ALLaM-AI/ALLaM-7B-Instruct-preview` | — |

The HuggingFace backend loads in fp16 by default. Pass `--load-in-4bit` to use 4-bit quantization instead (lower VRAM, slight quality loss).

---

## Retrieval Strategies

| Strategy | Description |
|---|---|
| `bm25` | Sparse keyword search using BM25. Fast, no GPU required. |
| `dense` | Dense FAISS nearest-neighbour search using `nomic-embed-text-v2-moe` embeddings. |
| `hybrid` | Reciprocal Rank Fusion (BM25 + dense), then cross-encoder reranking. **Best quality — default.** |

Query cleaning is applied before all retrieval methods: intent-specific stop-words are removed and the query is de-diacriticised (tashkeel stripped).

---

## Evaluation Methodology

### LLM-as-judge scoring

Model answers are scored automatically by Gemini using a structured prompt that evaluates two dimensions:

- **Factual correctness** — no wrong dates, names, meanings, or fabricated sources
- **Completeness** — covers all key facts present in the reference answer

Scores follow a 5-point scale:

| Score | Meaning |
|---|---|
| `0%` | Wrong answer — any incorrect fact |
| `25%` | Partially correct — some correct elements, incorrect details, missing key parts |
| `50%` | Factually correct but incomplete — all stated facts are correct, key details missing |
| `75%` | Mostly correct and complete — correct facts, most details covered, minor precision gaps |
| `100%` | Fully correct and complete |

The judge runs with `temperature=0` for deterministic, reproducible scoring. The judge prompt is available in full at `src/evaluation/judge.py`.

### Data isolation

- The **evaluation QA set** (`islamic_full_testing.csv`, `islamic_meaning_testing.csv`) is sampled independently of the **retrieval training data** (QD pairs).
- The **QD evaluation split** is held out from cross-encoder fine-tuning (`val_ratio=0.10`, `test_ratio=0.10`).
- The **classifier test split** (`test_size=0.20`) is stratified and held out before any training.

---

## Reproducibility Notes

- All stochastic steps use `random_state=42` / `seed=42` (configurable in `config.yaml`).
- The cross-encoder fine-tuning additionally sets `random.seed`, `np.random.seed`, and `torch.manual_seed`.
- The LLM judge uses `temperature=0` for deterministic scoring.
- All required training scripts and their exact hyperparameters are documented in `config.yaml` and in each script's module-level docstring.
- Raw data, trained models, and large binary artefacts are excluded from version control (see `.gitignore`). A fresh run of `build_all.py` followed by the training scripts will regenerate everything except the raw source data.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{eltanbouly2026groundingarabicllmsdoha,
      title={Grounding Arabic LLMs in the Doha Historical Dictionary: Retrieval-Augmented Understanding of Quran and Hadith},
      author={Somaya Eltanbouly and Samer Rashwani},
      year={2026},
      eprint={2603.23972},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.23972},
}
```
