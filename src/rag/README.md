# src/rag/

Scripts for running the end-to-end Retrieval-Augmented Generation pipeline over the Doha Dictionary QA system.

---

## Files

| File | Description |
|---|---|
| `model_loader.py` | `ModelLoader` factory + backend classes (Fanar, Gemini, HuggingFace) |
| `prompt_builder.py` | `PromptBuilder` — intent-aware, mode-aware prompt assembly |
| `rag_pipeline.py` | `RAGPipeline` — orchestrates retrieval → prompting → generation; CLI entry point |

---

## Prerequisites

All scripts expect to be run from the **repo root**.  The FAISS index and
retrieval corpus must exist before running the pipeline; build them first with:

```bash
python src/retrieval/build_index.py
```

Set the required API key environment variable for the chosen model:

```bash
export FANAR_API_KEY="..."   # for --model fanar
export GEMINI_API_KEY="..."  # for --model gemini
# no key needed for --model hf (local inference)
```

---

## Quick start

```bash
# Few-shot RAG with Gemini (default mode)
python src/rag/rag_pipeline.py \
    --model gemini \
    --mode fs \
    --input data/questions_data/QA_data/evaluation_data/islamic_full_testing.csv \
    --output output/gemini_fs_results.csv

# Zero-shot RAG with Fanar
python src/rag/rag_pipeline.py \
    --model fanar \
    --mode zs \
    --input data/questions_data/QA_data/evaluation_data/islamic_full_testing.csv \
    --output output/fanar_zs_results.csv

# Baseline (no retrieval) with ALLaM
python src/rag/rag_pipeline.py \
    --model hf \
    --mode baseline \
    --input data/questions_data/QA_data/evaluation_data/islamic_full_testing.csv \
    --output output/allam_baseline_results.csv

# Resume an interrupted run from row 250
python src/rag/rag_pipeline.py \
    --model gemini \
    --mode fs \
    --input data/questions_data/QA_data/evaluation_data/islamic_full_testing.csv \
    --output output/gemini_fs_results.csv \
    --start-from 250
```

---

## 1. `model_loader.py` — Model backends and factory

### `ModelLoader.load(model_type, model_id=None, **kwargs)`

Returns a configured `GenerationBackend` instance.

| `model_type` | Backend class | Default model | Required env var |
|---|---|---|---|
| `"fanar"` | `FanarBackend` | `Fanar-C-1-8.7B` | `FANAR_API_KEY` |
| `"gemini"` | `GeminiBackend` | `gemini-2.5-pro` | `GEMINI_API_KEY` |
| `"hf"` | `HuggingFaceBackend` | `ALLaM-AI/ALLaM-7B-Instruct-preview` | — |

All backends share a single interface: `backend.generate(prompt: str) -> str`.

**HuggingFace backend precision options:**

| kwarg | Default | Description |
|---|---|---|
| `load_in_4bit` | `False` | Load with 4-bit BitsAndBytes quantization instead of fp16 |
| `max_new_tokens` | `512` | Maximum tokens to generate |

```python
from src.rag.model_loader import ModelLoader

# fp16 (default)
backend = ModelLoader.load("hf")

# 4-bit quantization
backend = ModelLoader.load("hf", load_in_4bit=True)

# custom model
backend = ModelLoader.load("hf", model_id="/path/to/local/model")
```

---

## 2. `prompt_builder.py` — Dynamic prompt assembly

### `PromptBuilder.build(intent, mode, query, documents)`

Constructs a complete prompt from three parts:

1. **base** — system role, document field descriptions, and answer instructions (always included)
2. **examples** — few-shot demonstration pairs (included only when `mode="fs"`)
3. **footer** — the final query + documents template

| Argument | Values |
|---|---|
| `intent` | Classifier output label (see table below) |
| `mode` | `"fs"` (few-shot) or `"zs"` (zero-shot) |
| `query` | The user's Arabic question |
| `documents` | JSON-serialised retrieved documents |

### Supported intents

| Classifier label | Prompt specialisation |
|---|---|
| `basic_meaning` | Exact-match meaning extraction with diacritic sensitivity |
| `contextual_meaning` | Meaning within a specific citation |
| `source_of_citation` | Source reference lookup |
| `author_of_citation` | القائل extraction |
| `historical_date` | تاريخ استعمال الشاهد lookup |
| `part_of_speech` | Morphological derivation (الاشتقاق الصرفي) |
| `etymology` | Root etymology across historical languages |
| `inscription` | Carving/inscription details |
| `other` | General-purpose fallback covering all document fields |

---

## 3. `rag_pipeline.py` — End-to-end pipeline

### Pipeline modes

| Mode | Retrieval | Examples in prompt |
|---|---|---|
| `fs` | Yes | Yes (few-shot) |
| `zs` | Yes | No (zero-shot) |
| `baseline` | No | No |

### Input CSV format

| Column | Description |
|---|---|
| `question` | Arabic question |
| `answer` | Gold reference answer |

### Output CSV format

| Column | Description |
|---|---|
| `question` | Original question |
| `answer` | Model-generated answer |
| `correct_answer` | Gold reference answer (from input) |

### CLI options

| Option | Default | Description |
|---|---|---|
| `--model` | *(required)* | `fanar`, `gemini`, or `hf` |
| `--model-id` | backend default | Override model ID or local path |
| `--mode` | `fs` | `fs`, `zs`, or `baseline` |
| `--method` | `hybrid` | Retrieval method: `bm25`, `dense`, or `hybrid` |
| `--input` | *(required)* | Path to input CSV |
| `--output` | *(required)* | Path for output CSV |
| `--start-from` | `0` | Row index to resume from after interruption |
| `--load-in-4bit` | off | Enable 4-bit quantization for the `hf` backend |
