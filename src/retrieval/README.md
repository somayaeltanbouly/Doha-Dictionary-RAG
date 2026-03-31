# src/retrieval/

Scripts for building the search index, running retrieval, and evaluating retrieval quality.

---

## Files

| File | Description |
|---|---|
| `build_index.py` | Embed the corpus and save a FAISS index to disk |
| `retriever.py` | `HybridRetriever` class — the main search component |
| `evaluate_retrieval.py` | Measure retrieval quality (Recall@K, MRR, MAP) on labelled data |
| `retrieval_pipeline.py` | Run indexing and/or evaluation in one command |
| `metrics.py` | Metric functions shared by `evaluate_retrieval.py` |

---

## Prerequisites

All scripts expect to be run from the **repo root**.  The corpus files must
exist before running any script here; generate them first with:

```bash
python src/data_processing/build_retrieval_corpus.py
```

---

## Quick start

```bash
# Step 1: embed the corpus and build the FAISS index
python src/retrieval/build_index.py

# Step 2: evaluate retrieval quality
python src/retrieval/evaluate_retrieval.py

# — or run both steps together —
python src/retrieval/retrieval_pipeline.py
```

---

## 1. `build_index.py` — Build the search index

Encodes every document in the corpus using a SentenceTransformer model and writes:

- `embeddings_<model-tag>.npy` — dense vector representations of all documents
- `faiss_<model-tag>.index` — the search index loaded by `HybridRetriever` at query time

The model tag is derived from the model name so that outputs from different models
never overwrite each other. For example, `nomic-ai/nomic-embed-text-v2-moe` produces
`embeddings_nomic-embed.npy` and `faiss_nomic-embed.index`.

If both output files already exist, the script exits without re-running.
Pass `--force` to rebuild from scratch.

### Input

`data/retrieval_corpus/processed/DHDA_text_to_embed.csv` — produced by `build_retrieval_corpus.py`

| Column | Description |
|---|---|
| `text` | Concatenation of root, word, meaning, and citation fields (tashkeel stripped) |

### Options

| Option | Default | Description |
|---|---|---|
| `--corpus` | `data/retrieval_corpus/processed/DHDA_text_to_embed.csv` | Input corpus CSV |
| `--model` | `nomic-ai/nomic-embed-text-v2-moe` | SentenceTransformer model (local path or HuggingFace ID) |
| `--index-type` | `Flat` | `Flat` (exact search), `IVFFlat` (approximate, faster for large corpora), or `HNSW` (approximate, graph-based) |
| `--batch-size` | `256` | Documents encoded per forward pass |
| `--emb-out` | `data/retrieval_corpus/vector_database/embeddings.npy` | Output path for embeddings (model tag inserted automatically) |
| `--idx-out` | `data/retrieval_corpus/vector_database/faiss.index` | Output path for index (model tag inserted automatically) |
| `--force` | off | Overwrite existing outputs |

### Examples

```bash
# default settings
python src/retrieval/build_index.py

# force rebuild
python src/retrieval/build_index.py --force

# use approximate index for faster queries on large corpora
python src/retrieval/build_index.py --index-type IVFFlat --nlist 100
```

---

## 2. `retriever.py` — HybridRetriever

The main search class. Used by both `evaluate_retrieval.py` and the RAG generation pipeline.

Supports three retrieval methods:

| Method | Description |
|---|---|
| `"bm25"` | Keyword search (BM25) |
| `"dense"` | Semantic vector search (FAISS) |
| `"hybrid"` | Combines BM25 and dense results using Reciprocal Rank Fusion, then reranks with the cross-encoder. **Recommended.** |

Before searching, `analyze_query()` classifies the question intent (e.g. `basic_meaning`,
`author_of_citation`) and strips query words that have no retrieval value (e.g. "ما معنى",
"من القائل"). The intent is also used by `format_documents()` to return only the columns
relevant to answering that type of question.

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `corpus_path` | `data/retrieval_corpus/processed/DHDA_filtered_AR.csv` | Structured corpus with all entry fields |
| `text_data_path` | `data/retrieval_corpus/processed/DHDA_text_to_embed.csv` | Plain-text corpus used for BM25 scoring |
| `embeddings_path` | `data/retrieval_corpus/vector_database/embeddings_nomic-embed.npy` | Pre-computed document embeddings |
| `index_path` | `data/retrieval_corpus/vector_database/faiss_nomic-embed.index` | FAISS index built by `build_index.py` |
| `classifier_path` | `models/RF_intent_classifier.joblib` | Intent classifier |
| `cross_encoder_path` | `models/finetuned_CE_bge` | Reranking model. Pass `None` to disable reranking. |
| `embedding_model_path` | `models/nomic` | Model used to encode queries at search time |
| `top_k` | `10` | Number of documents returned per query |
| `k_bm25` | `50` | Candidate pool size for BM25-only or dense-only search |
| `k_rrf` | `300` | Candidate pool per method before fusion (hybrid only) |
| `k_rerank` | `50` | Number of candidates passed to the cross-encoder |
| `rrf_k_const` | `60` | RRF smoothing constant |
| `bm25_weight` | `0.55` | Score weight for BM25 results in hybrid fusion |
| `dense_weight` | `0.45` | Score weight for dense results in hybrid fusion |

### Example

```python
from retriever import HybridRetriever

retriever = HybridRetriever()

# Classify intent and clean the query
query_info = retriever.analyze_query("ما معنى كلمة هَنَّأَهُ؟")
# → {"q1": "هناه", "intent": "basic_meaning"}

# Retrieve top-10 matching documents
docs_df, indices = retriever.retrieve(query_info["q1"], method="hybrid")

# Get a formatted context string for the LLM (intent-specific columns only)
context = retriever.format_documents(docs_df, query_info["intent"])
```

---

## 3. `evaluate_retrieval.py` — Evaluate retrieval quality

Runs a retrieval method on all queries in the evaluation set and computes three metrics:

| Metric | Description |
|---|---|
| **Recall@K** | On average, what fraction of the relevant documents appear in the top-K results |
| **MRR** | Mean Reciprocal Rank — how high the first relevant document ranks on average |
| **MAP** | Mean Average Precision — accounts for the rank of every relevant document |

### Input

`data/questions_data/QD_data/QD_evaluation_data.csv` — produced by `build_qd_pairs.py`

| Column | Description |
|---|---|
| `query_id` | Unique integer per question |
| `question` | Arabic question text |
| `doc_id` | Relevant document ID (multiple rows per `query_id` = multiple relevant documents) |
| `type` | Question intent class |

### Output

`retrieval_output/<method>_results.csv`  (when run via `python run.py eval-retrieval`)
`data/retrieval_corpus/evaluation_results/<method>_results.csv`  (when run directly)

| Column | Description |
|---|---|
| `qid` | Query ID |
| `docs` | Retrieved document IDs in rank order |
| `ground_truth` | Relevant document IDs for this query |

Metric summary is also printed to stdout.

### Options

| Option | Default | Description |
|---|---|---|
| `--method` | `hybrid` | Retrieval method: `bm25`, `dense`, or `hybrid` |
| `--eval-data` | `data/questions_data/QD_data/QD_evaluation_data.csv` | Labelled evaluation data |
| `--cross-encoder` | `models/finetuned_CE_bge` | Reranker model path. Pass `none` to disable. |
| `--top-k` | `10` | Number of documents to retrieve per query |
| `--output` | `data/retrieval_corpus/evaluation_results/<method>_results.csv` | Output CSV path |

### Examples

```bash
# hybrid retrieval with reranking (default)
python src/retrieval/evaluate_retrieval.py

# BM25 only, no reranking
python src/retrieval/evaluate_retrieval.py --method bm25 --cross-encoder none

# dense retrieval, retrieve top-5
python src/retrieval/evaluate_retrieval.py --method dense --top-k 5
```

---

## 4. `retrieval_pipeline.py` — Run indexing and evaluation together

Runs `build_index.py` and/or `evaluate_retrieval.py` in sequence from a single command.
Useful when starting fresh or when changing the embedding model and wanting to rebuild and re-evaluate in one step.

### Options

| Option | Default | Description |
|---|---|---|
| `--stages` | `index eval` | Which stages to run (`index`, `eval`, or both) |
| `--methods` | `hybrid` | Retrieval methods to evaluate (space-separated) |
| `--embedding-model` | `models/nomic` | Model used for both embedding and query encoding |
| `--force` | off | Rebuild index even if files already exist |
| `--emb-out-dir` | `data/retrieval_corpus/vector_database` | Output directory for embeddings and index files |
| `--eval-out-dir` | `data/retrieval_corpus/evaluation_results` | Output directory for evaluation result CSVs |

### Examples

```bash
# full pipeline: build index then evaluate hybrid
python src/retrieval/retrieval_pipeline.py

# only evaluate (index already built), compare all three methods
python src/retrieval/retrieval_pipeline.py --stages eval --methods bm25 dense hybrid

# force rebuild and evaluate BM25 without reranking
python src/retrieval/retrieval_pipeline.py \
    --stages index eval --force --methods bm25 --cross-encoder none
```

