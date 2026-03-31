# src/models_training/

Scripts for training the two models used by the DHDA RAG pipeline.

---

## Scripts

| Script | Purpose | Saved model |
|---|---|---|
| `finetune_cross_encoder.py` | Fine-tune a cross-encoder reranker | `models/finetuned_CE_bge/` |
| `train_classifier.py` | Train the Arabic question-type intent classifier | `models/RF_intent_classifier.joblib` |

---

## Cross-Encoder Fine-Tuning (`finetune_cross_encoder.py`)

Fine-tunes **`BAAI/bge-reranker-v2-m3`** on contrastive query-document pairs
using the `sentence-transformers` `CrossEncoder.fit()` API.

### Prerequisites

```bash
pip install "sentence-transformers>=3.0.0,<5.0.0" datasets scikit-learn
```

### Input data

`data/questions_data/QD_data/QD_contrastive_data.csv`

| Column | Description |
|---|---|
| `question` | Arabic query text |
| `text` | Candidate document passage |
| `label` | Relevance label — `1.0` (positive) or `0.0` (negative) |

Generate this file first with:
```bash
python src/data_processing/build_ce_training_data.py
```

### Usage

```bash
# defaults — reproduce models/finetuned_CE_bge
python src/models_training/finetune_cross_encoder.py

# with explicit settings
python src/models_training/finetune_cross_encoder.py \
    --data    data/questions_data/QD_data/QD_contrastive_data.csv \
    --output  models/finetuned_CE_bge \
    --epochs  3 --batch-size 8 --lr 5e-5 --seed 42
```

### Key hyperparameters (matching `models/finetuned_CE_bge`)

| Parameter | Default |
|---|---|
| `--model-name` | `BAAI/bge-reranker-v2-m3` |
| `--max-length` | `512` |
| `--epochs` | `3` |
| `--batch-size` | `8` |
| `--lr` | `5e-5` |
| `--warmup-steps` | `0` |
| `--fp16` | `True` |
| `--eval-strategy` | `steps` |
| `--optim` | `adamw_torch_fused` |
| `--lr-scheduler` | `linear` |
| `--seed` | `42` |
| `--val-ratio` | `0.1` |
| `--test-ratio` | `0.1` |

### Reported results (saved model)

| Metric | Value |
|---|---|
| Pearson  | **0.9923** |
| Spearman | **0.8658** |

---

## Intent Classifier (`train_classifier.py`)

Trains a scikit-learn `Pipeline(TfidfVectorizer + RandomForestClassifier)` on
Arabic question-type labels. The pipeline reproduces `models/RF_intent_classifier.joblib`.

### Prerequisites

```bash
pip install scikit-learn joblib pandas
```

### Input data

`data/questions_data/classification_data/classification_data.csv`

| Column | Description |
|---|---|
| `type` | Question-type label (e.g. `inscription`, `etymology`, …) |
| `question` | Arabic question text (tashkeel is stripped automatically) |

Generate this file first with:
```bash
python src/data_processing/build_classification_data.py
```

### Usage

```bash
# defaults — reproduce models/RF_intent_classifier.joblib
python src/models_training/train_classifier.py

# with explicit settings
python src/models_training/train_classifier.py \
    --data        data/questions_data/classification_data/classification_data.csv \
    --output      models/RF_intent_classifier.joblib \
    --n-estimators 200 --threshold 0.6
```

### Key hyperparameters (matching `models/RF_intent_classifier.joblib`)

| Parameter | Default | Description |
|---|---|---|
| `--n-estimators` | `200` | Trees in the Random Forest |
| `--ngram-range` | `3 5` | Character n-gram range for TF-IDF |
| `--max-features` | `30000` | TF-IDF vocabulary cap |
| `--test-size` | `0.20` | Fraction held out for evaluation |
| `--threshold` | `0.60` | Min confidence; below → predicts `"other"` |
| `--random-state` | `42` | Seed for split and forest |

---

## Running order

```bash
# 1. Build training data (if not already present)
python src/data_processing/build_ce_training_data.py
python src/data_processing/build_classification_data.py

# 2. Train models
python src/models_training/finetune_cross_encoder.py
python src/models_training/train_classifier.py
```
