"""
finetune_cross_encoder.py — Fine-tune a cross-encoder reranker
============================================================

Fine-tunes ``BAAI/bge-reranker-v2-m3`` on contrastive query-document pairs
using the same ``CrossEncoder.fit()`` approach used in ``ce_finetuning.ipynb``.
The only difference is that data is read from the pre-built CSV produced by
``build_ce_training_data.py`` rather than being generated inside the script.

The default hyperparameters reproduce the configuration used to train
``models/finetuned_CE_bge`` (Pearson 0.9923, Spearman 0.8658 on validation).

Pipeline
--------
1. Load ``QD_contrastive_data.csv`` (columns: ``question``, ``text``, ``label``).
2. Convert rows to ``InputExample`` objects.
3. Split into train / val / test (80 / 10 / 10), seeded for reproducibility.
4. Build ``CECorrelationEvaluator`` from the val split.
5. Fine-tune with ``model.fit()`` + ``DataLoader``.
6. Evaluate on the held-out test split and print metrics.

Defaults
--------
- Data            : ``data/questions_data/QD_data/QD_contrastive_data.csv``
- Output          : ``models/finetuned_CE_bge``
- Base model      : ``BAAI/bge-reranker-v2-m3``
- Max seq length  : 512
- Epochs          : 3
- Batch size      : 8
- Learning rate   : 5e-5
- Warmup steps    : 0
- AMP (fp16)      : True
- Eval steps      : 500
- Seed            : 42

Usage::

    # from the repo root
    python src/models_training/finetune_cross_encoder.py

    # custom data / output
    python src/models_training/finetune_cross_encoder.py \\
        --data    data/questions_data/QD_data/QD_contrastive_data.csv \\
        --output  models/finetuned_CE_bge \\
        --epochs  3 --batch-size 8 --lr 5e-5
"""
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from torch.utils.data import DataLoader

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Data loading ──────────────────────────────────────────────────── #

def _load_samples(csv_path: str) -> list[InputExample]:
    """Load ``question / text / label`` CSV and return a list of ``InputExample``."""
    df = pd.read_csv(csv_path)
    required = {"question", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[["question", "text", "label"]].dropna()
    samples = [
        InputExample(texts=[str(row.question), str(row.text)], label=float(row.label))
        for row in df.itertuples(index=False)
    ]
    logger.info("Loaded %d samples from %s", len(samples), csv_path)
    return samples


# ── Train / val / test split ──────────────────────────────────────── #

def _split(
    samples: list[InputExample],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[list, list, list]:
    """
    Shuffle and partition *samples* into train / val / test splits.

    The split is deterministic for a given *random_state*.  Test samples are
    taken from the front of the shuffled list so that the indices are stable
    regardless of the val_ratio value.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    data = samples.copy()
    random.seed(random_state)
    random.shuffle(data)

    total  = len(data)
    n_test = int(total * test_ratio)
    n_val  = int(total * val_ratio)

    test_samples  = data[:n_test]
    val_samples   = data[n_test: n_test + n_val]
    train_samples = data[n_test + n_val:]

    logger.info(
        "Split: %d train / %d val / %d test  (total %d)",
        len(train_samples), len(val_samples), len(test_samples), total,
    )
    return train_samples, val_samples, test_samples


# ── Test evaluation ───────────────────────────────────────────────── #

def _evaluate_test(model: CrossEncoder, test_samples: list[InputExample]) -> dict:
    """Score the test split and print regression + correlation metrics."""
    pairs       = [ex.texts for ex in test_samples]
    true_labels = [ex.label for ex in test_samples]
    predictions = model.predict(pairs)

    mse  = mean_squared_error(true_labels, predictions)
    mae  = mean_absolute_error(true_labels, predictions)
    rmse = float(np.sqrt(mse))
    corr = float(np.corrcoef(true_labels, predictions)[0, 1])

    logger.info("Test results  →  MSE=%.4f  RMSE=%.4f  MAE=%.4f  Correlation=%.4f",
                mse, rmse, mae, corr)
    return {"mse": mse, "rmse": rmse, "mae": mae, "correlation": corr,
            "num_samples": len(test_samples)}


# ── Main ──────────────────────────────────────────────────────────── #

def finetune(args: argparse.Namespace) -> None:
    """Orchestrate the full fine-tuning pipeline from data loading to evaluation."""
    # 1. Load data
    samples = _load_samples(args.data)
    train_samples, val_samples, test_samples = _split(
        samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    # 2. Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading base model: %s  (max_length=%d, device=%s)",
                args.model_name, args.max_length, device)
    model = CrossEncoder(
        args.model_name,
        num_labels=1,
        max_length=args.max_length,
        device=device,
    )

    # 3. Evaluator from val split
    evaluator = CECorrelationEvaluator.from_input_examples(val_samples, name="validation")

    # 4. Train
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training  →  output: %s", output_path)
    model.fit(
        train_dataloader=DataLoader(train_samples, shuffle=True, batch_size=args.batch_size),
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        output_path=str(output_path),
        evaluation_steps=args.eval_steps,
        save_best_model=True,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        use_amp=args.use_amp,
    )
    logger.info("Training complete. Model saved → %s", output_path)

    # 5. Test-set evaluation
    logger.info("Evaluating on held-out test split (%d samples) …", len(test_samples))
    _evaluate_test(model, test_samples)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune a cross-encoder reranker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default="data/questions_data/QD_data/QD_contrastive_data.csv",
        help="CSV with columns: question, text, label.",
    )
    p.add_argument(
        "--output",
        default="models/finetuned_CE_bge",
        help="Directory to save the fine-tuned model.",
    )
    p.add_argument("--model-name",   default="BAAI/bge-reranker-v2-m3",
                   dest="model_name",
                   help="HuggingFace model ID or local path for the base cross-encoder.")
    p.add_argument("--max-length",   type=int,   default=512,   dest="max_length",
                   help="Maximum token sequence length fed to the model.")
    p.add_argument("--epochs",       type=int,   default=3,
                   help="Number of full passes over the training data.")
    p.add_argument("--batch-size",   type=int,   default=8,     dest="batch_size",
                   help="Number of samples per training step.")
    p.add_argument("--lr",           type=float, default=5e-5,
                   help="Peak learning rate for the AdamW optimiser.")
    p.add_argument("--warmup-steps", type=int,   default=0,     dest="warmup_steps",
                   help="Number of linear warm-up steps before reaching the peak lr.")
    p.add_argument("--eval-steps",   type=int,   default=500,   dest="eval_steps",
                   help="Evaluate on the validation set every N steps.")
    p.add_argument("--use-amp",      action="store_true", default=True,  dest="use_amp",
                   help="Enable automatic mixed precision (fp16).")
    p.add_argument("--no-amp",       action="store_false", dest="use_amp")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Random seed for shuffling and reproducibility.")
    p.add_argument("--val-ratio",    type=float, default=0.1,   dest="val_ratio",
                   help="Fraction of data reserved for validation (used during training).")
    p.add_argument("--test-ratio",   type=float, default=0.1,   dest="test_ratio",
                   help="Fraction of data reserved for the final held-out test evaluation.")
    return p.parse_args(argv)


if __name__ == "__main__":
    finetune(parse_args())
