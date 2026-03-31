"""
build_index.py — Embed the retrieval corpus and build a FAISS index
====================================================================

Encodes each document in the corpus text file using a ``SentenceTransformer``
model and writes two files to the output directory:

* ``embeddings_<model-tag>.npy``    — float32 array of shape ``(N, D)``
* ``faiss_<model-tag>.index``       — FAISS index used by the retriever

The model tag is derived from the model name (e.g. ``nomic-ai/nomic-embed-text-v2-moe``
→ ``nomic-embed``), so outputs from different models never overwrite each other.

If both output files already exist the script exits without re-embedding.
Pass ``--force`` to overwrite them.

FAISS index types (``--index-type``)
-------------------------------------
* ``Flat``    — Exact L2 search.  Best choice for most corpus sizes.
* ``IVFFlat`` — Approximate search; faster for very large corpora (>500k entries).
                Requires a training step, controlled by ``--nlist``.
* ``HNSW``    — Graph-based approximate search; fast queries, no training needed.

Usage::

    # embed with default settings
    python src/retrieval/build_index.py

    # force rebuild
    python src/retrieval/build_index.py --force

    # use approximate IVF index
    python src/retrieval/build_index.py --index-type IVFFlat --nlist 100
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Supported FAISS index constructors
_SUPPORTED_INDEX_TYPES = ("Flat", "IVFFlat", "HNSW")


# ── Corpus loading ────────────────────────────────────────────────── #

def _load_corpus(csv_path: str, text_col: str) -> list[str]:
    """
    Load the text column from *csv_path* and return a list of strings.

    Null values are replaced with empty strings so the embedding model
    always receives a valid input.
    """
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(
            f"Column '{text_col}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )
    texts = df[text_col].fillna("").tolist()
    logger.info("Loaded %d documents from %s", len(texts), csv_path)
    return texts


# ── Embedding ─────────────────────────────────────────────────────── #

def _embed(
    texts: list[str],
    model_path: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Encode *texts* with the given ``SentenceTransformer`` model.

    Returns a float32 NumPy array of shape ``(N, D)`` where ``N`` is the
    number of documents and ``D`` is the embedding dimension of the model.
    ``trust_remote_code=True`` is passed to support models such as Nomic
    Embed that ship custom pooling code.
    """
    logger.info("Loading embedding model: %s  (device=%s)", model_path, device)
    model = SentenceTransformer(model_path, device=device, trust_remote_code=True)

    logger.info("Encoding %d documents (batch_size=%d) …", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    # Ensure float32 — faiss requires it
    embeddings = embeddings.astype(np.float32)
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


# ── FAISS index construction ──────────────────────────────────────── #

def _build_faiss_index(
    embeddings: np.ndarray,
    index_type: str,
    nlist: int,
) -> faiss.Index:
    """
    Construct and populate a FAISS index from *embeddings*.

    Args:
        embeddings:  Float32 array of shape ``(N, D)``.
        index_type:  One of ``"Flat"``, ``"IVFFlat"``, or ``"HNSW"``.
        nlist:       Number of Voronoi cells for ``IVFFlat`` (ignored for
                     other index types).

    Returns:
        A trained and populated ``faiss.Index`` instance.
    """
    dim = embeddings.shape[1]
    logger.info("Building FAISS %s index (dim=%d) …", index_type, dim)

    if index_type == "Flat":
        index = faiss.IndexFlatL2(dim)

    elif index_type == "IVFFlat":
        # IVFFlat needs a training pass over a representative sample of the
        # data before vectors can be added.
        quantiser = faiss.IndexFlatL2(dim)
        index     = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_L2)
        logger.info("Training IVFFlat index on %d vectors (nlist=%d) …",
                    len(embeddings), nlist)
        index.train(embeddings)

    elif index_type == "HNSW":
        # HNSW32 — 32 connections per layer is a good default balance of
        # speed and accuracy; no training step required.
        index = faiss.IndexHNSWFlat(dim, 32)

    else:
        raise ValueError(
            f"Unsupported index type '{index_type}'. "
            f"Choose from: {_SUPPORTED_INDEX_TYPES}"
        )

    index.add(embeddings)
    logger.info("Index populated: %d vectors.", index.ntotal)
    return index


# ── Main ──────────────────────────────────────────────────────────── #

def build(args: argparse.Namespace) -> None:
    """Orchestrate corpus loading, embedding, index construction, and saving."""
    # Derive a short model tag from the model name and embed it in the file
    # stems so that outputs from different models never collide.
    # Example: "nomic-ai/nomic-embed-text-v2-moe" → "nomic-embed"
    _basename = Path(args.model).name          # strip everything before the last /
    _parts    = _basename.split("-")
    model_tag = "-".join(_parts[:2])           # keep up to the second dash

    emb_path = Path(args.emb_out)
    emb_path = emb_path.with_name(f"{emb_path.stem}_{model_tag}{emb_path.suffix}")
    idx_path  = Path(args.idx_out)
    idx_path  = idx_path.with_name(f"{idx_path.stem}_{model_tag}{idx_path.suffix}")

    # ── Existence check ──────────────────────────────────────────── #
    if not args.force:
        emb_exists = emb_path.exists()
        idx_exists  = idx_path.exists()

        if emb_exists and idx_exists:
            logger.info(
                "Both outputs already exist — skipping.\n"
                "  embeddings : %s\n"
                "  FAISS index: %s\n"
                "Use --force to overwrite.",
                emb_path, idx_path,
            )
            return

        if emb_exists:
            logger.info(
                "Embeddings file already exists (%s). "
                "Loading it to build the FAISS index …", emb_path,
            )
            embeddings = np.load(str(emb_path)).astype(np.float32)
        else:
            # Full embedding pass required
            embeddings = None

    else:
        logger.info("--force set: existing outputs will be overwritten.")
        embeddings = None

    # ── Embed (if needed) ────────────────────────────────────────── #
    if embeddings is None:
        texts = _load_corpus(args.corpus, args.text_col)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        embeddings = _embed(texts, args.model, args.batch_size, device)

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_path), embeddings)
        logger.info("Embeddings saved → %s", emb_path)

    # ── Build and save FAISS index ───────────────────────────────── #
    index = _build_faiss_index(embeddings, args.index_type, args.nlist)

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    logger.info("FAISS index saved  → %s", idx_path)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embed the DHDA retrieval corpus and build a FAISS index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--corpus",
        default="data/retrieval_corpus/processed/DHDA_text_to_embed.csv",
        help="CSV corpus file produced by build_retrieval_corpus.py.",
    )
    p.add_argument(
        "--text-col",
        default="text",
        dest="text_col",
        help="Name of the column containing the text to embed.",
    )
    p.add_argument(
        "--model",
        default="nomic-ai/nomic-embed-text-v2-moe",
        help="SentenceTransformer model ID or local path used for encoding.",
    )
    p.add_argument(
        "--emb-out",
        default="data/retrieval_corpus/vector_database/embeddings.npy",
        dest="emb_out",
        help="Output path for the NumPy embeddings file (.npy).",
    )
    p.add_argument(
        "--idx-out",
        default="data/retrieval_corpus/vector_database/faiss.index",
        dest="idx_out",
        help="Output path for the serialised FAISS index.",
    )
    p.add_argument(
        "--index-type",
        default="Flat",
        dest="index_type",
        choices=_SUPPORTED_INDEX_TYPES,
        help=(
            "FAISS index type. 'Flat' is exact L2 search (recommended for "
            "corpora up to ~500k vectors). 'IVFFlat' and 'HNSW' are "
            "approximate but faster for larger corpora."
        ),
    )
    p.add_argument(
        "--nlist",
        type=int,
        default=100,
        help="Number of Voronoi cells for IVFFlat index (ignored for other types).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        dest="batch_size",
        help="Number of documents encoded per forward pass.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Compute device ('cuda', 'cpu'). Defaults to CUDA if available.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite existing embeddings and index files.",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    build(parse_args())
