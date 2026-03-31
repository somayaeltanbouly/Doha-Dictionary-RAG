"""
retriever.py — HybridRetriever for the Doha Historical Dictionary RAG
======================================================================

``HybridRetriever`` combines BM25 keyword search, dense FAISS vector search,
and cross-encoder reranking into a single configurable class.  The retrieval
strategy is selected at call time:

* ``"bm25"``   — BM25 keyword search with optional cross-encoder reranking.
* ``"dense"``  — FAISS nearest-neighbour search with optional reranking.
* ``"hybrid"`` — Reciprocal Rank Fusion of BM25 + dense, then reranking.

Usage::

    from retriever import HybridRetriever

    retriever = HybridRetriever()
    query_info = retriever.analyze_query(user_query)   # {"q1": ..., "intent": ...}
    docs_df, indices = retriever.retrieve(
        query_info["q1"], method="hybrid"
    )
    docs_str = retriever.format_documents(docs_df, query_info["intent"])
"""
from __future__ import annotations

import re
from typing import Optional

import faiss
import joblib
import numpy as np
import pandas as pd
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────── #
# Arabic NLP helpers                                                   #
# ─────────────────────────────────────────────────────────────────── #

_TASHKEEL_RE = re.compile(
    r"[\u064B-\u065F"   # fathatan … sukun (basic tashkeel)
    r"\u0610-\u061A"    # extended Arabic marks
    r"\u06D6-\u06DC"    # Quranic annotation marks
    r"\u06DF-\u06E4"    # Quranic annotation marks (cont.)
    r"\u06E7\u06E8"     # Arabic small high yeh / noon
    r"\u06EA-\u06ED]"   # Quranic annotation marks (cont.)
)

# General query words that carry no discriminative retrieval signal
_STOP_WORDS: set[str] = {
    "ما", "ماذا", "هل", "من", "متى", "كيف", "لخص", "اشتقاق", "شاهد",
    "سياق", "جذر", "معنى", "دلالات", "هو", "هي", "الفرق", "بين",
    "اين", "التي", "الذي", "كلمة", "لفظ", "عبارة", "المعاني",
    "الاصطلاحية", "الشاهد", "الاشتقاق", "الجذر", "المعنى",
    "العبارة", "الكلمة", "فسر",
}

# Additional keywords to drop that are specific to each intent class
_INTENT_STOP_WORDS: dict[str, list[str]] = {
    "author_of_citation":  ["قائل"],
    "historical_date":     ["تاريخ", "متى", "توثيق"],
    "part_of_speech":      ["اشتقاق", "صرفي"],
    "basic_meaning":       ["معنى", "دلالات"],
    "source_of_citation":  ["مصدر"],
    "contextual_meaning":  ["شاهد", "سياق"],
    "inscription":         ["نقش", "نقوش"],
    "etymology":           ["لغة", "عبرية", "سريانية", "قديمة"],
}

# Columns surfaced to the LLM per intent (restricts context to relevant fields)
_INTENT_COLUMNS: dict[str, list[str]] = {
    "author_of_citation":  ["الجذر", "الكلمة", "العبارة أو اللفظ المركب", "الشاهد", "القائل"],
    "historical_date":     ["الكلمة", "العبارة أو اللفظ المركب", "الشاهد", "تاريخ استعمال الشاهد"],
    "part_of_speech":      ["الجذر", "الكلمة", "الاشتقاق الصرفي للكلمة", "العبارة أو اللفظ المركب", "lemmaId"],
    "basic_meaning":       ["العبارة أو اللفظ المركب", "الشاهد", "المعنى", "الحقل الاصطلاحي"],
    "source_of_citation":  ["الكلمة", "العبارة أو اللفظ المركب", "المعنى", "الشاهد", "المصدر",
                            "رقم الصفحة", "السورة", "رقم الآية", "رقم الحديث"],
    "contextual_meaning":  ["الكلمة", "العبارة أو اللفظ المركب", "الشاهد", "المعنى", "الحقل الاصطلاحي"],
    "inscription":         ["rootId", "الجذر", "النقوش"],
    "etymology":           ["rootId", "الجذر", "اللغات القديمة"],
    "other":               ["الجذر", "الكلمة", "الاشتقاق الصرفي للكلمة", "العبارة أو اللفظ المركب",
                            "الشاهد", "المعنى", "القائل", "المصدر", "تاريخ استعمال الشاهد",
                            "الحقل الاصطلاحي", "lemmaId"],
}


def _remove_tashkeel(text: str) -> str:
    return _TASHKEEL_RE.sub("", str(text)) if text else text


def _tokenize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", "", text).split()


def _build_retrieval_query(user_query: str, intent: str) -> str:
    """Strip stop-words and intent-specific keywords from *user_query*.

    Single remaining keywords are triplicated so BM25 weights them higher.
    """
    clean = _remove_tashkeel(user_query)
    tokens = re.split(r"\s+", clean.strip())

    drop = _STOP_WORDS | set(_INTENT_STOP_WORDS.get(intent, []))
    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        norm = token.strip().lower()
        if norm not in drop and len(norm) > 1 and norm not in seen:
            keywords.append(token)
            seen.add(norm)

    if len(keywords) == 0:
        return clean
    if len(keywords) == 1:
        return " ".join(keywords * 3)   # triplicate for BM25 emphasis
    return " ".join(keywords)


# ─────────────────────────────────────────────────────────────────── #
# HybridRetriever                                                      #
# ─────────────────────────────────────────────────────────────────── #

class HybridRetriever:
    """Retrieves relevant dictionary entries for an Arabic query.

    Supports three retrieval strategies selectable at call time:

    * ``"bm25"``   — BM25 keyword search + optional cross-encoder reranking.
    * ``"dense"``  — FAISS dense search + optional cross-encoder reranking.
    * ``"hybrid"`` — Reciprocal Rank Fusion (BM25 + dense) + reranking
                     (default, best overall quality).

    The FAISS index is loaded from a serialised ``.index`` file produced by
    ``build_index.py``.  If that file is missing the index is built in-memory
    from the ``.npy`` embeddings file.  Both dense components are loaded
    **lazily** — BM25-only usage pays no embedding model loading cost.

    Args:
        corpus_path:           CSV with all structured dictionary entry fields
                               (must contain an ``ID`` column).
        text_data_path:        CSV with a plain-text ``text`` column used for
                               BM25 scoring and cross-encoder pair building.
        embeddings_path:       Path to pre-computed ``.npy`` float32 embeddings.
        index_path:            Path to serialised FAISS index (``.index`` file
                               produced by ``build_index.py``).  Loaded with
                               ``faiss.read_index()``.  Falls back to building
                               the index from *embeddings_path* if not found.
        classifier_path:       Trained scikit-learn intent classifier (``.joblib``).
        cross_encoder_path:    Fine-tuned cross-encoder for reranking.
                               Pass ``None`` to disable reranking entirely.
        embedding_model_path:  SentenceTransformer model for dense query encoding.
        top_k:                 Number of final documents returned per query.
        k_bm25:                Candidate pool size for single-method strategies
                               (BM25-only or dense-only).
        k_rrf:                 Candidate pool size per method before RRF fusion.
        k_rerank:              Top-RRF candidates forwarded to the cross-encoder.
        rrf_k_const:           RRF smoothing constant *k* (default 60).
        bm25_weight:           RRF score multiplier for BM25 results.
        dense_weight:          RRF score multiplier for dense results.
    """

    def __init__(
        self,
        corpus_path:           str = "data/retrieval_corpus/processed/DHDA_filtered_AR.csv",
        text_data_path:        str = "data/retrieval_corpus/processed/DHDA_text_to_embed.csv",
        embeddings_path:       str = "data/retrieval_corpus/vector_database/embeddings_nomic-embed.npy",
        index_path:            str = "data/retrieval_corpus/vector_database/faiss_nomic-embed.index",
        classifier_path:       str = "models/RF_intent_classifier.joblib",
        cross_encoder_path:    Optional[str] = "models/finetuned_CE_bge",
        embedding_model_path:  str = "models/nomic",
        top_k:     int   = 10,
        k_bm25:    int   = 50,
        k_rrf:     int   = 300,
        k_rerank:  int   = 50,
        rrf_k_const:  int   = 60,
        bm25_weight:  float = 0.55,
        dense_weight: float = 0.45,
    ) -> None:
        self.top_k       = top_k
        self.k_bm25      = k_bm25
        self.k_rrf       = k_rrf
        self.k_rerank    = k_rerank
        self.rrf_k_const = rrf_k_const
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # ── corpus data ────────────────────────────────────────────── #
        self.corpus_df = pd.read_csv(corpus_path)
        text_df = pd.read_csv(text_data_path)
        self.text_data: list[str] = list(text_df["text"].fillna(""))

        # ── BM25 index — built eagerly (cheap, CPU-only) ───────────── #
        print("Building BM25 index …")
        tokenized = [_tokenize(t) for t in self.text_data]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 ready ({len(self.text_data):,} documents).")

        # ── intent classifier ──────────────────────────────────────── #
        self._classifier = None
        try:
            self._classifier = joblib.load(classifier_path)
            print(f"Intent classifier loaded from {classifier_path!r}.")
        except FileNotFoundError:
            print(
                f"WARNING: classifier not found at {classifier_path!r}. "
                "Intent will default to 'other'."
            )

        # ── cross-encoder (reranker) — loaded eagerly if provided ──── #
        self._cross_encoder: Optional[CrossEncoder] = None
        if cross_encoder_path:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._cross_encoder = CrossEncoder(cross_encoder_path, device=device)
            self._cross_encoder.eval()
            print(f"Cross-encoder loaded from {cross_encoder_path!r} on {device}.")

        self._embedding_model_path = embedding_model_path
        self._embeddings_path      = embeddings_path
        self._index_path           = index_path
        self._embedding_model: Optional[SentenceTransformer] = None
        self._faiss_index:     Optional[faiss.Index]         = None

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def analyze_query(self, user_query: str) -> dict[str, str]:
        """Classify intent and extract a clean retrieval query.

        Returns:
            Dict with keys ``"q1"`` (cleaned query string) and
            ``"intent"`` (predicted intent label, e.g. ``"basic_meaning"``).
            Falls back to ``"other"`` when no classifier is loaded.
        """
        if self._classifier is None:
            return {"q1": _remove_tashkeel(user_query), "intent": "other"}

        intent: str = self._classifier.predict([user_query])[0]
        q1 = _build_retrieval_query(user_query, intent)
        return {"q1": q1, "intent": intent}

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
    ) -> tuple[pd.DataFrame, list[int]]:
        """Retrieve top-*k* dictionary entries for *query*.

        Args:
            query:  Pre-processed retrieval query (output of :meth:`analyze_query`).
            method: ``"bm25"``, ``"dense"``, or ``"hybrid"``.

        Returns:
            Tuple of (DataFrame of top-k corpus rows, list of integer corpus indices).
        """
        if method == "bm25":
            ranked     = self._retrieve_bm25(query, self.k_bm25)
            candidates = [idx for idx, _ in ranked]
        elif method == "dense":
            ranked     = self._retrieve_dense(query, self.k_bm25)
            candidates = [idx for idx, _ in ranked]
        else:  # hybrid (default)
            bm25_ranked  = self._retrieve_bm25(query, self.k_rrf)
            dense_ranked = self._retrieve_dense(query, self.k_rrf)
            candidates   = self._fuse_rrf(bm25_ranked, dense_ranked)[: self.k_rerank]

        final_indices, _ = self._rerank(query, candidates, self.top_k)

        rows = [self.corpus_df.iloc[i].to_frame().T for i in final_indices]
        result_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        return result_df, final_indices

    def format_documents(self, df: pd.DataFrame, intent: str) -> str:
        """Filter *df* to intent-relevant columns and serialise as JSON lines.

        Returns:
            A newline-separated JSON string, one document object per line.
            Columns listed in :data:`_INTENT_COLUMNS` that are absent from
            *df* are silently skipped.
        """
        required = _INTENT_COLUMNS.get(intent, _INTENT_COLUMNS["other"])
        cols = [c for c in required if c in df.columns]
        return df[cols].to_json(orient="records", lines=True, force_ascii=False)

    # ────────────────────────────────────────────────────────────────
    # Private helpers
    # ────────────────────────────────────────────────────────────────

    def _retrieve_bm25(self, query: str, k: int) -> list[tuple[int, int]]:
        """Return ``(corpus_idx, rank)`` pairs from BM25 (1-based rank)."""
        tokenized = _tokenize(query)
        # Pass indices as the corpus so get_top_n returns corpus positions directly,
        # avoiding an O(n) list.index() scan and incorrect results with duplicate text.
        top_indices = self.bm25.get_top_n(
            tokenized, list(range(len(self.text_data))), n=k
        )
        return [(int(idx), rank + 1) for rank, idx in enumerate(top_indices)]

    def _retrieve_dense(self, query: str, k: int) -> list[tuple[int, int]]:
        """Return ``(corpus_idx, rank)`` pairs from FAISS dense search (1-based rank)."""
        self._ensure_dense_loaded()
        assert self._embedding_model is not None
        assert self._faiss_index is not None

        q_emb = self._embedding_model.encode(query, convert_to_numpy=True)
        # L2-normalise so inner product equals cosine similarity
        q_emb = normalize(np.array(q_emb).reshape(1, -1).astype("float32"))
        _, I  = self._faiss_index.search(q_emb, k)
        return [(int(idx), rank + 1) for rank, idx in enumerate(I[0])]

    def _fuse_rrf(
        self,
        bm25_results:  list[tuple[int, int]],
        dense_results: list[tuple[int, int]],
    ) -> list[int]:
        """Reciprocal Rank Fusion → list of corpus indices sorted by fused score."""
        fused: dict[int, float] = {}
        for idx, rank in bm25_results:
            fused[idx] = fused.get(idx, 0.0) + self.bm25_weight / (self.rrf_k_const + rank)
        for idx, rank in dense_results:
            fused[idx] = fused.get(idx, 0.0) + self.dense_weight / (self.rrf_k_const + rank)
        return [idx for idx, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)]

    def _rerank(
        self,
        query:             str,
        candidate_indices: list[int],
        top_k:             int,
    ) -> tuple[list[int], list[float]]:
        """Cross-encoder reranking.

        Returns:
            ``(indices, scores)`` — both lists have length ``min(top_k, |candidates|)``.
            When no cross-encoder is loaded, candidates are returned as-is with
            dummy scores of ``1.0``.
        """
        if not candidate_indices:
            return [], []
        if self._cross_encoder is None:
            final = candidate_indices[:top_k]
            return final, [1.0] * len(final)

        pairs  = [[query, self.text_data[idx]] for idx in candidate_indices]
        scores = self._cross_encoder.predict(pairs)
        reranked = sorted(
            zip(candidate_indices, scores),
            key=lambda x: (x[1], -x[0]),
            reverse=True,
        )
        final = reranked[:top_k]
        return [i for i, _ in final], [float(s) for _, s in final]

    def _ensure_dense_loaded(self) -> None:
        """Load the embedding model and FAISS index on first use.

        Tries to load the pre-built index file first; falls back to building
        an in-memory ``IndexFlatL2`` from the ``.npy`` embeddings file.
        """
        if self._faiss_index is not None:
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model from {self._embedding_model_path!r} …")
        self._embedding_model = SentenceTransformer(
            self._embedding_model_path, device=device, trust_remote_code=True
        )
        self._embedding_model.eval()

        import os
        if os.path.isfile(self._index_path):
            # Preferred path: load the pre-built serialised index
            print(f"Loading FAISS index from {self._index_path!r} …")
            self._faiss_index = faiss.read_index(self._index_path)
            print(f"FAISS index ready ({self._faiss_index.ntotal:,} vectors, "
                  f"dim={self._faiss_index.d}).")
        else:
            # Fall-back: build IndexFlatL2 in-memory from the .npy embeddings
            print(
                f"FAISS index file not found at {self._index_path!r}. "
                f"Building in-memory index from {self._embeddings_path!r} …"
            )
            embeddings = np.load(self._embeddings_path).astype(np.float32)
            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatL2(dim)
            self._faiss_index.add(embeddings)
            print(f"FAISS index ready ({self._faiss_index.ntotal:,} vectors, dim={dim}).")
