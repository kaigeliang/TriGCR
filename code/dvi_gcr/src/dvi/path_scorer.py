"""Path scoring for DVI verification.

This module provides the v2 verifier: enumerate KG paths first, then rank them
against the question/constraint with a bi-encoder followed by a cross-encoder.
It avoids KG-LLM constrained decoding when `--path_scorer` is used.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _load_sentence_transformers():
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer, util
        return SentenceTransformer, CrossEncoder, util
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for --path_scorer. "
            "Install it with: pip install sentence-transformers"
        ) from e


class PathScorer:
    """Rank DFS-enumerated KG paths with bi-encoder retrieval and reranking."""

    DEFAULT_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        bi_encoder_name: str = DEFAULT_BI_ENCODER,
        cross_encoder_name: str = DEFAULT_CROSS_ENCODER,
        bi_k: int = 100,
        cross_k: int = 10,
        device: Optional[str] = None,
    ):
        SentenceTransformer, CrossEncoder, self._util = _load_sentence_transformers()

        self.bi_k = bi_k
        self.cross_k = cross_k

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info("[PathScorer] Loading bi-encoder: %s on %s", bi_encoder_name, device)
        self.bi_encoder = SentenceTransformer(bi_encoder_name, device=device)

        logger.info("[PathScorer] Loading cross-encoder: %s on %s", cross_encoder_name, device)
        self.cross_encoder = CrossEncoder(cross_encoder_name, device=device)

    def score_paths(
        self,
        paths: List[str],
        question: str,
        constraint_hint: str = "",
    ) -> List[str]:
        """Return top-ranked paths for one constraint."""
        if not paths:
            return []

        candidates = paths
        if len(paths) > self.bi_k:
            candidates = self._bi_encoder_filter(paths, question, constraint_hint)

        return self._cross_encoder_rerank(candidates, question, constraint_hint)

    def score_constraints(
        self,
        paths_per_constraint: dict,
        question: str,
        hints_per_constraint: dict | None = None,
    ) -> dict:
        """Rank paths for each constraint and return `{constraint_id: top_paths}`."""
        hints_per_constraint = hints_per_constraint or {}
        result = {}
        for cid, paths in paths_per_constraint.items():
            if not paths:
                result[cid] = []
                continue
            result[cid] = self.score_paths(
                paths,
                question,
                hints_per_constraint.get(cid, ""),
            )
        return result

    def _build_query(self, question: str, constraint_hint: str) -> str:
        if constraint_hint and constraint_hint.strip() and constraint_hint.strip() != question.strip():
            return f"{constraint_hint.strip()} {question.strip()}"
        return question.strip()

    def _bi_encoder_filter(
        self,
        paths: List[str],
        question: str,
        constraint_hint: str,
    ) -> List[str]:
        query = self._build_query(question, constraint_hint)
        query_emb = self.bi_encoder.encode(
            [query], convert_to_tensor=True, show_progress_bar=False
        )
        path_embs = self.bi_encoder.encode(
            paths, convert_to_tensor=True, show_progress_bar=False, batch_size=256
        )
        scores = self._util.cos_sim(query_emb, path_embs)[0].cpu().numpy()
        top_indices = np.argsort(scores)[::-1][: self.bi_k]
        return [paths[i] for i in top_indices]

    def _cross_encoder_rerank(
        self,
        candidates: List[str],
        question: str,
        constraint_hint: str,
    ) -> List[str]:
        if not candidates:
            return []

        query = self._build_query(question, constraint_hint)
        pairs = [(query, path) for path in candidates]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        top_indices = np.argsort(scores)[::-1][: self.cross_k]
        return [candidates[i] for i in top_indices]
