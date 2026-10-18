"""Training memory: char TF-IDF + SapBERT over training terms and gold displays, mapped to codes.

Built from training folds only during cross-validation, so no row can retrieve itself.
"""
from __future__ import annotations

import numpy as np

from termmap.retrieval.chargram import CharGramIndex
from termmap.retrieval.fusion import Ranking, collapse, rrf, top_n


class MemoryIndex:
    """Nearest training terms/displays; char and dense sub-rankings fused by RRF into one channel."""

    def __init__(self, texts: list[str], codes: list[str], emb: np.ndarray, rrf_k: int = 60):
        self.char = CharGramIndex(texts, codes)
        self.emb = np.ascontiguousarray(emb, dtype=np.float32)
        self.codes = np.asarray(codes, dtype=object)
        self.rrf_k = rrf_k

    def search(self, queries: list[str], q_emb: np.ndarray, top_desc: int = 100,
               top_codes: int = 50) -> tuple[list[Ranking], list[Ranking], list[Ranking]]:
        """Returns (fused, char-only, dense-only) rankings per query."""
        char = self.char.search(queries, top_desc=top_desc, top_codes=top_codes)
        dense: list[Ranking] = []
        for s in range(0, len(queries), 1024):
            S = q_emb[s:s + 1024] @ self.emb.T
            for row in S:
                idx, sc = top_n(row, top_desc)
                dense.append(collapse(idx, sc, self.codes, top_codes))
        fused = [rrf([c, d], self.rrf_k)[:top_codes] for c, d in zip(char, dense)]
        return fused, char, dense
