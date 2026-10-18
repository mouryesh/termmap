"""BM25 over word tokens of descriptions (bm25s, Lucene variant)."""
from __future__ import annotations

import numpy as np

from termmap.normalize import tokens
from termmap.retrieval.fusion import Ranking, collapse


class BM25Index:
    """Word-level BM25; each indexed text maps to a code."""

    def __init__(self, texts: list[str], codes: list[str]):
        import bm25s

        self.model = bm25s.BM25(k1=1.5, b=0.75)
        self.model.index([tokens(t) or ["<empty>"] for t in texts], show_progress=False)
        self.vocab = self.model.vocab_dict
        self.codes = np.asarray(codes, dtype=object)

    def search(self, queries: list[str], top_desc: int = 200, top_codes: int = 100) -> list[Ranking]:
        """Top codes per query; queries with no known token get an empty ranking."""
        toks = [[t for t in tokens(q) if t in self.vocab] for q in queries]
        out: list[Ranking] = [[] for _ in queries]
        live = [i for i, t in enumerate(toks) if t]
        if not live:
            return out
        k = min(top_desc, len(self.codes))
        docs, scores = self.model.retrieve([toks[i] for i in live], k=k, show_progress=False, n_threads=1)
        for j, i in enumerate(live):
            keep = scores[j] > 0
            out[i] = collapse(docs[j][keep], scores[j][keep], self.codes, top_codes)
        return out
