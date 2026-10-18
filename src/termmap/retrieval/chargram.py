"""Character 3-4-gram TF-IDF retrieval (char_wb, sublinear tf, cosine) — scispaCy-style candidates."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from termmap.retrieval.fusion import Ranking, collapse, top_n


class CharGramIndex:
    """TF-IDF over character n-grams of description texts, each text mapped to a code."""

    def __init__(self, texts: list[str], codes: list[str]):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True,
                                   dtype=np.float32, lowercase=False)
        self.X = self.vec.fit_transform(texts)
        self.XT = self.X.T.tocsr()
        self.codes = np.asarray(codes, dtype=object)

    def search(self, queries: list[str], top_desc: int = 200, top_codes: int = 100,
               batch: int = 64) -> list[Ranking]:
        """Top codes per query by best cosine similarity of any of the code's texts."""
        out: list[Ranking] = []
        Q = self.vec.transform(queries)
        for s in range(0, len(queries), batch):
            S = (Q[s:s + batch] @ self.XT).tocsr()
            for r in range(S.shape[0]):
                lo, hi = S.indptr[r], S.indptr[r + 1]
                cols, vals = S.indices[lo:hi], S.data[lo:hi]
                if len(vals) == 0:
                    out.append([])
                    continue
                sel, sc = top_n(vals, top_desc)
                out.append(collapse(cols[sel], sc, self.codes, top_codes))
        return out
