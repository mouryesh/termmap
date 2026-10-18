"""Stage 5a: joint re-scoring of the top-5 ICD x top-5 SNOMED codes for one term.

pair = log P(icd) + log P(snomed) + lambda * fit,  fit = 0.5 * co-occurrence + 0.5 * SapBERT cosine.
lambda = 0 is the independent baseline. Co-occurrence comes from training pairs only.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from termmap.normalize import level1, strip_semantic_tag


class CooccurrenceTable:
    """Smoothed co-occurrence: n(i, s) / (min(n(i), n(s)) + 1), in [0, 1)."""

    def __init__(self, pairs: list[tuple[str, str]]):
        self.n_pair = Counter(pairs)
        self.n_icd = Counter(i for i, _ in pairs)
        self.n_sct = Counter(s for _, s in pairs)

    def __call__(self, icd: str, sct: str) -> float:
        n = self.n_pair.get((icd, sct), 0)
        return n / (min(self.n_icd.get(icd, 0), self.n_sct.get(sct, 0)) + 1.0) if n else 0.0


class CodeEmbeddings:
    """SapBERT embeddings of each code's preferred description, cached per code."""

    def __init__(self, encoder, metas: dict[str, pd.DataFrame]):
        self.encoder, self.metas = encoder, metas
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def _text(self, system: str, code: str) -> str:
        m = self.metas[system]
        t = level1(m.at[code, "title"]) if code in m.index and isinstance(m.at[code, "title"], str) else code
        return strip_semantic_tag(t)

    def prime(self, keys: list[tuple[str, str]]) -> None:
        new = sorted({k for k in keys if k not in self._cache})
        if new:
            E = self.encoder.encode([self._text(s, c) for s, c in new])
            self._cache.update(zip(new, E))

    def cosine(self, icd: str, sct: str) -> float:
        return float(self._cache[("ICD10CM", icd)] @ self._cache[("SNOMEDCT_US", sct)])


def couple(icd: list[tuple[str, float]], sct: list[tuple[str, float]], lam: float,
           cooc: CooccurrenceTable, emb: CodeEmbeddings | None) -> tuple[str, str, float, float]:
    """Best (icd, snomed) pair over the given top-N lists of (code, prob).

    Returns (icd_code, snomed_code, fit of the chosen pair, co-occurrence of the chosen pair).
    Ties break by (icd, snomed) code strings.
    """
    best = None
    for ic, ip in icd:
        for sc, sp in sct:
            co = cooc(ic, sc)
            fit = 0.5 * co + 0.5 * (emb.cosine(ic, sc) if emb is not None and lam else 0.0)
            score = np.log(max(ip, 1e-12)) + np.log(max(sp, 1e-12)) + lam * fit
            key = (-score, ic, sc)
            if best is None or key < best[0]:
                best = (key, ic, sc, fit, co)
    return best[1], best[2], best[3], best[4]


def coupled_icd(icd_top: list[tuple[str, float]], icd_independent: str, snomed_top: list[tuple[str, float]],
                lam: float, cooc: CooccurrenceTable, emb: CodeEmbeddings | None) -> tuple[str, bool]:
    """ICD-only coupling for one term: (ICD code, whether coupling overrode the independent choice).

    icd_top is sorted best-first, so icd_top[0] is the argmax. When the best pair's ICD code is that
    argmax, the independent choice (which may come from the hierarchy step) is kept.
    """
    if not icd_top or not snomed_top or lam <= 0:
        return icd_independent, False
    ic, _, _, _ = couple(icd_top, snomed_top, lam, cooc, emb)
    if ic == icd_top[0][0]:
        return icd_independent, False
    return ic, True
