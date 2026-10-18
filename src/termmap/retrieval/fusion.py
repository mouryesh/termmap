"""Collapsing description hits to codes, and Reciprocal Rank Fusion (Cormack et al., 2009)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Ranking = list[tuple[str, float]]  # (code, score), best first


def collapse(idx: np.ndarray, scores: np.ndarray, codes: np.ndarray, top_codes: int) -> Ranking:
    """Map description hits to codes, keeping each code's best score; ties broken by code string."""
    best: dict[str, float] = {}
    for i, s in zip(idx.tolist(), scores.tolist()):
        c = codes[i]
        if c not in best or s > best[c]:
            best[c] = s
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:top_codes]


def top_n(row_scores: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and scores of the n largest entries of a 1-D array (unordered beyond n)."""
    n = min(n, row_scores.shape[0])
    if n <= 0:
        return np.empty(0, int), np.empty(0, np.float32)
    idx = np.argpartition(-row_scores, n - 1)[:n]
    return idx, row_scores[idx]


def best_rank_merge(rankings: Sequence[Ranking]) -> Ranking:
    """Merge several rankings of one channel (one per query variant) by best rank per code.

    The returned 'score' is the negative best rank, so ordering is by best rank then code.
    """
    best: dict[str, int] = {}
    for r in rankings:
        for rank, (code, _) in enumerate(r, start=1):
            if code not in best or rank < best[code]:
                best[code] = rank
    return sorted(((c, -float(r)) for c, r in best.items()), key=lambda kv: (-kv[1], kv[0]))


def rrf(rankings: Sequence[Ranking], k: int = 60, weights: Sequence[float] | None = None) -> Ranking:
    """(Weighted) Reciprocal Rank Fusion: score(code) = sum_i w_i / (k + rank_i); equal weights by default."""
    score: dict[str, float] = {}
    weights = weights if weights is not None else [1.0] * len(rankings)
    for r, w in zip(rankings, weights):
        if w == 0:
            continue
        for rank, (code, _) in enumerate(r, start=1):
            score[code] = score.get(code, 0.0) + w / (k + rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
