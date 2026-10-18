"""Stage 5b: confidence — calibrated top probability, threshold for a target precision, coverage curve."""
from __future__ import annotations

import numpy as np


def threshold_for_precision(conf: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Smallest confidence threshold whose accepted rows (conf >= t) reach `target` precision.

    Returns +inf when no threshold reaches it (then every row is flagged low-confidence).
    """
    order = np.argsort(-conf, kind="stable")
    c, ok = conf[order], correct[order].astype(float)
    prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    best = np.inf
    for i in range(len(c)):
        if prec[i] >= target and (i + 1 == len(c) or c[i] != c[i + 1]):
            best = c[i]
    return float(best)


def coverage_curve(conf: np.ndarray, correct: np.ndarray, points: int = 20) -> list[dict]:
    """Accuracy of the most-confident rows at each coverage level."""
    order = np.argsort(-conf, kind="stable")
    ok = correct[order].astype(float)
    out = []
    for q in np.linspace(1 / points, 1.0, points):
        k = max(1, int(round(q * len(ok))))
        out.append({"coverage": round(float(q), 3), "accuracy": round(float(ok[:k].mean()), 4),
                    "threshold": round(float(conf[order][k - 1]), 4)})
    return out


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-width probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(ece)
