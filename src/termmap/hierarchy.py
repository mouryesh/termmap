"""Stage 4: ICD hierarchy — temperature-scaled softmax, choose the 3-character category, then the code.

SNOMED skips the category step (the supplied files carry no hierarchy) but uses the same
temperature-scaled probabilities for confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

T_GRID = np.round(np.exp(np.linspace(np.log(0.05), np.log(5.0), 41)), 4)


def group_softmax(df: pd.DataFrame, score_col: str, T: float) -> np.ndarray:
    """Softmax of score / T within each term ('row'), aligned with df."""
    z = df[score_col].to_numpy(float) / T
    zmax = pd.Series(z, index=df.index).groupby(df["row"]).transform("max").to_numpy()
    e = np.exp(z - zmax)
    denom = pd.Series(e, index=df.index).groupby(df["row"]).transform("sum").to_numpy()
    return e / denom


def nll(df: pd.DataFrame, score_col: str, T: float) -> float:
    """Mean negative log-likelihood of the gold code, over terms whose pool contains it."""
    p = group_softmax(df, score_col, T)
    gold = p[df["label"].to_numpy() == 1]
    return float(-np.log(np.clip(gold, 1e-12, None)).mean()) if len(gold) else float("inf")


def fit_temperature(df: pd.DataFrame, score_col: str) -> float:
    """Grid search over T (log-spaced 0.05-5) minimizing gold NLL; ties to the smaller T."""
    losses = [(nll(df, score_col, T), T) for T in T_GRID]
    return float(min(losses)[1])


def choose_icd(codes: list[str], probs: np.ndarray, margin: float) -> tuple[str, dict]:
    """Pick the top category by summed probability (plus the runner-up if within `margin`),
    then the most probable code inside the kept categories. Ties break by code/category string.
    """
    mass: dict[str, float] = {}
    for c, p in zip(codes, probs):
        mass[c[:3]] = mass.get(c[:3], 0.0) + float(p)
    cats = sorted(mass.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = {cats[0][0]}
    if len(cats) > 1 and cats[0][1] - cats[1][1] <= margin:
        kept.add(cats[1][0])
    best = sorted(((p, c) for c, p in zip(codes, probs) if c[:3] in kept), key=lambda x: (-x[0], x[1]))[0]
    argmax = sorted(zip(probs, codes), key=lambda x: (-x[0], x[1]))[0][1]
    info = {"top_category": cats[0][0], "category_mass": round(cats[0][1], 4),
            "runner_up": cats[1][0] if len(cats) > 1 else "", "kept": "|".join(sorted(kept)),
            "changed_from_argmax": int(best[1] != argmax)}
    return best[1], info


def choose_argmax(codes: list[str], probs: np.ndarray) -> str:
    return sorted(zip(probs, codes), key=lambda x: (-x[0], x[1]))[0][1]
