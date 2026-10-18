"""Stage 3 Layer B: LightGBM LambdaRank per system (group = one term's candidates)."""
from __future__ import annotations

import logging

import lightgbm as lgb
import numpy as np
import pandas as pd

from termmap.rerank.features import feature_columns

log = logging.getLogger(__name__)


def ranker_params(cfg: dict) -> dict:
    return {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": cfg.get("ltr", {}).get("eval_at", [10]),
        "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 20,
        "feature_fraction": 0.8, "lambdarank_truncation_level": 20,
        "deterministic": True, "force_row_wise": True, "seed": cfg["seed"],
        "num_threads": cfg["num_threads"], "verbose": -1,
    }


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Group rows contiguously in a fixed order (row, then code) so LightGBM input is deterministic."""
    return df.sort_values(["row", "code"], kind="stable")


def _dataset(df: pd.DataFrame, cols: list[str], cat: list[str]) -> lgb.Dataset:
    sizes = df.groupby("row", sort=False).size().values
    return lgb.Dataset(df[cols], label=df["label"], group=sizes, categorical_feature=cat, free_raw_data=False)


def fit(train: pd.DataFrame, cfg: dict, cat: list[str], val: pd.DataFrame | None = None,
        num_rounds: int | None = None) -> tuple[lgb.Booster, int]:
    """Fit on rows whose pool contains the gold code; early-stop on `val` when given."""
    cols = feature_columns(train)
    train = _sorted(train[train.groupby("row")["label"].transform("max") > 0])
    params = ranker_params(cfg)
    dtrain = _dataset(train, cols, cat)
    if val is not None:
        val = _sorted(val[val.groupby("row")["label"].transform("max") > 0])
        dval = _dataset(val, cols, cat)
        booster = lgb.train(params, dtrain, num_boost_round=2000, valid_sets=[dval],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
        return booster, booster.best_iteration
    booster = lgb.train(params, dtrain, num_boost_round=num_rounds or 300)
    return booster, num_rounds or 300


def predict(booster: lgb.Booster, df: pd.DataFrame) -> np.ndarray:
    cols = feature_columns(df)
    return booster.predict(df[cols], num_iteration=booster.best_iteration or None)


def top1_from_scores(df: pd.DataFrame, score: np.ndarray, n_rows: int) -> float:
    """Share of all rows (including those with gold missing from the pool) ranked correctly first."""
    d = df[["row", "code", "label"]].assign(s=score)
    top = d.sort_values(["row", "s", "code"], ascending=[True, False, True]).drop_duplicates("row")
    return float(top["label"].sum()) / max(n_rows, 1)


def importances(booster: lgb.Booster) -> pd.DataFrame:
    return (pd.DataFrame({"feature": booster.feature_name(),
                          "gain": booster.feature_importance("gain"),
                          "splits": booster.feature_importance("split")})
            .sort_values("gain", ascending=False).reset_index(drop=True))


def split_groups(rows: np.ndarray, frac: float, seed: int) -> tuple[set, set]:
    """Deterministic group split of row ids for early stopping."""
    uniq = np.array(sorted(set(rows.tolist())))
    rng = np.random.default_rng(seed)
    val = set(rng.choice(uniq, size=max(1, int(frac * len(uniq))), replace=False).tolist())
    return set(uniq.tolist()) - val, val
