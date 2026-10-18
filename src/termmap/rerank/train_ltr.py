"""M3 driver: out-of-fold LambdaRank over cross-encoder + retrieval + qualifier features.

Features for fold f use statistics from the other folds only; the ranker for fold f is trained
on the other folds' (already out-of-fold) features and early-stopped on a group split of them.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from termmap.io import ICD, SNOMED
from termmap.rerank.cross_encoder import CandidateText
from termmap.rerank.features import FeatureBuilder
from termmap.rerank.ltr import fit, importances, predict, split_groups, top1_from_scores

log = logging.getLogger(__name__)
SETTINGS = {"entity+chunk": "ce_ctx", "entity-only": "ce_noctx", "no cross-encoder": None}


def oof_features(system: str, rows: pd.DataFrame, cands: pd.DataFrame, fb: FeatureBuilder,
                 ce_col: str | None) -> pd.DataFrame:
    parts = []
    for f in sorted(rows["fold"].unique()):
        te_rows, tr_rows = rows[rows.fold == f], rows[rows.fold != f]
        feats = fb.build(cands[cands.row.isin(te_rows.index)], te_rows, tr_rows, ce_col)
        feats["fold"] = int(f)
        parts.append(feats)
    return pd.concat(parts)


def cv_ranker(feats: pd.DataFrame, n_rows: int, cfg: dict, cat: list[str]) -> tuple[np.ndarray, list[int]]:
    """Out-of-fold ranker scores aligned with `feats`, and each fold's best iteration."""
    oof = np.full(len(feats), np.nan)
    iters = []
    for f in sorted(feats["fold"].unique()):
        tr, te = feats[feats.fold != f], feats[feats.fold == f]
        keep, val = split_groups(tr["row"].values, 0.1, cfg["seed"] + int(f))
        booster, best = fit(tr[tr.row.isin(keep)], cfg, cat, val=tr[tr.row.isin(val)])
        oof[feats.index.get_indexer(te.index)] = predict(booster, te)
        iters.append(int(best))
    return oof, iters


def run(cfg: dict, systems: tuple[str, ...] = (ICD, SNOMED)) -> dict:
    art = Path(cfg["paths"]["artifacts"])
    cv_dir, model_dir = art / "cv", art / "model"
    rows_all = pd.read_parquet(cv_dir / "rows.parquet")
    summary: dict = {}
    for s in systems:
        rows = rows_all[rows_all.coding_system == s]
        cands = pd.read_parquet(cv_dir / f"candidates_ce_{s}.parquet").reset_index(drop=True)
        table = pd.read_parquet(art / "index" / f"{s}_table.parquet")
        meta = pd.read_parquet(art / "index" / f"{s}_meta.parquet")
        fb = FeatureBuilder(s, table, meta, CandidateText(s, table, meta),
                            cfg.get("features", {}).get("qualifier_version", 1))
        res = {"rows": len(rows),
               "pool_ceiling": round(100 * cands.groupby("row")["gold"].max().sum() / len(rows), 1)}
        for name, ce_col in SETTINGS.items():
            feats = oof_features(s, rows, cands, fb, ce_col)
            if ce_col is None:
                feats = feats.drop(columns=[c for c in ("ce", "ce_missing", "ce_rank", "ce_gap") if c in feats])
            oof, iters = cv_ranker(feats, len(rows), cfg, fb.categorical)
            res[name] = {"top1": round(100 * top1_from_scores(feats, oof, len(rows)), 1), "best_iters": iters}
            log.info("%s %s: %s", s, name, res[name])
            if name == "entity+chunk":
                feats.assign(ltr=oof).to_parquet(cv_dir / f"ltr_oof_{s}.parquet", index=False)
                n_final = int(np.mean(iters))
                booster, _ = fit(feats, cfg, fb.categorical, num_rounds=n_final)
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(booster, model_dir / f"ltr_{s}.joblib")
                imp = importances(booster)
                imp.to_csv(Path(cfg["paths"]["reports"]) / f"feature_importance_{s}.csv", index=False)
                res["final_rounds"] = n_final
                res["top_features"] = imp.head(10)["feature"].tolist()
        summary[s] = res
    (cv_dir / "ltr_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
