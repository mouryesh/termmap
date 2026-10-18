"""Track F: out-of-fold ablation of the LambdaRank feature groups on the v1 saved pools and CE scores.

Usage: python scripts/ranker_features.py   (v1 config; writes reports/ranker_features.json)
Same folds, same CE scores, same ranker settings as v1; only the feature columns change. Features are
cached to artifacts/cv/rf_feats_<system>.parquet.
"""
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from termmap.config import load_config, set_seeds
from termmap.rerank.cross_encoder import CandidateText
from termmap.rerank.features import FeatureBuilder
from termmap.rerank.train_ltr import cv_ranker, oof_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
GROUPS = ("string", "term", "structure", "pool")
PREFIX = {"string": "s_", "term": "t_", "structure": "c_", "pool": "p_"}
cfg = load_config("configs/default.yaml")
set_seeds(cfg["seed"])
art = Path(cfg["paths"]["artifacts"])
rows_all = pd.read_parquet(art / "cv" / "rows.parquet")


def correct(feats, score, rows):
    top = feats[["row", "code", "label"]].assign(s=score).sort_values(["row", "s", "code"], ascending=[True, False, True])
    hit = top.drop_duplicates("row").set_index("row")["label"]
    return hit.reindex(rows.index).fillna(0).astype(int).values  # rows with gold outside the pool count as wrong


def bootstrap(a, b, n=2000, seed=42):
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n, len(a)))
    d = (b[idx] - a[idx]).mean(axis=1) * 100
    return [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]


out = {}
for s in sys.argv[1:] or ("ICD10CM", "SNOMEDCT_US"):
    t0 = time.time()
    rows = rows_all[rows_all.coding_system == s]
    cache = art / "cv" / f"rf_feats_{s}.parquet"
    if cache.exists():
        feats = pd.read_parquet(cache)
    else:
        cands = pd.read_parquet(art / "cv" / f"candidates_ce_{s}.parquet").reset_index(drop=True)
        table = pd.read_parquet(art / "index" / f"{s}_table.parquet")
        meta = pd.read_parquet(art / "index" / f"{s}_meta.parquet")
        fb = FeatureBuilder(s, table, meta, CandidateText(s, table, meta), 1, GROUPS)
        feats = oof_features(s, rows, cands, fb, "ce_ctx")
        feats.to_parquet(cache)
    cat = ["chapter"] if s == "ICD10CM" else ["semantic_tag"]
    gcols = {g: [c for c in feats if c.startswith(PREFIX[g])] for g in GROUPS}
    allg = sum(gcols.values(), [])
    log = logging.getLogger("rf")
    log.info("%s: %d rows, %d candidates, features built in %.0fs, groups %s", s, len(rows), len(feats),
             time.time() - t0, {g: len(c) for g, c in gcols.items()})
    variants = {"baseline": allg, **{f"+{g}": [c for c in allg if c not in gcols[g]] for g in GROUPS}, "all": []}
    res, hits = {}, {}
    for name, drop in variants.items():
        f = feats.drop(columns=drop)
        oof, iters = cv_ranker(f, len(rows), cfg, cat)
        hits[name] = correct(f, oof, rows)
        fold = rows["fold"].values
        res[name] = {"top1": round(100 * hits[name].mean(), 2), "best_iters": iters,
                     "per_fold": [round(100 * hits[name][fold == k].mean(), 2) for k in sorted(set(fold))]}
        if name != "baseline":
            res[name]["diff"] = round(res[name]["top1"] - res["baseline"]["top1"], 2)
            res[name]["ci95"] = bootstrap(hits["baseline"], hits[name])
            res[name]["fold_diff"] = [round(a - b, 2) for a, b in zip(res[name]["per_fold"], res["baseline"]["per_fold"])]
        log.info("%s %s: %s", s, name, res[name])
    out[s] = {"rows": len(rows), "candidates": len(feats), "variants": res, "seconds": round(time.time() - t0)}
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame(hits).assign(row=rows.index).to_parquet(art / "cv" / f"rf_hits_{s}.parquet")
    prev = json.loads(Path("reports/ranker_features.json").read_text()) if Path("reports/ranker_features.json").exists() else {}
    Path("reports/ranker_features.json").write_text(json.dumps({**prev, **out}, indent=1))
