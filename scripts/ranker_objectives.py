"""Track F 3.4: objective / regularisation on the cached features ('all' groups), chosen on the INNER split.

For each outer fold, each setting trains on the inner-train rows and is early-stopped and scored on the
inner-validation rows (top-1). The setting with the best mean inner top-1 is then reported out-of-fold.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import termmap.rerank.ltr as ltr
from termmap.config import load_config, set_seeds
from termmap.rerank.ltr import fit, predict, split_groups, top1_from_scores

cfg = load_config("configs/default.yaml"); set_seeds(cfg["seed"])
base = ltr.ranker_params
GRID = {"v1": {}, "lr_at1_t10": {"eval_at": [1], "lambdarank_truncation_level": 10},
        "xendcg": {"objective": "rank_xendcg", "eval_at": [1]},
        "leaves15": {"num_leaves": 15, "eval_at": [1], "lambdarank_truncation_level": 10},
        "leaves7_mcs50_l2": {"num_leaves": 7, "min_child_samples": 50, "lambda_l2": 5.0, "eval_at": [1],
                             "lambdarank_truncation_level": 10},
        "ff05": {"feature_fraction": 0.5, "eval_at": [1], "lambdarank_truncation_level": 10}}
out = {}
for s in sys.argv[1:] or ("ICD10CM", "SNOMEDCT_US"):
    feats = pd.read_parquet(f"artifacts/cv/rf_feats_{s}.parquet")
    n = int((pd.read_parquet("artifacts/cv/rows.parquet").coding_system == s).sum())
    cat = ["chapter"] if s == "ICD10CM" else ["semantic_tag"]
    res = {}
    for name, over in GRID.items():
        ltr.ranker_params = lambda c, over=over: {**base(c), **over}
        inner, oof = [], np.full(len(feats), np.nan)
        for f in sorted(feats.fold.unique()):
            tr, te = feats[feats.fold != f], feats[feats.fold == f]
            keep, val = split_groups(tr["row"].values, 0.1, cfg["seed"] + int(f))
            v = tr[tr.row.isin(val)]
            b, _ = fit(tr[tr.row.isin(keep)], cfg, cat, val=v)
            inner.append(100 * top1_from_scores(v, predict(b, v), v.row.nunique()))
            oof[feats.index.get_indexer(te.index)] = predict(b, te)
        res[name] = {"inner_top1": round(float(np.mean(inner)), 2), "oof_top1": round(100 * top1_from_scores(feats, oof, n), 2)}
        print(s, name, res[name], flush=True)
    best = max(res, key=lambda k: res[k]["inner_top1"])
    out[s] = {"grid": res, "chosen_on_inner": best}
ltr.ranker_params = base
Path("reports/ranker_objectives.json").write_text(json.dumps(out, indent=1))
print(json.dumps({s: v["chosen_on_inner"] for s, v in out.items()}))
