"""`termmap train`: every stage from dictionaries + training CSV to the final predict artifacts.

Stages: indexes -> 3-fold out-of-fold candidates -> cross-encoders (checkpointed per fold; delete
artifacts/cv to retrain from scratch) -> LambdaRank -> ICD hierarchy calibration -> coupling and
confidence -> final abbreviation table, memory and training statistics. Writes
reports/validation_summary.json; the narrative report is reports/validation.md.
`termmap retrieval-cv` reproduces the M1 retrieval study (5-fold, channel ablations).
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from termmap.io import ICD, SNOMED, SYSTEMS

log = logging.getLogger(__name__)


def save_cooc_pairs(rows: pd.DataFrame, model_dir: Path) -> None:
    """(ICD, SNOMED) gold pairs of training instances that carry both codes, for the coupling step."""
    key = rows["entity"] + "\x00" + rows["chunk_text"]
    pr = pd.DataFrame({"key": key, "sys": rows["coding_system"], "code": rows["code"]})
    w = pr.pivot_table(index="key", columns="sys", values="code", aggfunc="first").dropna()
    w = w.rename(columns={ICD: "icd", SNOMED: "snomed"})[["icd", "snomed"]].reset_index(drop=True)
    w.to_parquet(model_dir / "cooc_pairs.parquet", index=False)


def prepare(cfg: dict) -> None:
    """Rebuild every data-derived predict artifact from the supplied dictionaries and training CSV: the
    indexes, training memory, abbreviations, frequency stats, fallbacks and co-occurrence pairs. The trained
    models (cross-encoders, LambdaRank, calibration JSON) come from the published weights."""
    from termmap.build_dataset import build_rows
    from termmap.pipeline import fit_final
    from termmap.retrieval.engine import Indexes, Retriever, build_index

    build_index(cfg)
    idx = Indexes(cfg)
    retr = Retriever(idx, cfg)
    fit_final(cfg, idx, retr)
    save_cooc_pairs(build_rows(cfg, {s: idx.sys[s].codes for s in SYSTEMS}), Path(cfg["paths"]["artifacts"]) / "model")


def _save_m5(cfg: dict, m5: dict, rows: pd.DataFrame, model_dir: Path) -> None:
    """Final coupling lambda (mode of the per-fold inner-CV choices), co-occurrence pairs, thresholds."""
    save_cooc_pairs(rows, model_dir)
    lams = m5["icd_only"]["lambda_per_fold"]
    lam = float(pd.Series(lams).mode().min())
    (model_dir / "coupling.json").write_text(json.dumps(
        {"mode": "icd_only", "lambda": lam, "lambda_per_fold": lams, "top_n": cfg["coupling"]["top_n"]}, indent=1))
    conf = {s: {"threshold": m5["confidence"][s]["threshold_98"],
                "target_precision": cfg["confidence"]["target_precision"]} for s in SYSTEMS}
    (model_dir / "confidence.json").write_text(json.dumps(conf, indent=1))


def train(cfg: dict) -> dict:
    from termmap.build_dataset import build_rows, oof_candidates
    from termmap.evaluate import evaluate_hierarchy, evaluate_m5
    from termmap.pipeline import fit_final
    from termmap.rerank import train_ce, train_ltr
    from termmap.retrieval.engine import Indexes, Retriever, build_index

    art = Path(cfg["paths"]["artifacts"])
    cv_dir, model_dir = art / "cv", art / "model"
    cv_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    build_index(cfg)
    idx = Indexes(cfg)
    retr = Retriever(idx, cfg)
    df = build_rows(cfg, {s: idx.sys[s].codes for s in SYSTEMS})
    df.to_parquet(cv_dir / "rows.parquet")
    for s, c in oof_candidates(cfg, df, idx, retr).items():
        c.to_parquet(cv_dir / f"candidates_{s}.parquet", index=False)
    fit_final(cfg, idx, retr)
    for s in SYSTEMS:
        meta_path = art / "index" / f"{s}_meta.parquet"
        if not meta_path.exists():
            idx.sys[s].meta.to_parquet(meta_path)
    del idx, retr
    gc.collect()
    summary: dict = {"cross_encoder": train_ce.run(cfg)}
    summary["lambdarank"] = train_ltr.run(cfg)
    h = evaluate_hierarchy(cfg)
    c = h["calibration"]
    (model_dir / "hierarchy.json").write_text(json.dumps(
        {ICD: {"T": c[ICD]["T"], "margin": cfg["hierarchy"]["category_margin"]}, SNOMED: {"T": c[SNOMED]["T"]}}, indent=1))
    summary["hierarchy"] = h
    m5 = evaluate_m5(cfg)
    _save_m5(cfg, m5, df, model_dir)
    summary["coupling_confidence"] = {k: v for k, v in m5.items() if k != "confidence"}
    summary["confidence"] = {s: {k: v for k, v in m5["confidence"][s].items() if k != "curve"} for s in SYSTEMS}
    summary["seconds"] = round(time.time() - t0, 1)
    rep = Path(cfg["paths"]["reports"])
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "validation_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log.info("train finished in %.0fs", summary["seconds"])
    return summary


def retrieval_cv(cfg: dict) -> None:
    """M1 study: 5-fold retrieval CV with channel and variant ablations (writes reports/m1_results.json)."""
    from termmap.evaluate import run_retrieval_cv
    from termmap.retrieval.engine import CHANNELS, Indexes, Retriever

    cfg = {**cfg, "cv": {**cfg["cv"], "n_folds": 5}, "retrieval": {**cfg["retrieval"], "channels": list(CHANNELS)}}
    idx = Indexes(cfg)
    retr = Retriever(idx, cfg)
    out = {}
    for name, st in [("full", {}), ("no bm25", {"drop_channels": ("bm25",)}), ("no char", {"drop_channels": ("char",)}),
                     ("no dense", {"drop_channels": ("dense",)}), ("no memory", {"drop_channels": ("memory",)}),
                     ("no spell variant", {"spell": False}), ("no abbreviation variant", {"abbr": False})]:
        res = run_retrieval_cv(cfg, st, idx, retr)
        res.pop("_records")
        out[name] = {"cv": res["cv"], "long_tail": res["long_tail"]}
        log.info("%s: %s", name, {s: res["cv"][s]["R@50"] for s in SYSTEMS})
    (Path(cfg["paths"]["reports"]) / "m1_results.json").write_text(json.dumps(out, indent=1))
