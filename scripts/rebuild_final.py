"""Recovery driver: rebuild the final predict artifacts from the surviving out-of-fold tables.

The v1 out-of-fold candidates, cross-encoder scores and LambdaRank tables in artifacts/cv survived the
laptop loss; only the large artifacts (indexes, final cross-encoders, memory, LambdaRank boosters) were lost.
This script rebuilds exactly those, with the same code path and seed as `termmap train`, without recomputing CV:

  python scripts/rebuild_final.py index        # build-index (both systems), dictionary stamp checked
  python scripts/rebuild_final.py memory       # fit_final: abbreviations, training memory, stats, fallbacks
  python scripts/rebuild_final.py ce ICD10CM   # final cross-encoder on ALL rows (epochs = CV best epoch)
  python scripts/rebuild_final.py ltr          # out-of-fold LambdaRank re-measure + final boosters
  python scripts/rebuild_final.py cooc         # co-occurrence pairs used by coupling
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from termmap.config import load_config, set_seeds  # noqa: E402
from termmap.io import ICD, SNOMED, SYSTEMS  # noqa: E402

log = logging.getLogger("rebuild")


def step_index(cfg: dict) -> None:
    from termmap.retrieval.engine import _stamp, build_index

    out = Path(cfg["paths"]["artifacts"]) / "index"
    old = json.loads((out / "manifest.json").read_text()) if (out / "manifest.json").exists() else None
    if old is not None:
        if old["stamp"] != _stamp(cfg):
            raise SystemExit(f"dictionary files differ from the v1 build: {old['stamp']} vs {_stamp(cfg)}")
        (out / "manifest_v1.json").write_text(json.dumps(old, indent=2))
    build_index(cfg, force=True)


def step_memory(cfg: dict) -> None:
    from termmap.pipeline import fit_final
    from termmap.retrieval.engine import Indexes, Retriever

    idx = Indexes(cfg)
    fit_final(cfg, idx, Retriever(idx, cfg))


def step_ce(cfg: dict, s: str) -> None:
    """Same as the `final` block of rerank.train_ce.run (fold checkpoints are not needed for it)."""
    import torch

    from termmap.rerank.cross_encoder import CandidateText, CrossEncoder, build_training_pairs, icd_children_map

    art = Path(cfg["paths"]["artifacts"])
    cv_dir, model_dir = art / "cv", art / "model"
    rc, seed = cfg["rerank"], cfg["seed"]
    rows = pd.read_parquet(cv_dir / "rows.parquet")
    srows = rows[rows.coding_system == s]
    cands = pd.read_parquet(cv_dir / f"candidates_{s}.parquet")
    table = pd.read_parquet(art / "index" / f"{s}_table.parquet")
    meta = pd.read_parquet(art / "index" / f"{s}_meta.parquet")
    ctext = CandidateText(s, table, meta)
    children = icd_children_map(set(meta.index)) if s == ICD else None
    best = [json.loads(p.read_text())["best_epoch"] for p in sorted(cv_dir.glob(f"ce_{s}_fold*_history.json"))]
    epochs = Counter(best).most_common(1)[0][0]
    torch.manual_seed(seed)
    a, b, y = build_training_pairs(srows, cands, ctext, s, rc["n_negatives"], rc["context_dropout"],
                                   rc["context_chars"], seed, children)
    ce = CrossEncoder(rc["cross_encoder"], max_len=rc["max_len"])
    log.info("%s final model: %d rows, %d pairs, %d epochs (CV best epochs %s)", s, len(srows), len(a), epochs, best)
    hist = ce.train_pairs(a, b, y, cfg, seed, val_fn=None, epochs=epochs)
    ce.save(model_dir / f"ce_{s}")
    (cv_dir / f"ce_{s}_final_history.json").write_text(json.dumps(hist, indent=1, default=float))


def step_ltr(cfg: dict) -> None:
    from termmap.rerank import train_ltr

    print(json.dumps(train_ltr.run(cfg), indent=1))


def step_ltr_noce(cfg: dict) -> None:
    """Fallback ranker WITHOUT cross-encoder features (used when a system's final CE is not available).

    Out-of-fold measure (same folds/params as train_ltr's "no cross-encoder" setting), final booster refit on all
    out-of-fold features, nested temperature and 98%-precision threshold, ICD hierarchy, coverage curve.
    """
    import joblib
    import numpy as np

    from termmap.confidence import coverage_curve, expected_calibration_error, threshold_for_precision
    from termmap.hierarchy import choose_argmax, choose_icd, fit_temperature, group_softmax
    from termmap.rerank.cross_encoder import CandidateText
    from termmap.rerank.features import FeatureBuilder
    from termmap.rerank.ltr import fit, importances
    from termmap.rerank.train_ltr import cv_ranker, oof_features

    art = Path(cfg["paths"]["artifacts"])
    cv_dir, model_dir = art / "cv", art / "model"
    rows_all = pd.read_parquet(cv_dir / "rows.parquet")
    margin = cfg["hierarchy"]["category_margin"]
    calib, summary = {}, {}
    for s in SYSTEMS:
        rows = rows_all[rows_all.coding_system == s]
        cands = pd.read_parquet(cv_dir / f"candidates_ce_{s}.parquet").reset_index(drop=True)
        table = pd.read_parquet(art / "index" / f"{s}_table.parquet")
        meta = pd.read_parquet(art / "index" / f"{s}_meta.parquet")
        fb = FeatureBuilder(s, table, meta, CandidateText(s, table, meta))
        feats = oof_features(s, rows, cands, fb, None)
        feats = feats.drop(columns=[c for c in ("ce", "ce_missing", "ce_rank", "ce_gap") if c in feats])
        oof, iters = cv_ranker(feats, len(rows), cfg, fb.categorical)
        feats = feats.assign(ltr=oof)
        feats.to_parquet(cv_dir / f"ltr_oof_noce_{s}.parquet", index=False)
        n_final = int(np.mean(iters))
        booster, _ = fit(feats.drop(columns=["ltr"]), cfg, fb.categorical, num_rounds=n_final)
        joblib.dump(booster, model_dir / f"ltr_{s}_noce.joblib")
        importances(booster).to_csv(Path(cfg["paths"]["reports"]) / f"feature_importance_noce_{s}.csv", index=False)
        # nested temperature -> probabilities; choices (ICD hierarchy) on every OOF row
        o = feats[["row", "code", "label", "ltr", "fold"]].copy()
        o["prob"] = np.nan
        for f in sorted(o["fold"].unique()):
            T = fit_temperature(o[o.fold != f], "ltr")
            m = o.fold == f
            o.loc[m, "prob"] = group_softmax(o[m], "ltr", T)
        gold = rows["code"].to_dict()
        conf, ok, ok_arg = [], [], []
        for r, g in o.groupby("row"):
            codes, probs = g["code"].tolist(), g["prob"].to_numpy()
            c = choose_icd(codes, probs, margin)[0] if s == ICD else choose_argmax(codes, probs)
            conf.append(float(probs[codes.index(c)]))
            ok.append(c == gold[r])
            ok_arg.append(choose_argmax(codes, probs) == gold[r])
        conf, ok = np.array(conf), np.array(ok)
        thr = threshold_for_precision(conf, ok, cfg["confidence"]["target_precision"])
        T_all = fit_temperature(o, "ltr")
        calib[s] = {"T": T_all, "threshold": thr if np.isfinite(thr) else None, "final_rounds": n_final}
        n = len(rows)
        summary[s] = {"rows": n, "ranker_top1": round(100 * sum(ok_arg) / n, 1),
                      "final_top1": round(100 * ok.sum() / n, 1), "best_iters": iters, "final_rounds": n_final,
                      "T": T_all, "ece": round(expected_calibration_error(conf, ok), 4),
                      "threshold_98": calib[s]["threshold"],
                      "coverage_at_98": round(100 * float((conf >= thr).mean()), 1) if np.isfinite(thr) else 0.0,
                      "curve": coverage_curve(conf, ok)}
        log.info("%s no-CE: %s", s, {k: v for k, v in summary[s].items() if k != "curve"})
    (model_dir / "noce.json").write_text(json.dumps(calib, indent=1))
    (Path(cfg["paths"]["reports"]) / "noce_measurements.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({s: {k: v for k, v in d.items() if k != "curve"} for s, d in summary.items()}, indent=1, default=float))


def step_cooc(cfg: dict) -> None:
    """Co-occurrence pairs exactly as train._save_m5 writes them (coupling.json/confidence.json are kept)."""
    rows = pd.read_parquet(Path(cfg["paths"]["artifacts"]) / "cv" / "rows.parquet")
    key = rows["entity"] + "\x00" + rows["chunk_text"]
    pr = pd.DataFrame({"key": key, "sys": rows["coding_system"], "code": rows["code"]})
    w = pr.pivot_table(index="key", columns="sys", values="code", aggfunc="first").dropna()
    w = w.rename(columns={ICD: "icd", SNOMED: "snomed"})[["icd", "snomed"]].reset_index(drop=True)
    w.to_parquet(Path(cfg["paths"]["artifacts"]) / "model" / "cooc_pairs.parquet", index=False)
    log.info("cooc pairs: %d", len(w))


def step_measure(cfg: dict) -> None:
    """Re-measure hierarchy, coupling, confidence and error tags from the (re-fitted) out-of-fold ranker scores."""
    from termmap.error_analysis import tag_errors
    from termmap.evaluate import evaluate_hierarchy, evaluate_m5

    out = {"hierarchy": evaluate_hierarchy(cfg)}
    out["m5"] = evaluate_m5(cfg)
    out["errors"] = {k: v for k, v in tag_errors(cfg).items() if k != "examples"}
    rep = Path(cfg["paths"]["reports"])
    (rep / "rebuilt_measurements.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"hierarchy": out["hierarchy"], "m5": {k: v for k, v in out["m5"].items() if k != "confidence"}},
                     indent=1, default=float))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    cfg = load_config()
    set_seeds(cfg["seed"])
    step, args = sys.argv[1], sys.argv[2:]
    if step == "ce":
        assert args and args[0] in SYSTEMS, f"system must be one of {SYSTEMS}"
        step_ce(cfg, args[0])
    else:
        {"index": step_index, "memory": step_memory, "ltr": step_ltr, "ltr_noce": step_ltr_noce, "cooc": step_cooc, "measure": step_measure}[step](cfg)
