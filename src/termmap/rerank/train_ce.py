"""M2 driver: 3-fold out-of-fold cross-encoder scores per system, then final models on all rows.

Per-fold scores are checkpointed to artifacts/cv so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from termmap.io import ICD, SNOMED, group_key
from termmap.rerank.cross_encoder import (CandidateText, CrossEncoder, build_training_pairs, icd_children_map,
                                          score_candidates, top1_accuracy)
from termmap.retrieval.engine import load_dictionaries

log = logging.getLogger(__name__)


def _val_split(rows: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out `frac` of the canonical-chunk groups (computed in memory) for early stopping."""
    keys = np.array([group_key(c, i) for i, c in zip(rows.index, rows["chunk_text"])])
    uniq = np.array(sorted(set(keys)))
    rng = np.random.default_rng(seed)
    val_groups = set(rng.choice(uniq, size=max(1, int(frac * len(uniq))), replace=False).tolist())
    m = np.isin(keys, list(val_groups))
    return rows[~m], rows[m]


def run(cfg: dict, systems: tuple[str, ...] = (ICD, SNOMED), final: bool = True) -> dict:
    art = Path(cfg["paths"]["artifacts"])
    cv_dir, model_dir = art / "cv", art / "model"
    rows = pd.read_parquet(cv_dir / "rows.parquet")
    metas = {}
    for s in systems:
        mp = art / "index" / f"{s}_meta.parquet"
        metas[s] = pd.read_parquet(mp) if mp.exists() else load_dictionaries(cfg)[s].meta
    rc, seed = cfg["rerank"], cfg["seed"]
    summary: dict = {}
    for s in systems:
        cands = pd.read_parquet(cv_dir / f"candidates_{s}.parquet")
        table = pd.read_parquet(art / "index" / f"{s}_table.parquet")
        ctext = CandidateText(s, table, metas[s])
        children = icd_children_map(set(metas[s].index)) if s == ICD else None
        neg_source = rc.get("negatives", "fused")
        ltr_scores = pd.read_parquet(Path(rc["negatives_ltr_oof_dir"]) / f"ltr_oof_{s}.parquet") \
            if neg_source == "ltr" else None
        listwise = rc.get("loss", "bce") == "listwise"
        srows = rows[rows.coding_system == s]
        hist = {}
        for f in sorted(srows["fold"].unique()):
            out = cv_dir / f"ce_{s}_fold{f}.parquet"
            if out.exists():
                need = cands[cands.row.isin(srows.index[srows.fold == f]) & (cands.in_ce == 1)]
                have = pd.read_parquet(out).dropna(subset=["ce_ctx"])
                missing = len(need.merge(have[["row", "code"]], on=["row", "code"], how="left", indicator=True)
                              .query("_merge == 'left_only'"))
                if missing == 0:
                    log.info("%s fold %d: checkpoint covers all %d scored candidates, skipping", s, f, len(need))
                    continue
                log.warning("%s fold %d: checkpoint misses %d candidates (candidate set changed); retraining", s, f, missing)
            torch.manual_seed(seed + int(f))
            tr_all, te = srows[srows.fold != f], srows[srows.fold == f]
            tr, va = _val_split(tr_all, rc["val_frac"], seed + int(f))
            built = build_training_pairs(tr, cands, ctext, s, rc["n_negatives"], rc["context_dropout"],
                                         rc["context_chars"], seed + int(f), children, neg_source, ltr_scores,
                                         return_groups=True)
            a, b, y, grp = built
            va_c = cands[cands.row.isin(va.index)].copy()
            ce = CrossEncoder(rc["cross_encoder"], max_len=rc["max_len"])

            def val_fn():
                va_c["s"] = score_candidates(ce, va, va_c, ctext, True, rc["context_chars"])
                return top1_accuracy(va_c, "s", len(va))

            log.info("%s fold %d: %d train rows, %d val rows, %d pairs", s, f, len(tr), len(va), len(a))
            h = ce.train_pairs(a, b, y, cfg, seed + int(f), val_fn=val_fn, groups=grp if listwise else None)
            h.update(train_rows=len(tr), val_rows=len(va), test_rows=len(te), groups=int(len(set(grp))),
                     negatives=neg_source, loss=rc.get("loss", "bce"))
            te_c = cands[cands.row.isin(te.index)].copy()
            te_c["ce_ctx"] = score_candidates(ce, te, te_c, ctext, True, rc["context_chars"])
            te_c["ce_noctx"] = score_candidates(ce, te, te_c, ctext, False, rc["context_chars"])
            te_c[["row", "code", "ce_ctx", "ce_noctx"]].to_parquet(out, index=False)
            hist[int(f)] = h
            (cv_dir / f"ce_{s}_fold{f}_history.json").write_text(json.dumps(h, indent=1))
            del ce
            torch.cuda.empty_cache()
        parts = [pd.read_parquet(cv_dir / f"ce_{s}_fold{f}.parquet") for f in sorted(srows["fold"].unique())]
        scores = pd.concat(parts, ignore_index=True)
        merged = cands.merge(scores, on=["row", "code"], how="left")
        merged.loc[merged.in_ce == 0, ["ce_ctx", "ce_noctx"]] = np.nan  # score only the configured CE set
        merged.to_parquet(cv_dir / f"candidates_ce_{s}.parquet", index=False)
        n = len(srows)
        summary[s] = {
            "rows": n,
            "fused_top1": round(100 * ((merged.gold == 1) & (merged.fused_rank == 1)).sum() / n, 1),
            "ce_top1_with_context": round(100 * top1_accuracy(merged, "ce_ctx", n), 1),
            "ce_top1_entity_only": round(100 * top1_accuracy(merged, "ce_noctx", n), 1),
            "ce_set_ceiling": round(100 * ((merged.gold == 1) & (merged.in_ce == 1)).sum() / n, 1),
        }
        log.info("%s CE summary: %s", s, summary[s])
        if final:
            best = [json.loads((cv_dir / f"ce_{s}_fold{f}_history.json").read_text())["best_epoch"]
                    for f in sorted(srows["fold"].unique())]
            if rc.get("final_epochs", "mode") == "median":  # v2 (audit A1): CV-chosen count, no cap
                epochs = int(np.ceil(np.median(best)))
            else:  # v1
                epochs = Counter(best).most_common(1)[0][0]
            dest = model_dir / f"ce_{s}"
            if not (dest / "config.json").exists():
                torch.manual_seed(seed)
                a, b, y, grp = build_training_pairs(srows, cands, ctext, s, rc["n_negatives"], rc["context_dropout"],
                                                    rc["context_chars"], seed, children, neg_source, ltr_scores,
                                                    return_groups=True)
                ce = CrossEncoder(rc["cross_encoder"], max_len=rc["max_len"])
                log.info("%s final model: %d pairs, %d epochs (CV best epochs %s)", s, len(a), epochs, best)
                hf = ce.train_pairs(a, b, y, cfg, seed, val_fn=None, epochs=epochs, groups=grp if listwise else None)
                (cv_dir / f"ce_{s}_final_history.json").write_text(json.dumps(hf, indent=1))
                ce.save(dest)
                del ce
                torch.cuda.empty_cache()
            summary[s]["final_epochs"] = epochs
    (cv_dir / "ce_summary.json").write_text(json.dumps(summary, indent=1))
    return summary
