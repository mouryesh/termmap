"""Training-set assembly: long rows per system, invalid-code filtering, in-memory CV folds."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from termmap.io import ICD, SNOMED, group_key, load_train
from termmap.normalize import level1, strip_semantic_tag

log = logging.getLogger(__name__)


def build_rows(cfg: dict, valid_codes: dict[str, set[str]]) -> pd.DataFrame:
    """All training rows (subconcept ignored) with normalized term/display and a CV fold.

    Rows whose gold code is not in the dictionary are dropped and logged (decision D4).
    Folds come from GroupKFold over the in-memory canonical chunk key; no ID is stored.
    """
    df = load_train(cfg["paths"]["train"]).drop(columns=["subconcept"])
    ok = np.array([c in valid_codes[s] for s, c in zip(df.coding_system, df.code)])
    for s, c, e in df.loc[~ok, ["coding_system", "code", "entity"]].itertuples(index=False):
        log.warning("dropping row with invalid %s code %r", s, c)
    df = df[ok].reset_index(drop=True)
    df["term_norm"] = df["entity"].map(level1)
    df["display_norm"] = df["display"].map(level1).map(strip_semantic_tag)
    groups = [group_key(c, i) for i, c in enumerate(df["chunk_text"])]
    df["fold"] = -1
    gkf = GroupKFold(n_splits=cfg["cv"]["n_folds"])
    for f, (_, te) in enumerate(gkf.split(df, groups=groups)):
        df.loc[te, "fold"] = f
    df.attrs["groups"] = groups  # in memory only
    return df


def heldout_codes(df: pd.DataFrame, system: str, frac: float, seed: int) -> set[str]:
    """Deterministic random 20% of a system's distinct gold codes, for the long-tail split."""
    codes = np.array(sorted(df.loc[df.coding_system == system, "code"].unique()))
    rng = np.random.default_rng(seed)
    return set(rng.choice(codes, size=int(round(frac * len(codes))), replace=False).tolist())


def ce_scope(rc: dict, system: str) -> tuple[int, bool]:
    """(fused top-N, score ICD siblings?) for the cross-encoder; ce_top may be an int or a per-system map."""
    top = rc["ce_top"][system] if isinstance(rc["ce_top"], dict) else rc["ce_top"]
    return int(top), bool(rc.get("ce_siblings", False)) and system == ICD


def candidate_frame(system: str, row_ids: list[int], gold: list[str | None], cands: list, ce_top: int,
                    ce_siblings: bool = False) -> pd.DataFrame:
    """Flatten Stage 2 Candidates into one row per (training row, candidate code).

    in_ce marks what the cross-encoder scores: fused top-`ce_top` plus embedded codes, plus (with
    ce_siblings, ICD only; audit A4) every sibling expanded from the fused top-20.
    """
    recs = []
    for rid, g, c in zip(row_ids, gold, cands):
        fused_rank = {code: i for i, (code, _) in enumerate(c.fused, start=1)}
        rrf_score = dict(c.fused)
        for code in c.pool:
            fr = fused_rank.get(code)
            r = c.ranks.get(code, {})
            recs.append({
                "row": rid, "code": code, "gold": int(code == g),
                "fused_rank": fr if fr is not None else 999, "rrf": rrf_score.get(code, 0.0),
                "rank_char": r.get("char", 999), "rank_dense": r.get("dense", 999), "rank_memory": r.get("memory", 999),
                "expanded": int(fr is None and code not in c.embedded), "embedded": int(code in c.embedded),
                "in_ce": int((fr is not None and fr <= ce_top) or code in c.embedded
                             or (ce_siblings and fr is None)),
            })
    out = pd.DataFrame(recs)
    out["system"] = system
    return out


def oof_candidates(cfg: dict, df: pd.DataFrame, idx, retr) -> dict[str, pd.DataFrame]:
    """Out-of-fold Stage 2 candidate pools for every training row (memory/abbreviations from other folds)."""
    from termmap.evaluate import memory_for
    from termmap.normalize import embedded_codes, mine_abbreviations, query_variants

    out: dict[str, list[pd.DataFrame]] = {ICD: [], SNOMED: []}
    for f in sorted(df["fold"].unique()):
        tr, te = df[df.fold != f], df[df.fold == f]
        abbr = mine_abbreviations(tr["term_norm"].tolist(), tr["display_norm"].tolist(),
                                  cfg.get("normalize", {}).get("abbr_min_support", 1))
        for s in (ICD, SNOMED):
            t = te[te.coding_system == s]
            mem = memory_for(tr[tr.coding_system == s], retr)
            variants = [query_variants(x, idx.spell, abbr) for x in t["term_norm"]]
            per_rows = retr.channel_rankings(s, variants, mem)
            cands = []
            for raw, per in zip(t["entity"], per_rows):
                ei, es = embedded_codes(raw, idx.sys[ICD].codes, idx.sys[SNOMED].codes,
                                        cfg.get("normalize", {}).get("undotted_codes", False))
                cands.append(retr.fuse(s, per, ei if s == ICD else es))
            fr = candidate_frame(s, t.index.tolist(), t["code"].tolist(), cands, *ce_scope(cfg["rerank"], s))
            fr["fold"] = int(f)
            out[s].append(fr)
        log.info("candidates fold %d done", f)
    return {s: pd.concat(v, ignore_index=True) for s, v in out.items()}
