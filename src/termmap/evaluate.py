"""Cross-validated evaluation of the pipeline stages (M1: Stage 0 + Stage 2 retrieval)."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from termmap.build_dataset import build_rows, heldout_codes
from termmap.io import ICD, SNOMED, SYSTEMS
from termmap.normalize import embedded_codes, mine_abbreviations, query_variants
from termmap.retrieval.engine import CHANNELS, Indexes, Retriever
from termmap.retrieval.memory import MemoryIndex

log = logging.getLogger(__name__)


def memory_for(rows: pd.DataFrame, retr: Retriever) -> MemoryIndex:
    """Memory over training terms and their gold displays (one system's rows)."""
    texts = rows["term_norm"].tolist() + rows["display_norm"].tolist()
    codes = rows["code"].tolist() * 2
    keep = [i for i, t in enumerate(texts) if t]
    texts, codes = [texts[i] for i in keep], [codes[i] for i in keep]
    return MemoryIndex(texts, codes, retr.embed(texts), retr.k)


def _rank_of(code: str, ranking) -> int:
    for i, (c, _) in enumerate(ranking, start=1):
        if c == code:
            return i
    return 10**6


def retrieval_records(df_test: pd.DataFrame, system: str, retr: Retriever, memory: MemoryIndex,
                      abbr: dict[str, str], idx: Indexes, settings: dict) -> list[dict]:
    """Per test row: gold rank in each channel and in the fused list, plus pool membership."""
    spell = idx.spell if settings.get("spell", True) else None
    variants = [query_variants(t, spell, abbr if settings.get("abbr", True) else None)
                for t in df_test["term_norm"]]
    channels = tuple(c for c in retr.channels if c not in settings.get("drop_channels", ()))
    per_rows = retr.channel_rankings(system, variants, memory, channels=channels)
    recs = []
    for (_, r), per, v in zip(df_test.iterrows(), per_rows, variants):
        emb_icd, emb_sct = embedded_codes(r["entity"], idx.sys[ICD].codes, idx.sys[SNOMED].codes)
        cand = retr.fuse(system, per, emb_icd if system == ICD else emb_sct)
        rec = {"system": system, "code": r["code"], "fused_rank": _rank_of(r["code"], cand.fused),
               "in_pool": r["code"] in set(cand.pool),
               "top1": cand.fused[0][0] if cand.fused else "", "n_variants": len(v),
               "has_spell": "spell" in v, "has_abbr": "abbr" in v}
        for ch in per:
            rec[f"rank_{ch}"] = _rank_of(r["code"], per[ch])
        recs.append(rec)
    return recs


def summarize(recs: pd.DataFrame) -> dict:
    """Top-1, recall@10/30/50/80, pool recall and per-channel recall@50 / top-1."""
    out = {"n": len(recs)}
    for k in (1, 10, 30, 50, 80):
        out[f"R@{k}"] = round(100 * (recs["fused_rank"] <= k).mean(), 1)
    out["pool_recall@K+sib"] = round(100 * recs["in_pool"].mean(), 1)
    for ch in CHANNELS:
        col = f"rank_{ch}"
        if col in recs:
            out[f"{ch}_top1"] = round(100 * (recs[col] == 1).mean(), 1)
            out[f"{ch}_R@50"] = round(100 * (recs[col] <= 50).mean(), 1)
    return out


def run_retrieval_cv(cfg: dict, settings: dict | None = None, idx: Indexes | None = None,
                     retr: Retriever | None = None) -> dict:
    """5-fold grouped CV of Stage 0 + Stage 2, plus the held-out-code long-tail split."""
    settings = settings or {}
    idx = idx or Indexes(cfg)
    retr = retr or Retriever(idx, cfg)
    df = build_rows(cfg, {s: idx.sys[s].codes for s in SYSTEMS})
    results: dict = {"settings": settings}
    all_recs = []
    t0 = time.time()
    for f in sorted(df["fold"].unique()):
        tr, te = df[df.fold != f], df[df.fold == f]
        abbr = mine_abbreviations(tr["term_norm"].tolist(), tr["display_norm"].tolist())
        for s in SYSTEMS:
            mem = memory_for(tr[tr.coding_system == s], retr)
            recs = retrieval_records(te[te.coding_system == s], s, retr, mem, abbr, idx, settings)
            for r in recs:
                r["fold"] = int(f)
            all_recs.extend(recs)
        log.info("fold %d done (%.0fs)", f, time.time() - t0)
    R = pd.DataFrame(all_recs)
    results["cv"] = {s: summarize(R[R.system == s]) for s in SYSTEMS}
    results["cv"]["min_top1"] = min(results["cv"][s]["R@1"] for s in SYSTEMS)
    # long-tail: 20% of gold codes never seen in memory / abbreviation mining
    lt = {}
    groups = np.array(df.attrs["groups"])
    for s in SYSTEMS:
        held = heldout_codes(df, s, 0.2, cfg["seed"])
        test_mask = (df.coding_system == s) & df.code.isin(held)
        test_groups = set(groups[test_mask.values])
        train_mask = ~df.code.isin(held) & ~pd.Series(groups).isin(test_groups).values
        tr, te = df[train_mask], df[test_mask]
        abbr = mine_abbreviations(tr["term_norm"].tolist(), tr["display_norm"].tolist())
        mem = memory_for(tr[tr.coding_system == s], retr)
        lt[s] = summarize(pd.DataFrame(retrieval_records(te, s, retr, mem, abbr, idx, settings)))
    results["long_tail"] = lt
    results["seconds"] = round(time.time() - t0, 1)
    results["_records"] = R
    return results


# ---------------------------------------------------------------- M4: ICD hierarchy
MARGIN_GRID = (0.0, 0.02, 0.05, 0.1, 0.2)


def _icd_decisions(df: pd.DataFrame, gold: dict, margin: float | None) -> pd.DataFrame:
    """Per term: chosen code (hierarchy if margin is not None, else argmax) and correctness."""
    from termmap.hierarchy import choose_argmax, choose_icd

    out = []
    for r, g in df.groupby("row", sort=True):
        codes, probs = g["code"].tolist(), g["prob"].to_numpy()
        if margin is None:
            code, info = choose_argmax(codes, probs), {}
        else:
            code, info = choose_icd(codes, probs, margin)
        out.append({"row": r, "code": code, "exact": int(code == gold[r]), "cat": int(code[:3] == gold[r][:3]),
                    "top_category": info.get("top_category", code[:3]), "changed": info.get("changed_from_argmax", 0)})
    return pd.DataFrame(out)


def evaluate_hierarchy(cfg: dict) -> dict:
    """Nested over folds: T and margin fitted on the other folds' OOF LambdaRank scores, applied to this fold."""
    from termmap.hierarchy import fit_temperature, group_softmax

    art = Path(cfg["paths"]["artifacts"]) / "cv"
    rows = pd.read_parquet(art / "rows.parquet")
    res: dict = {}
    calib: dict = {}
    for s in SYSTEMS:
        oof = pd.read_parquet(art / f"ltr_oof_{s}.parquet")
        gold = rows.loc[rows.coding_system == s, "code"].to_dict()
        n = len(gold)
        per_fold, decisions = [], []
        for f in sorted(oof["fold"].unique()):
            tr, te = oof[oof.fold != f].copy(), oof[oof.fold == f].copy()
            T = fit_temperature(tr, "ltr")
            tr["prob"], te["prob"] = group_softmax(tr, "ltr", T), group_softmax(te, "ltr", T)
            rec = {"fold": int(f), "T": T}
            if s == ICD:
                margin = max(MARGIN_GRID, key=lambda m: (_icd_decisions(tr, gold, m)["exact"].mean(), -m))
                d_arg = _icd_decisions(te, gold, None)
                d_spec = _icd_decisions(te, gold, cfg["hierarchy"]["category_margin"])
                d_tuned = _icd_decisions(te, gold, margin)
                rec.update(margin=margin, argmax=int(d_arg.exact.sum()), spec=int(d_spec.exact.sum()),
                           tuned=int(d_tuned.exact.sum()), cat_argmax=int(d_arg.cat.sum()), cat_tuned=int(d_tuned.cat.sum()),
                           changed=int(d_tuned.changed.sum()))
                m = d_tuned.merge(d_arg[["row", "exact"]], on="row", suffixes=("", "_arg"))
                rec["fixed"] = int(((m.exact == 1) & (m.exact_arg == 0)).sum())
                rec["broken"] = int(((m.exact == 0) & (m.exact_arg == 1)).sum())
                decisions.append(d_tuned)
            per_fold.append(rec)
        summ = {"rows": n, "T_per_fold": [p["T"] for p in per_fold]}
        if s == ICD:
            tot = lambda k: sum(p[k] for p in per_fold)
            summ.update({
                "exact_argmax": round(100 * tot("argmax") / n, 1),
                "exact_hierarchy_margin_0.05": round(100 * tot("spec") / n, 1),
                "exact_hierarchy_tuned_margin": round(100 * tot("tuned") / n, 1),
                "category_acc_argmax": round(100 * tot("cat_argmax") / n, 1),
                "category_acc_hierarchy": round(100 * tot("cat_tuned") / n, 1),
                "margins_chosen": [p["margin"] for p in per_fold],
                "decisions_changed": tot("changed"), "fixed": tot("fixed"), "broken": tot("broken"),
            })
            pd.concat(decisions).to_parquet(art / "hierarchy_decisions_ICD10CM.parquet", index=False)
        res[s] = summ
        # final calibration from all OOF scores (used at predict time)
        allp = oof.copy()
        T_all = fit_temperature(allp, "ltr")
        calib[s] = {"T": T_all}
        if s == ICD:
            allp["prob"] = group_softmax(allp, "ltr", T_all)
            calib[s]["margin"] = max(MARGIN_GRID, key=lambda m: (_icd_decisions(allp, gold, m)["exact"].mean(), -m))
    res["calibration"] = calib
    return res


# ---------------------------------------------------------------- M5: coupling + confidence
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def _top(g: pd.DataFrame, n: int = 5) -> list[tuple[str, float]]:
    g = g.sort_values(["prob", "code"], ascending=[False, True]).head(n)
    return list(zip(g["code"], g["prob"]))


def evaluate_m5(cfg: dict) -> dict:
    """Nested over folds: T, co-occurrence and lambda from the other folds; confidence on all OOF rows."""
    from termmap.confidence import coverage_curve, expected_calibration_error, threshold_for_precision
    from termmap.coupling import CodeEmbeddings, CooccurrenceTable, couple
    from termmap.hierarchy import choose_argmax, choose_icd, fit_temperature, group_softmax
    from termmap.retrieval.dense import Encoder

    art = Path(cfg["paths"]["artifacts"])
    rows = pd.read_parquet(art / "cv" / "rows.parquet")
    margin = cfg["hierarchy"]["category_margin"]
    oof = {s: pd.read_parquet(art / "cv" / f"ltr_oof_{s}.parquet")[["row", "code", "label", "ltr", "fold"]] for s in SYSTEMS}
    for s in SYSTEMS:  # nested temperature
        oof[s]["prob"] = np.nan
        for f in sorted(oof[s]["fold"].unique()):
            T = fit_temperature(oof[s][oof[s].fold != f], "ltr")
            m = oof[s].fold == f
            oof[s].loc[m, "prob"] = group_softmax(oof[s][m], "ltr", T)
    groups = {s: dict(tuple(oof[s].groupby("row"))) for s in SYSTEMS}
    # independent choices for every row
    indep: dict[str, dict[int, tuple[str, float]]] = {s: {} for s in SYSTEMS}
    for s in SYSTEMS:
        for r, g in groups[s].items():
            codes, probs = g["code"].tolist(), g["prob"].to_numpy()
            c = choose_icd(codes, probs, margin)[0] if s == ICD else choose_argmax(codes, probs)
            indep[s][r] = (c, float(probs[codes.index(c)]))
    gold = rows["code"].to_dict()
    # paired instances (same raw entity + chunk text in both systems; always the same fold)
    key = rows["entity"] + "\x00" + rows["chunk_text"]
    pr = pd.DataFrame({"key": key, "sys": rows["coding_system"], "row": rows.index, "fold": rows["fold"]})
    wide = pr.pivot_table(index="key", columns="sys", values="row", aggfunc="first").dropna().astype(int)
    wide["fold"] = rows.loc[wide[ICD].values, "fold"].values
    wide = wide[wide[ICD].isin(groups[ICD].keys()) & wide[SNOMED].isin(groups[SNOMED].keys())]
    enc = Encoder(str(art / "index" / "sapbert"))
    metas = {s: pd.read_parquet(art / "index" / f"{s}_meta.parquet") for s in SYSTEMS}
    emb = CodeEmbeddings(enc, metas)
    tops = {r: _top(groups[ICD][r]) for r in wide[ICD]}
    topss = {r: _top(groups[SNOMED][r]) for r in wide[SNOMED]}
    emb.prime([(ICD, c) for t in tops.values() for c, _ in t] + [(SNOMED, c) for t in topss.values() for c, _ in t])

    def run_pairs(sub: pd.DataFrame, cooc, lam: float, mode: str) -> tuple[int, int]:
        ci = cs = 0
        for ri, rs in zip(sub[ICD], sub[SNOMED]):
            ic, sc, _, _ = couple(tops[ri], topss[rs], lam, cooc, emb)
            if ic == choose_argmax([c for c, _ in tops[ri]], np.array([p for _, p in tops[ri]])):
                ic = indep[ICD][ri][0]  # coupling agrees with argmax: keep the hierarchy's choice
            if mode == "icd_only":
                sc = indep[SNOMED][rs][0]
            ci += ic == gold[ri]
            cs += sc == gold[rs]
        return ci, cs

    res: dict = {"paired_instances": len(wide)}
    for mode in ("joint", "icd_only"):
        tot_i = tot_s = base_i = base_s = 0
        chosen = []
        for f in sorted(wide["fold"].unique()):
            tr, te = wide[wide.fold != f], wide[wide.fold == f]
            cooc = CooccurrenceTable(list(zip(rows.loc[tr[ICD], "code"], rows.loc[tr[SNOMED], "code"])))
            # lambda chosen without the held-out fold; inner co-occurrence excludes each inner fold
            scores = {}
            lambdas = tuple(cfg["coupling"].get("lambdas", LAMBDAS))
            for l in lambdas:
                tot = 0
                for g in sorted(tr["fold"].unique()):
                    inner_tr, inner_te = tr[tr.fold != g], tr[tr.fold == g]
                    c_in = CooccurrenceTable(list(zip(rows.loc[inner_tr[ICD], "code"], rows.loc[inner_tr[SNOMED], "code"])))
                    tot += sum(run_pairs(inner_te, c_in, l, mode))
                scores[l] = tot
            lam = max(lambdas, key=lambda l: (scores[l], -l))
            chosen.append(lam)
            a, b = run_pairs(te, cooc, lam, mode)
            a0, b0 = run_pairs(te, cooc, 0.0, mode)
            tot_i, tot_s, base_i, base_s = tot_i + a, tot_s + b, base_i + a0, base_s + b0
        n = len(wide)
        unpaired = [r for r in indep[ICD] if r not in set(wide[ICD])]
        unp_ok = sum(indep[ICD][r][0] == gold[r] for r in unpaired)
        n_icd = int((rows.coding_system == ICD).sum())
        res[mode + "_overall"] = {"icd_all_rows": round(100 * (tot_i + unp_ok) / n_icd, 1),
                                  "icd_all_rows_lambda0": round(100 * (base_i + unp_ok) / n_icd, 1)}
        res[mode] = {"lambda_per_fold": chosen, "icd_acc": round(100 * tot_i / n, 1), "snomed_acc": round(100 * tot_s / n, 1),
                     "icd_acc_lambda0": round(100 * base_i / n, 1), "snomed_acc_lambda0": round(100 * base_s / n, 1)}
        log.info("coupling %s: %s", mode, res[mode])
    # full-data lambda per mode for the final model
    cooc_all = CooccurrenceTable(list(zip(rows.loc[wide[ICD], "code"], rows.loc[wide[SNOMED], "code"])))
    res["cooc_pairs"] = len(cooc_all.n_pair)
    # confidence on every OOF row (independent choices)
    conf_res = {}
    for s in SYSTEMS:
        rr = sorted(indep[s])
        conf = np.array([indep[s][r][1] for r in rr])
        ok = np.array([indep[s][r][0] == gold[r] for r in rr])
        n_all = int((rows.coding_system == s).sum())
        thr = threshold_for_precision(conf, ok, cfg["confidence"]["target_precision"])
        acc_mask = conf >= thr
        conf_res[s] = {"rows_scored": len(rr), "rows_total": n_all, "ece": round(expected_calibration_error(conf, ok), 4),
                       "threshold_98": thr if np.isfinite(thr) else None,
                       "coverage_at_98": round(100 * acc_mask.mean(), 1),
                       "precision_at_98": round(100 * ok[acc_mask].mean(), 1) if acc_mask.any() else None,
                       "curve": coverage_curve(conf, ok)}
    res["confidence"] = conf_res
    return res


# ---------------------------------------------------------------- A9: weighted RRF channel weights
WEIGHT_GRID = {"bm25": (0.0, 0.25, 0.5, 0.75, 1.0), "char": (0.5, 1.0), "dense": (1.0, 1.5, 2.0),
               "memory": (1.0, 1.5, 2.0)}


def tune_rrf_weights(cfg: dict, idx: Indexes | None = None, retr: Retriever | None = None) -> dict:
    """Out-of-fold channel rankings (3 folds, all rows), then a grid search of per-system RRF weights
    maximizing recall@80 (ties: recall@30, then smaller BM25 weight). Equal weights are in the grid."""
    import itertools

    from termmap.retrieval.fusion import rrf

    idx = idx or Indexes(cfg)
    retr = retr or Retriever(idx, cfg)
    rows = pd.read_parquet(Path(cfg["paths"]["artifacts"]) / "cv" / "rows.parquet")
    chans = ("bm25", "char", "dense", "memory")
    per_sys: dict[str, list] = {s: [] for s in SYSTEMS}
    for f in sorted(rows["fold"].unique()):
        tr, te = rows[rows.fold != f], rows[rows.fold == f]
        abbr = mine_abbreviations(tr["term_norm"].tolist(), tr["display_norm"].tolist())
        for s in SYSTEMS:
            t = te[te.coding_system == s]
            mem = memory_for(tr[tr.coding_system == s], retr)
            variants = [query_variants(x, idx.spell, abbr) for x in t["term_norm"]]
            per_rows = retr.channel_rankings(s, variants, mem, channels=chans)
            per_sys[s].extend((g, [[c for c, _ in p[ch]] for ch in chans]) for g, p in zip(t["code"], per_rows))
        log.info("A9 channel rankings: fold %d done", f)
    out: dict = {}
    for s in SYSTEMS:
        data = per_sys[s]
        log.info("A9 %s: %d rows", s, len(data))
        results = []
        for w in itertools.product(*(WEIGHT_GRID[c] for c in chans)):
            hits = np.zeros(3)
            for gold, ranks in data:
                fused = [c for c, _ in rrf([[(c, 0.0) for c in r] for r in ranks], cfg["retrieval"]["rrf_k"], w)[:80]]
                pos = fused.index(gold) + 1 if gold in fused else 10**6
                hits += [pos <= 10, pos <= 30, pos <= 80]
            results.append((tuple(round(100 * h / len(data), 2) for h in hits), w))
        best = max(results, key=lambda r: (r[0][2], r[0][1], -r[1][0]))
        equal = next(r for r in results if r[1] == (1.0, 1.0, 1.0, 1.0))
        nobm25 = next(r for r in results if r[1] == (0.0, 1.0, 1.0, 1.0))
        out[s] = {"rows": len(data), "best_weights": dict(zip(chans, best[1])), "best_R@10/30/80": best[0],
                  "equal_R@10/30/80": equal[0], "no_bm25_R@10/30/80": nobm25[0]}
        log.info("A9 %s: %s", s, out[s])
    return out
