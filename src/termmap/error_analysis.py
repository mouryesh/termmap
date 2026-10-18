"""M7 error analysis: one tag per out-of-fold error, rule-based, first matching rule wins.

Tags (in precedence order): not retrieved, probable label noise, over-specific, under-specific,
abbreviation, noisy term, unseen code, systems disagree, near string wrong concept, other ranking error.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from termmap.io import ICD, SNOMED, SYSTEMS
from termmap.normalize import level1, strip_semantic_tag, tokens
from termmap.rerank.features import qualifiers

TAGS = ("not retrieved", "probable label noise", "over-specific", "under-specific", "abbreviation",
        "noisy term", "unseen code", "systems disagree", "near string, wrong concept", "other ranking error")
_SHORT = re.compile(r"^[a-z]{2,5}$")


def _title(meta: pd.DataFrame, code: str, system: str) -> str:
    t = meta.at[code, "title"] if code in meta.index else ""
    t = level1(t) if isinstance(t, str) else ""
    return strip_semantic_tag(t) if system == SNOMED else t


def tag_errors(cfg: dict) -> dict:
    from termmap.retrieval.dense import Encoder

    art = Path(cfg["paths"]["artifacts"])
    rows = pd.read_parquet(art / "cv" / "rows.parquet")
    vocab = set(json.loads((art / "index" / "vocab.json").read_text()))
    abbr_short = set(json.loads((art / "model" / "abbreviations.json").read_text()))
    metas = {s: pd.read_parquet(art / "index" / f"{s}_meta.parquet") for s in SYSTEMS}
    preds, pools = {}, {}
    for s in SYSTEMS:
        oof = pd.read_parquet(art / "cv" / f"ltr_oof_{s}.parquet")[["row", "code", "label", "ltr"]]
        top = oof.sort_values(["row", "ltr", "code"], ascending=[True, False, True]).drop_duplicates("row")
        preds[s] = top.set_index("row")["code"]
        pools[s] = oof.groupby("row")["label"].max()
    key = rows["entity"] + "\x00" + rows["chunk_text"]
    partner = {}
    for _, g in rows.groupby(key):
        if len(g) == 2:
            a, b = g.index
            partner[a], partner[b] = b, a
    enc = Encoder(str(art / "index" / "sapbert"))
    recs = []
    for s in SYSTEMS:
        r = rows[rows.coding_system == s]
        seen = np.zeros(len(r), bool)
        for f in sorted(r.fold.unique()):
            trc = set(r.loc[r.fold != f, "code"])
            m = (r.fold == f).values
            seen[m] = r.loc[m, "code"].isin(trc).values
        seen = pd.Series(seen, index=r.index)
        for rid, row in r.iterrows():
            pred = preds[s].get(rid, "")
            if pred == row["code"]:
                continue
            recs.append({"row": rid, "system": s, "term": row["term_norm"], "entity": row["entity"],
                         "gold": row["code"], "gold_desc": _title(metas[s], row["code"], s),
                         "pred": pred, "pred_desc": _title(metas[s], pred, s) if pred else "",
                         "in_pool": bool(pools[s].get(rid, 0)), "seen": bool(seen[rid])})
    E = pd.DataFrame(recs)
    texts = sorted(set(E["term"]) | set(E["gold_desc"]) | set(E["pred_desc"]))
    V = dict(zip(texts, enc.encode(texts)))
    cos = lambda a, b: float(V[a] @ V[b]) if a and b else 0.0
    tags = []
    for e in E.itertuples():
        t, gd, pd_ = e.term, e.gold_desc, e.pred_desc
        sim_gold = max(fuzz.token_set_ratio(t, gd), 100 * cos(t, gd))
        sim_pred = max(fuzz.token_set_ratio(t, pd_), 100 * cos(t, pd_))
        tq, gq, pq = qualifiers(t), qualifiers(gd), qualifiers(pd_)
        words = tokens(t)
        other = partner.get(e.row)
        if not e.in_pool:
            tag = "not retrieved"
        elif sim_gold < 45 and sim_pred >= 75:
            tag = "probable label noise"
        elif (e.system == ICD and e.pred.startswith(e.gold) and len(e.pred) > len(e.gold)) or (pq - tq and not gq - tq and len(pq) > len(gq)):
            tag = "over-specific"
        elif (e.system == ICD and e.gold.startswith(e.pred) and len(e.gold) > len(e.pred)) or (gq - pq and len(gq) > len(pq)) \
                or ("unspecified" in pd_ and "unspecified" not in gd and e.gold[:3] == e.pred[:3]):
            tag = "under-specific"
        elif any(w in abbr_short or (_SHORT.match(w) and w not in vocab) for w in words):
            tag = "abbreviation"
        elif any(w.isalpha() and len(w) > 5 and w not in vocab for w in words) or "<br" in e.entity.lower():
            tag = "noisy term"
        elif not e.seen:
            tag = "unseen code"
        elif other is not None and preds[rows.at[other, "coding_system"]].get(other) == rows.at[other, "code"]:
            tag = "systems disagree"
        elif fuzz.token_set_ratio(t, pd_) >= 70:
            tag = "near string, wrong concept"
        else:
            tag = "other ranking error"
        tags.append(tag)
    E["tag"] = tags
    E.to_parquet(art / "cv" / "error_tags.parquet", index=False)
    n_rows = {s: int((rows.coding_system == s).sum()) for s in SYSTEMS}
    summary = {s: {t: int(((E.system == s) & (E.tag == t)).sum()) for t in TAGS} for s in SYSTEMS}
    examples = {}
    for t in TAGS:
        ex = E[E.tag == t].sort_values(["system", "row"]).groupby("system").head(2).head(3)
        examples[t] = ex[["system", "term", "gold", "gold_desc", "pred", "pred_desc"]].to_dict("records")
    return {"rows": n_rows, "errors": {s: int((E.system == s).sum()) for s in SYSTEMS},
            "tags": summary, "examples": examples}
