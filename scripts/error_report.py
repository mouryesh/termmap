"""Error analysis from the out-of-fold LambdaRank scores (artifacts/cv/ltr_oof_*) and candidate pools.

Writes reports/error_analysis.md and reports/error_analysis.json. Prediction = LambdaRank argmax per row
(ties by code), i.e. before the ICD hierarchy/coupling steps. Error classes (a row can be in several):
  not_in_pool   gold code absent from the candidate pool (an indexing / retrieval miss)
  near_twin     in pool, wrong pick, char 3-4-gram TF-IDF cosine(pred title, gold title) >= 0.5
  abbreviation  a term token is a key of the mined abbreviation table
  unseen_code   gold code never appears in the other folds' training rows
Plus the label-noise ceiling: among rows with identical normalised terms, share of row pairs with the same code.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from termmap.io import ICD, SNOMED, SYSTEMS  # noqa: E402
from termmap.normalize import strip_semantic_tag, tokens  # noqa: E402

ART, REP = ROOT / "artifacts", ROOT / "reports"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""  # "" = v1 ranker with cross-encoder, "noce" = fallback
OOF = f"ltr_oof_{VARIANT}_" if VARIANT else "ltr_oof_"
SUFFIX = f"_{VARIANT}" if VARIANT else ""
N_EX = 20


def title(meta: pd.DataFrame, code: str, s: str) -> str:
    t = meta["title"].get(code, "") if code in meta.index else ""
    t = t if isinstance(t, str) else ""
    return strip_semantic_tag(t.lower()) if s == SNOMED else t.lower()


def tag(meta: pd.DataFrame, code: str) -> str:
    return meta["semantic_tag"].get(code, "") if code in meta.index else ""


def main() -> None:
    rows = pd.read_parquet(ART / "cv" / "rows.parquet")
    abbr = set(json.loads((ART / "model" / "abbreviations.json").read_text()))
    out, md = {}, [f"# Error analysis (out-of-fold, {'ranker without cross-encoder (submitted fallback)' if VARIANT else 'v1 ranker with cross-encoder'})", "",
                   "Source: `artifacts/cv/ltr_oof_*.parquet` (3-fold GroupKFold by chunk, LambdaRank out-of-fold "
                   "scores) and the candidate pools. Prediction = LambdaRank argmax per row (before ICD hierarchy and "
                   "coupling). Classes overlap; examples are the first rows in file order, not hand-picked.", ""]
    for s in SYSTEMS:
        r = rows[rows.coding_system == s]
        meta = pd.read_parquet(ART / "index" / f"{s}_meta.parquet")
        oof = pd.read_parquet(ART / "cv" / f"{OOF}{s}.parquet")[["row", "code", "label", "ltr"]]
        top = oof.sort_values(["row", "ltr", "code"], ascending=[True, False, True]).drop_duplicates("row")
        pred = top.set_index("row")["code"].reindex(r.index)
        in_pool = oof.groupby("row")["label"].max().reindex(r.index).fillna(0).astype(bool)
        df = pd.DataFrame({"term": r["term_norm"], "gold": r["code"], "pred": pred.fillna(""), "fold": r["fold"],
                           "in_pool": in_pool})
        df["correct"] = df["gold"] == df["pred"]
        seen = {f: set(r[r.fold != f]["code"]) for f in r["fold"].unique()}
        df["seen"] = [g in seen[f] for g, f in zip(df["gold"], df["fold"])]
        df["gold_title"] = [title(meta, c, s) for c in df["gold"]]
        df["pred_title"] = [title(meta, c, s) for c in df["pred"]]
        err = df[~df.correct].copy()
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(
            pd.concat([err["gold_title"], err["pred_title"]]))
        a, b = vec.transform(err["gold_title"]), vec.transform(err["pred_title"])
        err["char_cos"] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
        err["not_in_pool"] = ~err["in_pool"]
        err["near_twin"] = err["in_pool"] & (err["char_cos"] >= 0.5)
        err["abbreviation"] = [any(t in abbr for t in tokens(x)) for x in err["term"]]
        err["unseen_code"] = ~err["seen"]
        n, ne, n_in = len(df), len(err), int(err["in_pool"].sum())
        res = {"rows": n, "top1": round(100 * df.correct.mean(), 1), "errors": ne,
               "seen": {"n": int(df.seen.sum()), "acc": round(100 * df[df.seen].correct.mean(), 1)},
               "unseen": {"n": int((~df.seen).sum()), "acc": round(100 * df[~df.seen].correct.mean(), 1)},
               "pool_ceiling": round(100 * df.in_pool.mean(), 1), "classes": {}}
        if s == ICD:
            res["partial_same_3char"] = round(100 * (df.gold.str[:3] == df.pred.str[:3]).mean(), 1)
            ch = meta["chapter"]
            res["partial_same_chapter"] = round(100 * np.mean([ch.get(g) == ch.get(p) for g, p in zip(df.gold, df.pred)]), 1)
        else:
            res["partial_same_semantic_tag"] = round(100 * np.mean([tag(meta, g) == tag(meta, p) and tag(meta, g) != ""
                                                                    for g, p in zip(df.gold, df.pred)]), 1)
        # label-noise ceiling: identical normalised terms
        agree = tot = 0
        n_rows_grp = 0
        for _, g in df[df.term != ""].groupby("term"):
            if len(g) < 2:
                continue
            n_rows_grp += len(g)
            for x, y in combinations(g["gold"].tolist(), 2):
                tot += 1
                agree += x == y
        res["identical_term_pair_agreement"] = {"pairs": tot, "rows": n_rows_grp,
                                                "same_code_pct": round(100 * agree / max(tot, 1), 1)}
        md += [f"## {s}", "",
               f"Rows {n:,}; LambdaRank top-1 {res['top1']}; errors {ne:,}; pool ceiling {res['pool_ceiling']}%.", "",
               "| class | errors | % of errors | note |", "|---|---|---|---|"]
        notes = {"not_in_pool": "retrieval / indexing miss: no reranker can fix",
                 "near_twin": f"in pool, char-cosine >= 0.5 ({round(100 * err.near_twin.sum() / max(n_in, 1), 1)}% of in-pool errors)",
                 "abbreviation": "term contains a mined abbreviation",
                 "unseen_code": "gold code absent from the training folds"}
        for k in notes:
            c = int(err[k].sum())
            res["classes"][k] = {"errors": c, "pct_of_errors": round(100 * c / max(ne, 1), 1)}
            md.append(f"| {k} | {c:,} | {res['classes'][k]['pct_of_errors']} | {notes[k]} |")
        res["classes"]["near_twin"]["pct_of_in_pool_errors"] = round(100 * err.near_twin.sum() / max(n_in, 1), 1)
        lp = res["identical_term_pair_agreement"]
        md += ["", f"Label-noise ceiling: {lp['rows']:,} rows share their normalised term with another row; of their "
               f"{lp['pairs']:,} row pairs, **{lp['same_code_pct']}%** carry the same gold code.", ""]
        for k in notes:
            ex = err[err[k]].head(N_EX)
            md += [f"### {k}: {N_EX} examples", "", "| term | gold | predicted |", "|---|---|---|"]
            for _, e in ex.iterrows():
                cells = [e.term[:50], f"{e.gold} {e.gold_title[:45]}", f"{e.pred} {e.pred_title[:45]}"]
                md.append("| " + " | ".join(c.replace("|", "/") for c in cells) + " |")
            md.append("")
        out[s] = res
    (REP / f"error_analysis{SUFFIX}.json").write_text(json.dumps(out, indent=1))
    (REP / f"error_analysis{SUFFIX}.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
