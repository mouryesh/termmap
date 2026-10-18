"""Pick demo rows by rule from the OUT-OF-FOLD results (no hand-fitting): the first row in file order that
meets each criterion. Writes demo_inputs.csv (entity, chunk_text) and demo_cases.csv (why each row was chosen);
both are git-ignored because they contain training text.

Probabilities: softmax of the submitted (no-cross-encoder) ranker's out-of-fold scores, temperature from noce.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from termmap.io import ICD, SNOMED  # noqa: E402
from termmap.normalize import tokens  # noqa: E402

ART = ROOT / "artifacts"


def oof_table(s: str, rows: pd.DataFrame, T: float, thr: float) -> pd.DataFrame:
    oof = pd.read_parquet(ART / "cv" / f"ltr_oof_noce_{s}.parquet")[["row", "code", "label", "ltr"]]
    oof["p"] = np.exp((oof["ltr"] - oof.groupby("row")["ltr"].transform("max")) / T)
    oof["p"] /= oof.groupby("row")["p"].transform("sum")
    top = oof.sort_values(["row", "ltr", "code"], ascending=[True, False, True]).drop_duplicates("row").set_index("row")
    r = rows[rows.coding_system == s].copy()
    r["pred"], r["prob"] = top["code"].reindex(r.index), top["p"].reindex(r.index)
    r["correct"] = r["pred"] == r["code"]
    r["low_conf"] = r["prob"] < thr
    seen = {f: set(r[r.fold != f]["code"]) for f in r["fold"].unique()}
    r["unseen"] = [c not in seen[f] for c, f in zip(r["code"], r["fold"])]
    return r


def main() -> None:
    rows = pd.read_parquet(ART / "cv" / "rows.parquet")
    noce = json.loads((ART / "model" / "noce.json").read_text())  # submitted ranker (no cross-encoder)
    calib = {s: {"T": v["T"]} for s, v in noce.items()}
    conf = {s: {"threshold": v["threshold"]} for s, v in noce.items()}
    abbr = set(json.loads((ART / "model" / "abbreviations.json").read_text()))
    meta = pd.read_parquet(ART / "index" / f"{SNOMED}_meta.parquet")["title"]
    sn = oof_table(SNOMED, rows, calib[SNOMED]["T"], conf[SNOMED]["threshold"])
    ic = oof_table(ICD, rows, calib[ICD]["T"], conf[ICD]["threshold"])
    sn = sn[sn.chunk_text.str.len().between(20, 400)]  # readable on screen
    ic = ic[ic.chunk_text.str.len().between(20, 400)]
    err = sn[~sn.correct & sn.pred.notna()].copy()
    g, p = err["code"].map(meta).fillna("").str.lower(), err["pred"].map(meta).fillna("").str.lower()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(pd.concat([g, p]))
    err["cos"] = np.asarray(vec.transform(g).multiply(vec.transform(p)).sum(axis=1)).ravel()
    is_abbr = sn["term_norm"].map(lambda t: any(x in abbr for x in tokens(t)) and len(t) <= 8)
    picks = [
        ("easy hit (ICD, confident, correct)", ic[ic.correct & (ic.prob > 0.9)]),
        ("easy hit (SNOMED, confident, correct)", sn[sn.correct & (sn.prob > 0.9)]),
        ("abbreviation, correct", sn[is_abbr & sn.correct]),
        ("abbreviation, wrong", sn[is_abbr & ~sn.correct]),
        ("near-twin error (char-cosine >= 0.5)", err[err.cos >= 0.5]),
        ("abstained: low confidence", sn[sn.low_conf & (sn.prob < 0.3)]),
        ("long-tail code (unseen in training folds), correct", sn[sn.unseen & sn.correct]),
        ("long-tail code (unseen in training folds), ICD correct", ic[ic.unseen & ic.correct]),
    ]
    out, used = [], set()
    for why, cand in picks:
        cand = cand[~cand["entity"].isin(used)]
        if cand.empty:
            continue
        r = cand.iloc[0]
        used.add(r["entity"])
        out.append({"case": why, "entity": r["entity"], "chunk_text": r["chunk_text"], "system": r["coding_system"],
                    "gold": r["code"], "oof_pred": r["pred"], "oof_prob": round(float(r["prob"]), 3)})
    cases = pd.DataFrame(out)
    cases[["entity", "chunk_text"]].to_csv(ROOT / "demo_inputs.csv", index=False)
    cases.to_csv(ROOT / "demo_cases.csv", index=False)
    print(cases.drop(columns="chunk_text").to_string())


if __name__ == "__main__":
    main()
