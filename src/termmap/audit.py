"""M0 data audit: reproduce the measured facts the design rests on. Aggregate statistics only."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from termmap.config import load_config
from termmap.dictionaries import load_icd, load_snomed
from termmap.io import ICD, SNOMED, group_key, load_train, pair_instances
from termmap.normalize import level1, tokens

LATERAL = re.compile(r"\b(?:left|right|bilateral|lt|rt)\b")
LATERAL_DESC = re.compile(r"\b(?:left|right|bilateral)\b")
NEG = re.compile(r"^(?:no|negative|neg|ø|denies|without)\b")
EMBED = re.compile(r"\b([A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?)\b", re.I)
SEM_OK = {"disorder", "finding", "situation", "morphologic abnormality", "event"}


def pct(a: float, b: float) -> float:
    return round(100.0 * a / b, 1) if b else 0.0


def run(cfg_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    train = load_train(cfg["paths"]["train"])
    icd = load_icd(cfg["paths"]["icd_dict"], cfg["paths"]["artifacts"])
    sno = load_snomed(cfg["paths"]["snomed_dict"], cfg["paths"]["artifacts"])
    r: dict = {}

    # --- dictionaries
    r["dict"] = {
        "icd_rows": len(icd.rows), "icd_codes": len(icd.codes),
        "icd_parent_codes": int((~icd.meta["is_leaf"]).sum()),
        "snomed_rows": len(sno.rows), "snomed_concepts": len(sno.codes),
        "snomed_top_tags": sno.meta["semantic_tag"].value_counts().head(12).to_dict(),
    }

    # --- pairing
    inst = pair_instances(train)
    has_i, has_s = inst["icd_code"].notna(), inst["snomed_code"].notna()
    r["rows"] = len(train)
    r["rows_by_system"] = train["coding_system"].value_counts().to_dict()
    r["instances"] = {"total": len(inst), "both": int((has_i & has_s).sum()),
                      "snomed_only": int((~has_i & has_s).sum()), "icd_only": int((has_i & ~has_s).sum())}
    multi = train.groupby(["entity", "chunk_text", "coding_system"]).size()
    r["instances_with_multiple_codes_same_system"] = int((multi > 1).sum())

    # subconcept is used ONLY here, to reproduce the brief's descriptive figures.
    sub = train.groupby(["entity", "chunk_text"])["subconcept"].agg(lambda s: "DiagnosisCode" in set(s))
    inst = inst.merge(sub.rename("is_dx").reset_index(), on=["entity", "chunk_text"], how="left")
    dx = inst[inst["is_dx"]]
    r["dx_instances"] = {"total": len(dx), "both": int((dx.icd_code.notna() & dx.snomed_code.notna()).sum()),
                         "snomed_only": int((dx.icd_code.isna() & dx.snomed_code.notna()).sum()),
                         "icd_only": int((dx.icd_code.notna() & dx.snomed_code.isna()).sum())}
    r["subconcept_counts_rows"] = train["subconcept"].value_counts().to_dict()

    # --- term and code repetition
    train["term_norm"] = train["entity"].map(level1)
    out = {}
    for sysname in (ICD, SNOMED):
        t = train[train.coding_system == sysname].reset_index(drop=True)
        code_n = t["code"].map(t["code"].value_counts())
        term_n = t["term_norm"].map(t["term_norm"].value_counts())
        # LOO exact-term lookup: another row with the same normalized term and its majority code
        loo_hit = 0
        by_term = t.groupby("term_norm")["code"].apply(list).to_dict()
        for term, code in zip(t["term_norm"], t["code"]):
            others = list(by_term[term]); others.remove(code)
            if others and Counter(others).most_common(1)[0][0] == code:
                loo_hit += 1
        out[sysname] = {"rows": len(t), "distinct_codes": t["code"].nunique(),
                        "pct_rows_code_seen_elsewhere": pct((code_n > 1).sum(), len(t)),
                        "pct_rows_term_seen_elsewhere": pct((term_n > 1).sum(), len(t)),
                        "loo_exact_term_accuracy_pct": pct(loo_hit, len(t)),
                        "singleton_codes": int((t["code"].value_counts() == 1).sum())}
    r["repetition"] = out

    # --- invalid codes (label noise)
    valid = {ICD: icd.codes, SNOMED: sno.codes}
    inval = train[[c not in valid[s] for s, c in zip(train.coding_system, train.code)]]
    r["invalid_codes"] = {"rows": len(inval), "by_system": inval["coding_system"].value_counts().to_dict(),
                          "examples": inval[["coding_system", "code"]].drop_duplicates().head(10).values.tolist()}

    # --- exact match of normalized term to a normalized description
    exact = {}
    for sysname, d in ((ICD, icd), (SNOMED, sno)):
        dn = pd.DataFrame({"n": d.rows["description"].map(level1), "code": d.rows["code"]})
        lookup = dn.groupby("n")["code"].apply(lambda s: sorted(set(s))).to_dict()
        t = train[(train.coding_system == sysname) & train.code.isin(d.codes)]
        m = t["term_norm"].map(lookup)
        hit = m.notna()
        any_right = sum(c in cs for c, cs in zip(t.code[hit], m[hit]))
        uniq = hit & m.map(lambda x: isinstance(x, list) and len(x) == 1)
        uniq_right = sum(cs[0] == c for c, cs in zip(t.code[uniq], m[uniq]))
        exact[sysname] = {"pct_terms_with_exact_match": pct(hit.sum(), len(t)),
                          "pct_gold_among_matched": pct(any_right, hit.sum()),
                          "pct_correct_when_unique_match": pct(uniq_right, uniq.sum())}
    r["exact_match"] = exact

    # --- ICD gold code profile
    ti = train[(train.coding_system == ICD) & train.code.isin(icd.codes)].copy()
    all_desc = icd.rows.assign(d=icd.rows["description"].str.lower()).groupby("code")["d"].apply(" | ".join)
    desc = ti["code"].map(all_desc).fillna("")  # any description of the code
    lat_terms = ti["term_norm"].str.contains(LATERAL)
    r["icd_gold"] = {
        "pct_unspecified": pct(desc.str.contains("unspecified").sum(), len(ti)),
        "pct_other": pct(desc.str.contains(r"\bother\b").sum(), len(ti)),
        "pct_chapter_R": pct((ti["code"].str[0] == "R").sum(), len(ti)),
        "rows_parent_gold": int((~ti.code.map(icd.meta["is_leaf"])).sum()),
        "distinct_parent_gold_codes": int(ti.loc[~ti.code.map(icd.meta["is_leaf"]), "code"].nunique()),
        "pct_rows_parent_gold": pct((~ti.code.map(icd.meta["is_leaf"])).sum(), len(ti)),
        "parent_gold_examples": sorted(ti.loc[~ti.code.map(icd.meta["is_leaf"]), "code"].unique())[:10],
        "terms_naming_side": int(lat_terms.sum()),
        "pct_side_terms_with_sideless_gold": pct((lat_terms & ~desc.str.contains(LATERAL_DESC)).sum(), lat_terms.sum()),
        "top_chapters": ti["code"].str[0].value_counts().head(8).to_dict(),
    }

    # --- SNOMED tags of diagnosis rows (subconcept used descriptively only)
    ts = train[(train.coding_system == SNOMED) & (train.subconcept == "DiagnosisCode") & train.code.isin(sno.codes)]
    tags = ts["code"].map(sno.meta["semantic_tag"])
    r["snomed_dx_tags"] = {"rows": len(ts), "top": (tags.value_counts(normalize=True) * 100).round(1).head(8).to_dict(),
                           "pct_in_core5": pct(tags.isin(SEM_OK).sum(), len(ts))}
    tall = train[(train.coding_system == SNOMED) & train.code.isin(sno.codes)]["code"].map(sno.meta["semantic_tag"])
    r["snomed_all_tags"] = (tall.value_counts(normalize=True) * 100).round(1).head(10).to_dict()

    # --- chunks
    chunks = inst["chunk_text"]
    keys = [group_key(c, i) for i, c in enumerate(chunks)]
    per_chunk = pd.Series(keys).value_counts()  # entities per canonical chunk
    r["chunks"] = {"distinct_raw": int(chunks.nunique()), "distinct_canonical": len(set(keys)),
                   "empty": int((chunks.str.strip() == "").sum()),
                   "pct_chunks_one_entity": pct((per_chunk == 1).sum(), len(per_chunk)),
                   "paired_rows_share_chunk": True}

    # --- noise
    vocab = set()
    for d in (icd, sno):
        for s in d.rows["description"]:
            vocab.update(tokens(level1(s)))
    ents = inst["entity"]
    en = ents.map(level1)
    oov = en.map(lambda s: any(w not in vocab and not w.isdigit() for w in tokens(s)))
    emb = ents.map(lambda s: any(m.upper() in icd.codes for m in EMBED.findall(s)))
    r["noise"] = {"pct_terms_with_oov_word": pct(oov.sum(), len(ents)),
                  "pct_terms_with_br": pct(ents.str.contains("<br", case=False).sum(), len(ents)),
                  "terms_with_hcc": int(ents.str.contains(r"\(hcc\)", case=False).sum()),
                  "terms_starting_negation": int(en.str.contains(NEG).sum()),
                  "terms_with_embedded_icd_code": int(emb.sum()),
                  "term_words_median": float(en.map(lambda s: len(s.split())).median()),
                  "terms_over_32_words": int((en.map(lambda s: len(s.split())) > 32).sum()),
                  "vocab_size": len(vocab)}

    rep = Path(cfg["paths"]["reports"]); rep.mkdir(parents=True, exist_ok=True)
    (rep / "data_audit.json").write_text(json.dumps(r, indent=2, default=str))
    return r


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
