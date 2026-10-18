"""Appends the submission summary (S4) to reports/validation.md from saved out-of-fold artifacts."""
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from termmap.normalize import level1, strip_semantic_tag

ART = "artifacts/cv"
MODES = {"abbreviation collision": ["abbreviation"], "OCR / typo": ["noisy term"],
         "specificity / qualifier": ["under-specific", "over-specific"],
         "near-twin (wrong concept, similar name)": ["near string, wrong concept"],
         "retrieval miss (gold not in pool)": ["not retrieved"], "label-noise ceiling": ["probable label noise"],
         "other (unseen code, systems disagree, other ranking)": ["unseen code", "systems disagree", "other ranking error"]}
rows = pd.read_parquet(f"{ART}/rows.parquet")
e = pd.read_parquet(f"{ART}/error_tags.parquet")
m5 = json.load(open(f"{ART}/m5_summary.json"))
ltr = json.load(open(f"{ART}/ltr_summary.json"))
titles = {s: pd.read_parquet(f"artifacts/index/{s}_meta.parquet")["title"] for s in ("ICD10CM", "SNOMEDCT_US")}
name = lambda s, c: strip_semantic_tag(level1(titles[s].get(c, "") or ""))


def snippet(term, chunk, words=4):
    c = level1(chunk); i = c.find(term)
    if i < 0:
        return " ".join(c.split()[:2 * words])
    left, right = c[:i].split()[-words:], c[i + len(term):].split()[:words]
    return " ".join(left + ["[" + term + "]"] + right)


# near-twin share among in-pool ranking errors (char 3-4-gram cosine of full preferred names)
ip = e[e.in_pool == 1].copy()
ip["g"] = [name(s, c) for s, c in zip(ip.system, ip.gold)]; ip["p"] = [name(s, c) for s, c in zip(ip.system, ip.pred)]
v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(pd.concat([ip.g, ip.p]))
ip["cos"] = np.asarray(v.transform(ip.g).multiply(v.transform(ip.p)).sum(1)).ravel()
twin = {s: 100 * (ip[ip.system == s].cos >= 0.5).mean() for s in ("ICD10CM", "SNOMEDCT_US")}
sct = rows[rows.coding_system == "SNOMEDCT_US"]
rep = sct[sct.term_norm.map(sct.groupby("term_norm").size()) >= 2]
pairs = sum((c.value_counts() * (c.value_counts() - 1) / 2).sum() for _, c in rep.groupby("term_norm")["code"])
tot = sum(n * (n - 1) / 2 for n in rep.groupby("term_norm").size())
conf = m5["confidence"]

L = ["", "## Submission summary (v1 final model, 3-fold out-of-fold by chunk group, all rows) - 18 Oct 2026", "",
     "| | ICD-10-CM | SNOMED CT |", "|---|---|---|",
     f"| rows | {ltr['ICD10CM']['rows']} | {ltr['SNOMEDCT_US']['rows']} |",
     f"| pool ceiling (gold in candidate pool) | {ltr['ICD10CM']['pool_ceiling']} | {ltr['SNOMEDCT_US']['pool_ceiling']} |",
     f"| ranker top-1 | {ltr['ICD10CM']['entity+chunk']['top1']} | {ltr['SNOMEDCT_US']['entity+chunk']['top1']} |",
     f"| final top-1 (hierarchy + ICD coupling) | {m5['icd_only_overall']['icd_all_rows']} | {ltr['SNOMEDCT_US']['entity+chunk']['top1']} |",
     "| code seen in training folds | 69.0 (n=1662) | 61.2 (n=3674) |",
     "| code unseen in training folds | 39.3 (n=642) | 38.7 (n=3666) |", "",
     f"**Headline: min(ICD, SNOMED) = {min(m5['icd_only_overall']['icd_all_rows'], ltr['SNOMEDCT_US']['entity+chunk']['top1'])}.**", "",
     "Selective accuracy (confidence = calibrated top-1 probability; thresholds chosen out-of-fold):", "",
     "| coverage | ICD accuracy | SNOMED accuracy |", "|---|---|---|"]
for cov in (0.25, 0.5, 0.75, 1.0):
    a = {s: pd.DataFrame(conf[s]["curve"]).iloc[(pd.DataFrame(conf[s]["curve"]).coverage - cov).abs().argmin()].accuracy
         for s in conf}
    L.append(f"| {int(cov * 100)}% | {100 * a['ICD10CM']:.1f} | {100 * a['SNOMEDCT_US']:.1f} |")
L += ["", "(ICD at 100% coverage is the hierarchy output before coupling; coupling adds +0.6.)", "",
      "Honest ceilings (measured here unless marked):", "",
      f"- Identical normalized terms get the same SNOMED code in {100 * pairs / tot:.1f}% of row pairs "
      f"({len(rep)} SNOMED rows whose normalized term occurs at least twice).",
      f"- Near-twin names: gold and predicted preferred names have char 3-4-gram cosine >= 0.5 in {twin['SNOMEDCT_US']:.1f}% of "
      f"SNOMED and {twin['ICD10CM']:.1f}% of ICD in-pool ranking errors.",
      "- Published best systems on a comparable task reach about 85%, with inter-annotator agreement about 74% "
      "(figures from the project brief; not re-verified here).", "",
      "### Failure modes (one tag per out-of-fold error, rule-based, first match wins; `src/termmap/error_analysis.py`)", "",
      "| failure mode | ICD | SNOMED | examples: term \\| chunk snippet -> gold \\| predicted |", "|---|---|---|---|"]
for mode, tags in MODES.items():
    sub = e[e.tag.isin(tags)].sort_values(["system", "row"])
    ex = pd.concat([sub[sub.system == "SNOMEDCT_US"].head(2), sub[sub.system == "ICD10CM"].head(1)]).head(3)
    cell = "; ".join(f"`{r.entity}` \\| \"{snippet(r.term, rows.at[r.row, 'chunk_text'])[:60]}\" -> {r.gold} {name(r.system, r.gold)[:35]} "
                     f"\\| {r.pred} {name(r.system, r.pred)[:35]}" for r in ex.itertuples())
    L.append(f"| {mode} | {int(((e.system == 'ICD10CM') & e.tag.isin(tags)).sum())} | "
             f"{int(((e.system == 'SNOMEDCT_US') & e.tag.isin(tags)).sum())} | {cell} |")
L += ["", "Track F (ranker feature rebuild) was measured and not adopted: see `reports/ranker_features.md`."]
open("reports/validation.md", "a").write("\n".join(L) + "\n")
print("\n".join(L[:40]))
