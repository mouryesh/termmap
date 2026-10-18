"""Steps 1-3 (inspection, EDA): profile the long-format training data and draw the EDA figures.

Run: .venv/bin/python ml_pipeline/01_eda.py  (aggregate statistics only; writes figures to ml_pipeline/figures).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import guard  # noqa: E402

from termmap.config import load_config
from termmap.dictionaries import load_icd, load_snomed
from termmap.io import load_train, group_key
from termmap.normalize import level1

cfg = load_config()
train = load_train(cfg["paths"]["train"])
icd = load_icd(cfg["paths"]["icd_dict"], cfg["paths"]["artifacts"])
sno = load_snomed(cfg["paths"]["snomed_dict"], cfg["paths"]["artifacts"])

# in-memory only: never written to disk
df = train.copy()
df["group"] = [group_key(c, i) for i, c in enumerate(df["chunk_text"])]
df["term_words"] = df["entity"].map(lambda s: len(level1(s).split()))
df["chunk_chars"] = df["chunk_text"].str.len()
prof = guard.profile(df[["entity", "coding_system", "code", "term_words", "chunk_chars", "group"]],
                     target="code", group_col="group")
guard.eda_figures(df[["coding_system", "code", "term_words", "chunk_chars"]], target="code")

# long tail of gold codes
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, s in zip(axes, ["ICD10CM", "SNOMEDCT_US"]):
    vc = df.loc[df.coding_system == s, "code"].value_counts().values
    ax.loglog(range(1, len(vc) + 1), vc, ".")
    ax.set_title(f"{s}: rows per gold code ({len(vc)} codes)")
    ax.set_xlabel("code rank"); ax.set_ylabel("rows")
n_icd_single = (df[df.coding_system == "ICD10CM"].code.value_counts() == 1).sum()
n_sno_single = (df[df.coding_system == "SNOMEDCT_US"].code.value_counts() == 1).sum()
guard.fig(2, "code_long_tail", fig,
          f"Gold codes follow a long tail: {n_icd_single} ICD and {n_sno_single} SNOMED codes appear exactly once. "
          "A classifier over seen codes cannot reach the brief's bar, so retrieval over the whole dictionary is required.")

# SNOMED semantic tags and ICD chapters of gold codes
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
tags = df.loc[df.coding_system == "SNOMEDCT_US", "code"].map(sno.meta["semantic_tag"]).value_counts().head(10)
tags[::-1].plot.barh(ax=axes[0]); axes[0].set_title("SNOMED gold semantic tags (top 10)")
ch = df.loc[df.coding_system == "ICD10CM", "code"].str[0].value_counts().head(12)
ch.plot.bar(ax=axes[1]); axes[1].set_title("ICD gold chapter (first letter)")
guard.fig(2, "tags_and_chapters", fig,
          f"SNOMED gold codes span many tags (top: {tags.index[0]} {tags.iloc[0]}, {tags.index[1]} {tags.iloc[1]}), "
          f"so the tag is a ranking feature, not a filter. ICD is dominated by chapter R ({ch.get('R', 0)} rows): symptoms coded as symptoms.")

# term length
fig, ax = plt.subplots(figsize=(6, 4))
df["term_words"].clip(upper=15).plot.hist(bins=15, ax=ax)
ax.set_title("Words per normalized term (clipped at 15)"); ax.set_xlabel("words")
guard.fig(2, "term_length", fig,
          f"Terms are short (median {df.term_words.median():.0f} words), so abbreviations and spelling noise carry "
          "most of the difficulty; chunk context is the main extra signal.")
print("profile traits:", prof.get("traits"))
print("leakage suspects:", prof.get("leakage_suspects"))
