"""M3 figures: accuracy by stage vs ceilings, and LambdaRank feature importance (gain)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import guard  # noqa: E402

ROOT = Path(__file__).parents[1]
ce = json.loads((ROOT / "artifacts/cv/ce_summary.json").read_text())
lt = json.loads((ROOT / "artifacts/cv/ltr_summary.json").read_text())
systems = ["ICD10CM", "SNOMEDCT_US"]
stages = {
    "fused retrieval": [ce[s]["fused_top1"] for s in systems],
    "cross-encoder": [ce[s]["ce_top1_with_context"] for s in systems],
    "LambdaRank w/o CE": [lt[s]["no cross-encoder"]["top1"] for s in systems],
    "LambdaRank (full)": [lt[s]["entity+chunk"]["top1"] for s in systems],
}
fig, ax = plt.subplots(figsize=(9, 5))
fig.subplots_adjust(bottom=0.2, right=0.85)
x = np.arange(len(systems))
w = 0.2
for i, (name, vals) in enumerate(stages.items()):
    b = ax.bar(x + (i - 1.5) * w, vals, w, label=name)
    ax.bar_label(b, fmt="%.1f", fontsize=8)
for j, s in enumerate(systems):
    ax.hlines(lt[s]["pool_ceiling"], j - 2 * w, j + 2 * w, colors="k", linestyles="--", lw=1)
    ax.text(j + 2 * w, lt[s]["pool_ceiling"], f" pool ceiling {lt[s]['pool_ceiling']}", va="center", fontsize=8)
ax.set_xticks(x, systems); ax.set_ylabel("top-1 accuracy % (3-fold OOF, entity+chunk)"); ax.set_ylim(0, 100)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8); ax.set_title("Exact-match accuracy by pipeline stage")
guard.fig(12, "stage_accuracy", fig,
          f"Each stage adds accuracy: ICD {stages['fused retrieval'][0]} -> {stages['cross-encoder'][0]} -> "
          f"{stages['LambdaRank (full)'][0]}%, SNOMED {stages['fused retrieval'][1]} -> {stages['cross-encoder'][1]} -> "
          f"{stages['LambdaRank (full)'][1]}%. The dashed line is the most any reranker could reach with these "
          "candidates, so the remaining gap splits into ranking errors (below the line) and retrieval misses (above it).")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, s in zip(axes, systems):
    imp = pd.read_csv(ROOT / f"reports/feature_importance_{s}.csv").head(15)[::-1]
    ax.barh(imp["feature"], imp["gain"] / imp["gain"].sum() * 100)
    ax.set_title(f"{s}: top-15 features (% of gain)"); ax.set_xlabel("% of total gain (top 15)")
fig.tight_layout()
top = {s: pd.read_csv(ROOT / f"reports/feature_importance_{s}.csv").iloc[0]["feature"] for s in systems}
guard.fig(12, "feature_importance", fig,
          f"Cross-encoder features dominate both rankers (top: {top['ICD10CM']} / {top['SNOMEDCT_US']}); retrieval "
          "ranks and code metadata (is_leaf, chapter, semantic tag, training frequency) come next. The qualifier "
          "features carry little gain, so specificity errors are not yet being fixed by them.")
