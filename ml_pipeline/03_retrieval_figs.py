"""Step 9 figure: recall@k of the fused list and each channel, from the M1 CV records."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import guard  # noqa: E402

R = pd.read_parquet(Path(__file__).parents[1] / "artifacts" / "m1_cv_records.parquet")
ks = list(range(1, 101))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, s in zip(axes, ["ICD10CM", "SNOMEDCT_US"]):
    r = R[R.system == s]
    for col, lab in [("fused_rank", "fused (RRF)"), ("rank_dense", "dense SapBERT"), ("rank_memory", "training memory"),
                     ("rank_char", "char TF-IDF"), ("rank_bm25", "BM25")]:
        ax.plot(ks, [100 * (r[col] <= k).mean() for k in ks], label=lab, lw=2.2 if col == "fused_rank" else 1.2)
    ax.axvline(80, color="grey", ls=":", lw=1)
    ax.set_title(f"{s}: gold code within top k (5-fold CV, n={len(r)})")
    ax.set_xlabel("k"); ax.set_xscale("log"); ax.grid(alpha=.3)
axes[0].set_ylabel("recall %"); axes[0].legend(loc="lower right")
icd, sct = R[R.system == "ICD10CM"], R[R.system == "SNOMEDCT_US"]
guard.fig(4, "recall_at_k", fig,
          f"Fused retrieval puts the gold code in the top 80 for {100*(icd.fused_rank<=80).mean():.1f}% of ICD and "
          f"{100*(sct.fused_rank<=80).mean():.1f}% of SNOMED rows, but first only {100*(icd.fused_rank==1).mean():.1f}% / "
          f"{100*(sct.fused_rank==1).mean():.1f}%. The candidates are there; the reranker (M2-M3) must pick the right one, "
          "and these recall values are its ceiling.")
