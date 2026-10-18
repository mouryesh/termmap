"""Step 13 figure: share of out-of-fold errors per failure tag, both systems."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import guard  # noqa: E402

r = json.loads((Path(__file__).parents[1] / "artifacts/cv/error_analysis.json").read_text())
tags = list(r["tags"]["ICD10CM"].keys())
fig, ax = plt.subplots(figsize=(10, 5))
y = np.arange(len(tags))
for i, s in enumerate(["ICD10CM", "SNOMEDCT_US"]):
    v = [100 * r["tags"][s][t] / r["errors"][s] for t in tags]
    ax.barh(y + (i - 0.5) * 0.4, v, 0.4, label=f"{s} ({r['errors'][s]} errors)")
ax.set_yticks(y, tags); ax.invert_yaxis(); ax.set_xlabel("% of that system's errors"); ax.legend()
ax.set_title("Out-of-fold errors by failure mode (one tag per error)"); fig.tight_layout()
top = {s: max(r["tags"][s], key=r["tags"][s].get) for s in r["tags"]}
guard.fig(13, "error_tags", fig,
          f"Largest error sources: ICD '{top['ICD10CM']}' and SNOMED '{top['SNOMEDCT_US']}'. Retrieval misses "
          f"({r['tags']['SNOMEDCT_US']['not retrieved']} SNOMED rows) and abbreviations/unseen codes dominate SNOMED; "
          "ICD errors are mostly specificity (under/over-specific) and abbreviations, which better qualifier "
          "handling or abbreviation expansion could address.")
