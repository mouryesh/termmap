"""M5 figure: accuracy versus coverage (most-confident rows first), 3-fold OOF, both systems."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import guard  # noqa: E402

r = json.loads((Path(__file__).parents[1] / "artifacts/cv/m5_summary.json").read_text())["confidence"]
fig, ax = plt.subplots(figsize=(7, 4.5))
for s, c in r.items():
    ax.plot([p["coverage"] * 100 for p in c["curve"]], [p["accuracy"] * 100 for p in c["curve"]], marker="o", ms=3,
            label=f"{s} (ECE {c['ece']:.3f}; 98% precision at {c['coverage_at_98']}% coverage)")
ax.axhline(98, color="grey", ls=":", lw=1)
ax.set_xlabel("coverage % (rows answered, most confident first)"); ax.set_ylabel("accuracy % of answered rows")
ax.set_title("Accuracy vs coverage (temperature-scaled LambdaRank probability)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
i, s_ = r["ICD10CM"], r["SNOMEDCT_US"]
q = lambda c, cov: next(p["accuracy"] for p in c["curve"] if p["coverage"] >= cov) * 100
guard.fig(12, "coverage_curve", fig,
          f"Confidence ranks rows well: the most confident 25% are right {q(i, .25):.0f}% (ICD) / {q(s_, .25):.0f}% "
          f"(SNOMED) of the time versus {q(i, 1.0):.0f}% / {q(s_, 1.0):.0f}% overall, but 98% precision is only reached on "
          f"{i['coverage_at_98']}% / {s_['coverage_at_98']}% of rows, so nearly every row carries the low-confidence flag.")
