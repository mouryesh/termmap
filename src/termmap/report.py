"""Writes reports/validation.md from evaluation results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _table(rows: dict[str, dict], cols: list[str]) -> str:
    head = "| setting | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    body = "".join(f"| {k} | " + " | ".join(str(v.get(c, "")) for c in cols) + " |\n" for k, v in rows.items())
    return head + body


def write_m1(path: Path, main: dict, ablations: dict[str, dict], latency: dict, baselines: dict) -> None:
    cols = ["n", "R@1", "R@10", "R@30", "R@50", "R@80", "pool_recall@50+sib",
            "bm25_R@50", "char_R@50", "dense_R@50", "memory_R@50"]
    lines = ["# TermMap validation report\n",
             "All numbers: 5-fold GroupKFold by in-memory canonical chunk text (training memory and "
             "abbreviation table rebuilt from training folds only). Long-tail = 20% of gold codes held out "
             "entirely (no row of those codes, or sharing their chunk, in memory).\n",
             "## M1: Stage 0 + Stage 2 (fused retrieval top-1)\n",
             f"Headline min(ICD, SNOMED) top-1: **{main['cv']['min_top1']}%**\n",
             "### Cross-validation\n", _table({s: main["cv"][s] for s in ("ICD10CM", "SNOMEDCT_US")}, cols),
             "\n### Long-tail (held-out codes)\n", _table(main["long_tail"], cols),
             "\n### Baselines (same folds)\n", _table(baselines, ["ICD10CM", "SNOMEDCT_US"]),
             "\n### Ablations (CV top-1 / R@50)\n"]
    abl = {}
    for name, res in ablations.items():
        abl[name] = {s: f"{res['cv'][s]['R@1']} / {res['cv'][s]['R@50']}" for s in ("ICD10CM", "SNOMEDCT_US")}
    lines.append(_table(abl, ["ICD10CM", "SNOMEDCT_US"]))
    lines.append("\nentity + chunk_text and entity-only are identical at M1 (retrieval uses the term only; "
                 "context enters at the cross-encoder in M2).\n")
    lines.append(f"\n### Latency (CPU/GPU as run, single-term calls, warm)\n{latency}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
