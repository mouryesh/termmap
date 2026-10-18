"""Build artifacts_v2/kaggle/<job>/: code plus the minimum data one Kaggle job needs.

Usage: python kaggle/make_bundle.py k1
Uploaded as a PRIVATE Kaggle dataset (user-approved; deleted after the deadline, see README).
Rows carry only the columns the job uses.
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts_v2"
sys.path.insert(0, str(ROOT / "src"))


def k1(out: Path) -> None:
    from termmap.llm_rewrite import unique_pairs
    unique_pairs(pd.read_parquet(ART / "cv" / "rows.parquet")).to_parquet(out / "rewrite_pairs.parquet", index=False)


def k2(out: Path) -> None:
    cols = ["coding_system", "term_norm", "display_norm", "code", "fold", "chunk_text"]
    pd.read_parquet(ART / "cv" / "rows.parquet")[cols].to_parquet(out / "rows.parquet")  # index kept
    for s in ("ICD10CM", "SNOMEDCT_US"):
        pd.read_parquet(ART / "index" / f"{s}_table.parquet").to_parquet(out / f"{s}_table.parquet", index=False)
        pd.read_parquet(ART / "index" / f"{s}_meta.parquet")[["title"]].to_parquet(out / f"{s}_meta.parquet")


JOBS = {"k1": k1, "k2": k2}

if __name__ == "__main__":
    job = sys.argv[1]
    out = ART / "kaggle" / job
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    JOBS[job](out)
    # Kaggle auto-extracts .zip uploads, so the code archive uses another extension.
    with zipfile.ZipFile(out / "termmap_src.bin", "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "configs").glob("*.yaml")):
            z.write(f, f.relative_to(ROOT))
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": f"termmap-{job}", "id": f"mouryeshreddy/termmap-{job}", "licenses": [{"name": "other"}]}, indent=1))
    for f in sorted(out.iterdir()):
        print(f"{f.name:40s} {f.stat().st_size / 2**20:7.2f} MB")
