"""Build artifacts/kaggle_upload/: the minimum needed to train the SNOMED cross-encoder on Kaggle.

Uploaded as a PRIVATE Kaggle dataset (user-approved 2026-10-17; to be deleted after training).
Rows carry only the columns the cross-encoder uses (normalized term, chunk text, gold code,
fold); the row index is preserved.
"""
import json
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ART / "kaggle_upload"
DATASET = "mouryeshreddy/termmap-snomed-ce"

if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)
rows = pd.read_parquet(ART / "cv" / "rows.parquet")
rows = rows[rows.coding_system == "SNOMEDCT_US"][["coding_system", "term_norm", "chunk_text", "code", "fold"]]
rows.to_parquet(OUT / "rows_snomed.parquet")  # index kept: candidate 'row' ids refer to it
shutil.copy(ART / "cv" / "candidates_SNOMEDCT_US.parquet", OUT)
shutil.copy(ART / "index" / "SNOMEDCT_US_table.parquet", OUT)
shutil.copy(ART / "index" / "SNOMEDCT_US_meta.parquet", OUT)
# Kaggle auto-extracts .zip uploads, so the code archive uses another extension.
with zipfile.ZipFile(OUT / "termmap_src.bin", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted((ROOT / "src").rglob("*.py")):
        z.write(f, f.relative_to(ROOT))
    z.write(ROOT / "configs" / "default.yaml", "configs/default.yaml")
(OUT / "dataset-metadata.json").write_text(json.dumps(
    {"title": "termmap-snomed-ce", "id": DATASET, "licenses": [{"name": "other"}]}, indent=1))
for f in sorted(OUT.iterdir()):
    print(f"{f.name:40s} {f.stat().st_size / 2**20:7.1f} MB")
