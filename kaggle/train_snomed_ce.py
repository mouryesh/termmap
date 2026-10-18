# TermMap - SNOMED cross-encoder training on Kaggle (private notebook, GPU, Internet ON).
# Input: private dataset mouryeshreddy/termmap-snomed-ce built by kaggle/make_bundle.py.
# Output: /kaggle/working/snomed_ce_outputs.zip (out-of-fold scores, histories, final model).
import glob
import json
import logging
import os
import shutil
import subprocess
import sys
import zipfile

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.46.3"], check=True)


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]


WORK = "/kaggle/working/termmap"
os.makedirs(WORK, exist_ok=True)
zipfile.ZipFile(find("termmap_src.bin")).extractall(WORK)
art = f"{WORK}/artifacts"
os.makedirs(f"{art}/cv", exist_ok=True)
os.makedirs(f"{art}/index", exist_ok=True)
shutil.copy(find("rows_snomed.parquet"), f"{art}/cv/rows.parquet")
shutil.copy(find("candidates_SNOMEDCT_US.parquet"), f"{art}/cv/candidates_SNOMEDCT_US.parquet")
shutil.copy(find("SNOMEDCT_US_table.parquet"), f"{art}/index/SNOMEDCT_US_table.parquet")
shutil.copy(find("SNOMEDCT_US_meta.parquet"), f"{art}/index/SNOMEDCT_US_meta.parquet")

sys.path.insert(0, f"{WORK}/src")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)
from termmap.config import load_config, set_seeds  # noqa: E402

cfg = load_config(f"{WORK}/configs/default.yaml")
set_seeds(cfg["seed"])
import torch  # noqa: E402

print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
from termmap.rerank.train_ce import run  # noqa: E402

summary = run(cfg, systems=("SNOMEDCT_US",), final=True)
print(json.dumps(summary, indent=1))

out = "/kaggle/working/snomed_ce_outputs.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(f"{art}/cv/ce_SNOMEDCT_US*") + [f"{art}/cv/ce_summary.json"]):
        z.write(f, f"cv/{os.path.basename(f)}")
    for f in sorted(glob.glob(f"{art}/model/ce_SNOMEDCT_US/*")):
        z.write(f, f"model/ce_SNOMEDCT_US/{os.path.basename(f)}")
shutil.rmtree(WORK)  # leave only the outputs zip in /kaggle/working
print("done:", out, os.path.getsize(out) // 2**20, "MB")
