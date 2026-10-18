"""Upload / download the trained TermMap artifacts on the Hugging Face Hub.

  hf auth login                                   # once, with YOUR token (never put it in code or .env.example)
  python scripts/hf_weights.py upload             # artifacts/model (42 MB) -> HF model repo (private)
  python scripts/hf_weights.py upload --with-index  # also artifacts/index (2.4 GB; else rebuild: termmap build-index)
  python scripts/hf_weights.py download           # evaluators: HF -> artifacts/ (then `termmap predict ...`)

Repo id: env TERMMAP_HF_REPO (default Vorun/termmap-weights). The token is read by huggingface_hub from its own
login cache or the HF_TOKEN environment variable; this script never handles it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
REPO = os.environ.get("TERMMAP_HF_REPO", "Vorun/termmap-weights")


def upload() -> None:
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
    subs = ("model", "index") if "--with-index" in sys.argv else ("model",)
    for sub in subs:  # without index/, evaluators run `termmap build-index` (manifest must not exist then)
        api.upload_folder(folder_path=str(ROOT / "artifacts" / sub), path_in_repo=sub, repo_id=REPO,
                          repo_type="model", ignore_patterns=["*.log"])
    print(f"uploaded to https://huggingface.co/{REPO}")


def download() -> None:
    snapshot_download(REPO, repo_type="model", local_dir=str(ROOT / "artifacts"))
    print(f"downloaded {REPO} -> {ROOT / 'artifacts'}")


if __name__ == "__main__":
    {"upload": upload, "download": download}[sys.argv[1]]()
